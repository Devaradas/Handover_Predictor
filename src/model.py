# The actual model: four tier-routed RandomForest experts give a per-second handover probability,
# isotonic regression calibrates those probabilities per arm, and a small gradient-boosted "episode
# gate" then keeps or drops each alarm. I call the whole thing the calibrated cascade.
#
# It tries hard not to leak: train and test never share a drive, train probabilities are out of
# fold, and every threshold and the gate are chosen on the train data only. fit_regime(regime) runs
# the whole thing in one call - train, evaluate on the held-out drives, and save the model.
import os, json
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split, KFold, GroupKFold
import data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "output", "models")
os.makedirs(MODELS, exist_ok=True)

# which expert handles each tier value, and the fixed fire threshold for each expert (found once
# with an F1 sweep, then frozen)
TIER_DISPATCH = {"5g": {1: "e_5g_4g", 0: "e_4g_5g"}, "lte": {1: "e_4g_sub4g", 0: "e_sub4g_4g"}}
TAU = {"e_5g_4g": 0.30, "e_4g_5g": 0.40, "e_4g_sub4g": 0.30, "e_sub4g_4g": 0.30}
LEAD, PP_T, MAXG = data.HORIZON, data.PINGPONG_T, data.MAX_GAP
FEATS = ["len_s", "nfire", "peak_p", "mean_p", "peak_margin", "mean_margin", "p_slope",
         "rsrp_mean", "rsrp_min", "rsrp_slope", "dl_mean", "dl_min", "since_ho"]
GBM = dict(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=data.SEED)

# the gate is the hardest filter that still holds at least this much recall on the train data.
# because it's chosen on train only, the held-out test recall comes out a few points lower (the
# train->test gap). raise these to keep more recall, lower them to filter harder.
TARGET_RECALL = {"5g": 0.80, "lte": 0.90}


# ----------------------------------------------------------------- episode helpers
def episodes(fire, sec):
    """Merge consecutive fire-seconds into episodes, joining two fires when the gap in `sec` is
    <= LEAD. Returns a list of (i0, i1) index pairs."""
    idx = np.where(fire == 1)[0]
    if len(idx) == 0:
        return []
    eps, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if sec[i] - sec[prev] <= LEAD:
            prev = i
        else:
            eps.append((start, prev))
            start = prev = i
    eps.append((start, prev))
    return eps


def slope(v):
    # least-squares slope of v ignoring NaNs; 0 if there aren't at least two real points
    v = np.asarray(v, float)
    m = np.isfinite(v)
    if m.sum() < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(v))[m], v[m], 1)[0])


def col(d, name):
    # pull a column as a float array, or return all-NaN if the column isn't there
    if name in d.columns:
        return pd.to_numeric(d[name], errors="coerce").to_numpy(float)
    return np.full(len(d), np.nan)


def episode_features(stream, i0, i1):
    # features describing one alarm episode - its shape, the radio around it, and how long since the
    # last real handover. this is what the episode gate learns from.
    s = slice(i0, i1 + 1)
    d = stream["d"]
    rsrp, dl = col(d, "RSRP"), col(d, "DL_bitrate")
    p, tau, sec = stream["p"], stream["tau"], stream["sec"]
    before = stream["gen_idx"][stream["gen_idx"] < i0]   # real handovers before this episode
    return {"len_s": sec[i1] - sec[i0] + 1, "nfire": int(i1 - i0 + 1),
            "peak_p": np.nanmax(p[s]), "mean_p": np.nanmean(p[s]),
            "peak_margin": np.nanmax((p - tau)[s]), "mean_margin": np.nanmean((p - tau)[s]),
            "p_slope": slope(p[s]),
            "rsrp_mean": np.nanmean(rsrp[s]), "rsrp_min": np.nanmin(rsrp[s]), "rsrp_slope": slope(rsrp[s]),
            "dl_mean": np.nanmean(dl[s]), "dl_min": np.nanmin(dl[s]),
            "since_ho": (sec[i0] - sec[before[-1]]) if len(before) else 999.0}


def evaluate(by_trace, eps, keep):
    """Score a set of kept episodes. `keep` is a bool mask over `eps`. Returns the recall (share of
    real handovers caught), the alarm precision, and the false alarms per minute."""
    # rebuild the surviving fire stream per trace from the kept episodes
    surv = {ti: np.zeros_like(d["fire"]) for ti, d in by_trace.items()}
    for (ti, i0, i1), k in zip(eps, keep):
        if k:
            surv[ti][i0:i1 + 1] = by_trace[ti]["fire"][i0:i1 + 1]
    caught = ho = false = true = 0
    mins = 0.0
    for ti, d in by_trace.items():
        sec, gen, f = d["sec"], d["gen"], surv[ti]
        mins += (sec[-1] - sec[0] + 1) / 60 if len(sec) else 0
        gi = np.where(gen == 1)[0]
        ho += len(gi)
        for h in gi:                          # a handover counts as caught if we fired in the LEAD s before it
            if any(h - kk >= 0 and f[h - kk] == 1 for kk in range(1, LEAD + 1)):
                caught += 1
        for i0, i1 in episodes(f, sec):       # was each surviving alarm close to a real handover?
            hi = min(len(gen) - 1, i1 + LEAD)
            true += int(gen[i0:hi + 1].any())
            false += int(not gen[i0:hi + 1].any())
    al = true + false
    return dict(recall=caught / ho if ho else 0, caught=caught, ho=ho, true=true, false=false,
                fa_per_min=false / mins if mins else 0, precision=true / al if al else 0)


# ------------------------------------------------------ windows + base experts
def load_windows(regime):
    """Load every drive, build the windows, and split the drives 70/30 into train/test files (so a
    drive is never on both sides). Returns one dict carrying the windows, the per-window trace/second
    index, the file split, and the per-tier thresholds."""
    traces = data.load_regime(regime)
    X, y, cur, meta = data.build_rich_windows(traces, regime)
    trace_of, tcol = meta[:, 0].astype(int), meta[:, 1].astype(int)
    tr_files, te_files = train_test_split(np.arange(len(traces)), test_size=0.30, random_state=data.SEED)
    return dict(regime=regime, traces=traces, X=X, y=y, cur=cur, trace_of=trace_of, tcol=tcol,
                tr_files=np.array(sorted(int(i) for i in tr_files)),
                te_set=set(int(i) for i in te_files),
                tau_t={t: TAU[name] for t, name in TIER_DISPATCH[regime].items()})


def base_probabilities(split):
    """Base handover probability for every window. The full-train experts score the test windows
    directly; train windows get out-of-fold scores instead (5-fold over the train files), so no
    window is ever scored by an expert that trained on it. Also returns the deployable full-train
    experts."""
    X, y, cur, trace_of = split["X"], split["y"], split["cur"], split["trace_of"]
    bal = np.zeros(len(y), bool)
    bal[data.balanced_index(y)] = True
    p = np.full(len(y), np.nan)
    experts = {}
    # one full-train expert per tier -> used to score the held-out test windows
    for t, name in TIER_DISPATCH[split["regime"]].items():
        fit = (cur == t) & bal & np.isin(trace_of, split["tr_files"])
        experts[name] = RandomForestClassifier(**data.EXPERT_CLF).fit(X[fit], y[fit])
        m_te = (cur == t) & np.isin(trace_of, list(split["te_set"]))
        if m_te.any():
            p[m_te] = experts[name].predict_proba(X[m_te])[:, 1]
    # out-of-fold scores for the train windows
    for tri, tei in KFold(5, shuffle=True, random_state=data.SEED).split(split["tr_files"]):
        fit_files, pred_files = set(split["tr_files"][tri]), set(split["tr_files"][tei])
        for t in TIER_DISPATCH[split["regime"]]:
            fit = (cur == t) & bal & np.isin(trace_of, list(fit_files))
            prd = (cur == t) & np.isin(trace_of, list(pred_files))
            if fit.sum() and prd.any():
                rf = RandomForestClassifier(**data.EXPERT_CLF).fit(X[fit], y[fit])
                p[prd] = rf.predict_proba(X[prd])[:, 1]
    return p, experts


def fit_calibrators(split, p):
    """Per-arm isotonic calibration, fit on the train out-of-fold scores so a probability of 0.6
    actually means about 60%. Returns the calibrators, the calibrated probabilities, and the
    thresholds run through the same calibration."""
    is_tr = np.isin(split["trace_of"], split["tr_files"])
    p_cal = p.copy()
    tau_cal = dict(split["tau_t"])
    calib = {}
    for t, name in TIER_DISPATCH[split["regime"]].items():
        arm = (split["cur"] == t) & np.isfinite(p)
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[arm & is_tr], split["y"][arm & is_tr])
        calib[name] = iso
        p_cal[arm] = iso.predict(p[arm])
        tau_cal[t] = float(iso.predict([split["tau_t"][t]])[0])
    return calib, p_cal, tau_cal


def per_trace_streams(split, p):
    """Spread the per-window probabilities back out into a per-second stream for each trace. Each
    stream carries the probability, the threshold, the fire flag (p >= threshold), the real handover
    positions, and the raw dataframe - everything the episode stage needs."""
    # window probability indexed by (trace, second)
    p_by = {ti: {} for ti in range(len(split["traces"]))}
    for k, (ti, t) in enumerate(zip(split["trace_of"], split["tcol"])):
        p_by[int(ti)][int(t)] = float(p[k])
    cell_column = data.cell_col(split["regime"])
    out = {}
    for ti, df in enumerate(split["traces"]):
        n = len(df)
        ts = df["_ts"].to_numpy()
        ds = np.full(n, 1.0)
        ds[1:] = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")
        sec = ((ts - ts[0]) / np.timedelta64(1, "s")).astype(float)
        cells = df[cell_column].astype("object").to_numpy()
        tier = df["tier_state"].to_numpy(int)
        pv = np.array([p_by[ti].get(t, np.nan) for t in range(n)])
        tau = np.array([split["tau_t"][int(tier[t])] for t in range(n)])
        fire = (np.isfinite(pv) & (pv >= tau)).astype(int)
        gen = np.zeros(n, int)
        for a, b in data._segments(ds, MAXG):
            gen[a:b] = data.handover_events(data.remove_pingpong(cells[a:b], PP_T))
        out[ti] = dict(sec=sec, fire=fire, gen=gen, p=pv, tau=tau, d=df, n=n,
                       gen_idx=np.where(gen == 1)[0])
    return out


def episode_tables(split, streams):
    """Build the episode table the gate trains on. Each alarm episode becomes one row with its
    features, a label (was there a real handover within LEAD s), its trace id (used as the CV group),
    and which side of the train/test split it's on. Returns the train/test matrices plus the streams
    needed to score them."""
    rows = []
    for ti, stream in streams.items():
        for (i0, i1) in episodes(stream["fire"], stream["sec"]):
            hi = min(stream["n"] - 1, i1 + LEAD)
            label = int(stream["gen"][i0:hi + 1].any())
            rows.append((ti, i0, i1, episode_features(stream, i0, i1), label, ti in split["te_set"]))
    Xall = pd.DataFrame([r[3] for r in rows])[FEATS]
    is_te = np.array([r[5] for r in rows])
    lab = np.array([r[4] for r in rows])
    gid = np.array([r[0] for r in rows])
    by = lambda ids: {ti: {k: streams[ti][k] for k in ("sec", "fire", "gen")} for ti in ids}
    tr_ids = [ti for ti in streams if ti not in split["te_set"]]
    te_ids = sorted(ti for ti in streams if ti in split["te_set"])
    return dict(Xtr=Xall[~is_te].to_numpy(), ytr=lab[~is_te], gtr=gid[~is_te],
                Xte=Xall[is_te].to_numpy(),
                tr_eps=[(r[0], r[1], r[2]) for r in rows if not r[5]], tr_by=by(tr_ids),
                te_eps=[(r[0], r[1], r[2]) for r in rows if r[5]], te_by=by(te_ids), te_ids=te_ids)


def select_gate(gbm, tables, target):
    """Pick the gate threshold: the hardest (highest) cut that still keeps train recall >= target.
    Recall here is measured on out-of-fold GBM scores (GroupKFold by trace) so the gate is never
    tuned on episodes it was trained on."""
    oof = np.full(len(tables["ytr"]), np.nan)
    for tri, vai in GroupKFold(5).split(tables["Xtr"], tables["ytr"], tables["gtr"]):
        fold = GradientBoostingClassifier(**GBM).fit(tables["Xtr"][tri], tables["ytr"][tri])
        oof[vai] = fold.predict_proba(tables["Xtr"][vai])[:, 1]
    gate = 0.0
    for g in np.arange(0.0, 0.9, 0.01):
        if evaluate(tables["tr_by"], tables["tr_eps"], oof >= g)["recall"] >= target:
            gate = g
    return gate


# --------------------------------------------------------------------- fit / save
def fit_regime(regime, save=True):
    """Train the whole calibrated cascade for one regime, score it on the held-out test drives, and
    (optionally) save the deployable model + its operating point. Returns everything the reporting
    code needs."""
    split = load_windows(regime)
    p_raw, experts = base_probabilities(split)
    calib, p_cal, tau_cal = fit_calibrators(split, p_raw)
    split_cal = dict(split, tau_t=tau_cal)               # same split, but thresholds in calibrated units
    streams = per_trace_streams(split_cal, p_cal)
    tables = episode_tables(split_cal, streams)
    gbm = GradientBoostingClassifier(**GBM).fit(tables["Xtr"], tables["ytr"])
    gate = select_gate(gbm, tables, TARGET_RECALL[regime])
    keep = gbm.predict_proba(tables["Xte"])[:, 1] >= gate
    res = dict(regime=regime, W=split_cal, streams=streams, te_ids=tables["te_ids"],
               te_eps=tables["te_eps"], by_trace=tables["te_by"], keep=keep, gate=gate,
               tau_cal=tau_cal,
               win=dict(p=p_raw, y=split["y"], cur=split["cur"],
                        is_te=np.isin(split["trace_of"], list(split["te_set"]))))
    if save:
        joblib.dump(dict(experts=experts, calib=calib, tau_cal=tau_cal, gbm=gbm, gate=gate,
                         feats=FEATS, arm=TIER_DISPATCH[regime]),
                    os.path.join(MODELS, f"model_{regime}.pkl"))
        op = evaluate(res["by_trace"], res["te_eps"], res["keep"])
        json.dump(dict(regime=regime, gate=round(float(gate), 4),
                       tau_cal={str(k): round(v, 4) for k, v in tau_cal.items()},
                       operating_point={k: round(float(op[k]), 4)
                                        for k in ("recall", "precision", "fa_per_min")},
                       n_test_handovers=int(op["ho"])),
                  open(os.path.join(MODELS, f"model_{regime}.json"), "w"), indent=2)
    return res


def predict_regime(regime):
    """Reload a saved model and reproduce the held-out test result without retraining - same split,
    calibration and gate as fit_regime. Used by the KPI replay in report.py."""
    split = load_windows(regime)
    saved = joblib.load(os.path.join(MODELS, f"model_{regime}.pkl"))
    experts, calib = saved["experts"], saved["calib"]
    tau_cal, gbm, gate = saved["tau_cal"], saved["gbm"], saved["gate"]
    is_te = np.isin(split["trace_of"], list(split["te_set"]))
    # calibrated probability on the test windows only
    p = np.full(len(split["y"]), np.nan)
    for t, name in TIER_DISPATCH[regime].items():
        m = (split["cur"] == t) & is_te
        if m.any():
            p[m] = calib[name].predict(experts[name].predict_proba(split["X"][m])[:, 1])
    split_cal = dict(split, tau_t=tau_cal)
    streams = per_trace_streams(split_cal, p)
    tables = episode_tables(split_cal, streams)
    if len(tables["Xte"]):
        keep = gbm.predict_proba(tables["Xte"])[:, 1] >= gate
    else:
        keep = np.zeros(0, bool)
    return dict(regime=regime, W=split_cal, streams=streams, te_ids=tables["te_ids"],
                te_eps=tables["te_eps"], by_trace=tables["te_by"], keep=keep, gate=gate,
                tau_cal=tau_cal, win=dict(p=p, y=split["y"], cur=split["cur"], is_te=is_te))
