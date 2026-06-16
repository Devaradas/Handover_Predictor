# Loads the drive-test logs and turns them into the features and labels the models use.
# Works for both regimes (5G and LTE).
#
# Every CSV in input/ is one drive ("trace"), logged once per second. For each trace I build
# 13 per-second features plus a binary "which tier am I on" flag, then stack 5 seconds of those
# into one observation window. A window's label is whether a handover happens in the next 3 s.
# There are also a few helpers for ping-pong removal and serving-cell-change detection that the
# rest of the code reuses.
import os, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUT = os.path.join(ROOT, "input")

# window / label / sampling settings, shared by both regimes
WIN_BIN, HORIZON = 5, 3          # 5 s of history per window, look 3 s ahead for the label
MAX_GAP = 2                      # a window can't cross a recording gap longer than this (seconds)
PINGPONG_T = 3                   # an A->B->A bounce with B held <= this many seconds is a ping-pong
N_POS, CLOSE_WIN, RECENT_WIN, SEED = 750, 10, 3, 0
PER_SEC_ALL = ["DL_bitrate", "UL_bitrate", "RSSI", "RSRQ", "RSRP", "NRxRSRP", "NRxRSRQ",
               "SNR", "CQI", "Speed", "rat_tier", "time_on_cell", "recent_switch"]

# the RandomForest settings the four experts all share (fairly deep trees, sqrt feature subset)
EXPERT_CLF = dict(n_estimators=500, max_depth=20, max_features="sqrt", min_samples_split=10,
                  min_samples_leaf=2, criterion="gini", n_jobs=-1, random_state=SEED)

# where each regime's data lives and how the tier is decided. the tier flag is binary inside a
# regime:
#   5G  -> 1 means 5G/NR,  0 means anything older (4G/3G)
#   LTE -> 1 means 4G/LTE, 0 means sub-4G (3G/2G)
REGIMES = {
    "5g":  {"dirs": [os.path.join(INPUT, "5g")],
            "group_from_folder": False,
            "tier_rule": lambda s: int(("5G" in s) or ("NR" in s))},
    "lte": {"dirs": [os.path.join(INPUT, "lte", g) for g in ("bus", "car", "train")],
            "group_from_folder": True,
            "tier_rule": lambda s: int("LTE" in s)},
}


def list_csvs(dirs):
    files = []
    for folder in dirs:
        files += glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    # skip mac zip junk and the precomputed transitions file if it happens to be there
    files = [f for f in files if "__MACOSX" not in f
             and os.path.basename(f).lower() != "cellid_transitions.csv"]
    return sorted(files)


def load_trace(path, tier_rule, group):
    df = pd.read_csv(path).replace("-", np.nan)
    df["_ts"] = pd.to_datetime(df["Timestamp"], format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    df = df.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    # bitrates come in kbit/s, convert to Mbit/s
    for col in ["DL_bitrate", "UL_bitrate"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 1000.0
    for col in ["RSSI", "RSRQ", "RSRP", "NRxRSRP", "NRxRSRQ", "SNR", "CQI", "Speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["tier_state"] = df["NetworkMode"].apply(lambda s: tier_rule(str(s).upper()))
    df["rat_tier"] = df["tier_state"]

    # how many seconds we've been on the current serving cell (resets to 0 when CellID changes).
    # I first tried feeding the CellID itself, but the ids are just arbitrary numbers per file, so
    # they meant nothing once you train on some files and test on others. time-on-cell is the same
    # idea but it actually transfers across drives.
    cell_run = (df["CellID"] != df["CellID"].shift()).cumsum()
    df["time_on_cell"] = df.groupby(cell_run).cumcount().astype(float)

    # was there a tier change in the last few seconds?
    switched = (df["tier_state"] != df["tier_state"].shift(1)).fillna(False).astype(int)
    df["recent_switch"] = switched.rolling(RECENT_WIN, min_periods=1).max().astype(int)
    df["group"] = group

    # missing radio fields are kept as NaN on purpose. each model handles them its own way later
    # (the RF just maps NaN to 0), and the RLF check needs the real NaN so it can tell "no data"
    # apart from "zero throughput". DL/UL/RSRP/Speed are never actually missing in these datasets.
    return df


def load_regime(regime):
    cfg = REGIMES[regime]
    traces = []
    for path in list_csvs(cfg["dirs"]):
        if cfg["group_from_folder"]:
            group = os.path.basename(os.path.dirname(path))
        else:
            group = "driving"
        traces.append(load_trace(path, cfg["tier_rule"], group))
    return traces


def rf_fill(X):
    # RandomForest can't take NaN, so replace missing values with 0. it's a constant fill, so doing
    # it here vs inside a CV fold gives the same result (nothing leaks across folds).
    return np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)


def build_rich_windows(traces, regime, max_gap=MAX_GAP, include_gps=False):
    """Build the 5 s observation windows and their 3 s look-ahead handover labels.

    Returns (X, y, modes, meta), where meta[k] = (trace_index, t) records which trace and which
    second window k came from. Each window stacks the per-second features (see feature_frame) over
    the last 5 seconds in time order, then add_triggers tacks on three neighbour-cell triggers. The
    label is 1 if a (non ping-pong) serving-cell change happens in the next 3 seconds. Windows that
    would straddle a recording gap are dropped, and GPS is dropped by default so the model can't
    just memorise where it is."""
    cell_column = cell_col(regime)
    base, keep = None, None
    Xs, ys, modes, meta = [], [], [], []
    for ti, df in enumerate(traces):
        feats = feature_frame(df).to_numpy(float)
        if base is None:                       # work out the column layout once, from the first trace
            all_cols = base_features(df)
            keep = [i for i, c in enumerate(all_cols)
                    if include_gps or c not in ("Latitude", "Longitude")]
            base = [all_cols[i] for i in keep]
        feats = feats[:, keep]

        tier_state = df["tier_state"].to_numpy(int)
        cells = df[cell_column].astype("object").to_numpy()
        ts = df["_ts"].to_numpy()
        n = len(df)

        # seconds between consecutive rows
        dsec = np.full(n, 1.0)
        if n > 1:
            dsec[1:] = (ts[1:] - ts[:-1]) / np.timedelta64(1, "s")

        # per-second handover label, done segment by segment so the ping-pong cleanup never looks
        # across a recording gap
        cell_ho = np.zeros(n, int)
        for (a, b) in _segments(dsec, max_gap):
            cell_ho[a:b] = handover_events(remove_pingpong(cells[a:b], PINGPONG_T))

        for t in range(WIN_BIN - 1, n - HORIZON):
            if np.nanmax(dsec[t - WIN_BIN + 2: t + HORIZON + 1]) > max_gap:
                continue                       # this window crosses a gap, skip it
            Xs.append(feats[t - WIN_BIN + 1: t + 1].flatten())
            ys.append(int(cell_ho[t + 1: t + HORIZON + 1].any()))
            modes.append(int(tier_state[t]))
            meta.append((ti, t))

    Xdf = pd.DataFrame(np.asarray(Xs), columns=windowed_columns(base, WIN_BIN))
    Xdf = add_triggers(Xdf)                    # + gap_now / gap_close / rsrp_drop
    X = rf_fill(Xdf.to_numpy(float))
    return X, np.asarray(ys), np.asarray(modes), np.asarray(meta)


def balanced_index(y, seed=SEED):
    # all the positive windows (capped at N_POS) plus an equal number of negatives. half of the
    # negatives are taken from near a positive ("close") and half from anywhere else, so the model
    # also sees hard negatives.
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    if len(pos) > N_POS:
        pos = rng.choice(pos, N_POS, replace=False)
    neg = np.where(y == 0)[0]
    dist = np.min(np.abs(neg[:, None] - pos[None, :]), axis=1)
    close, far = neg[dist <= CLOSE_WIN], neg[dist > CLOSE_WIN]
    n_close = min(len(pos) // 2, len(close))
    sel_neg = np.concatenate([rng.choice(close, n_close, replace=False),
                              rng.choice(far, len(pos) - n_close, replace=False)])
    idx = np.concatenate([pos, sel_neg])
    rng.shuffle(idx)
    return idx


# --------------------------------------------------------------------- handover labels
def cell_col(regime):
    # the column holding the serving-cell id for this regime
    return "RAWCELLID" if regime == "5g" else "CellID"


def remove_pingpong(cells, T):
    """Remove A->B->A bounces where the phone sat on B for <= T samples, by relabelling that short
    B run back to A (kills both fake transitions). Repeats until nothing changes. NaN cells are
    never merged. This only touches the labels - the features always see the raw cell sequence, so
    no future denoising leaks into the model inputs."""
    cells = list(cells)
    changed = True
    while changed:
        changed = False
        # split into runs of equal cell value: [value, start, end)
        runs, s = [], 0
        for k in range(1, len(cells) + 1):
            if (k == len(cells) or cells[k] != cells[s]
                    or pd.isna(cells[k]) or pd.isna(cells[s])):
                runs.append([cells[s], s, k])
                s = k
        for r in range(1, len(runs) - 1):
            a0, b, a1 = runs[r - 1][0], runs[r][0], runs[r + 1][0]
            lenB = runs[r][2] - runs[r][1]
            if (pd.notna(a0) and pd.notna(b) and pd.notna(a1)
                    and a0 == a1 and a0 != b and lenB <= T):
                for idx in range(runs[r][1], runs[r][2]):
                    cells[idx] = a0
                changed = True
                break
    return cells


def handover_events(cells):
    # 1 wherever the serving cell changes from one second to the next (and both ids are known).
    # callers segment on gaps first, so there's no need to check the time delta here.
    he = np.zeros(len(cells), int)
    for t in range(1, len(cells)):
        a, b = cells[t - 1], cells[t]
        if pd.notna(a) and pd.notna(b) and a != b:
            he[t] = 1
    return he


def _segments(dt, max_gap):
    # [start, end) ranges where consecutive samples are <= max_gap apart (i.e. no recording gap)
    segs, s = [], 0
    for t in range(1, len(dt)):
        if dt[t] > max_gap:
            segs.append((s, t))
            s = t
    segs.append((s, len(dt)))
    return segs


# ------------------------------------------------- per-second feature frame + triggers
# Ping latency (PINGAVG/MIN/MAX/STDEV/LOSS) is left out: it mostly tells you where the phone is
# (it tracks location / serving gateway), which is the same reason raw GPS is dropped.
NUM_FEATS = list(PER_SEC_ALL) + ["Latitude", "Longitude"]
NET_MAP   = {"5G": 5, "LTE": 4, "HSPA+": 3.4, "HSUPA": 3.3, "HSDPA": 3.2, "UMTS": 3.0}   # RAT level
STATE_MAP = {"D": 0, "I": 1, "V": 2, "VD": 3}                                            # data/idle/voice
MISS_FLAG = ["SNR", "CQI", "RSSI", "RSRQ", "NRxRSRP", "NRxRSRQ"]                          # can go missing


def feature_frame(d):
    """One row per second: PER_SEC_ALL + GPS + coded NetworkMode/State, plus a *_na flag for each
    feature that tends to be missing. Everything is forward-filled (past values only), so building
    windows out of this never leaks the future. Columns the file doesn't have come out all NaN."""
    cols = {}
    for c in NUM_FEATS:
        cols[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)
    f = pd.DataFrame(cols)
    f["NetworkMode"] = d["NetworkMode"].map(NET_MAP)        # current RAT, coded (causal)
    f["State"] = d["State"].map(STATE_MAP)
    flags = f[MISS_FLAG].isna().astype(float).add_suffix("_na")   # 1 = value missing this second (pre-fill)
    f = f.ffill()                                                 # carry the last reading forward
    return pd.concat([f, flags], axis=1)


def base_features(d):
    # the ordered list of column names that feature_frame produces
    return list(feature_frame(d).columns)


def windowed_columns(features, W=WIN_BIN):
    # column names for a flattened W-second window, oldest second first: f_t-(W-1) ... f_t-0
    return [f"{f}_t-{W-1-k}" for k in range(W) for f in features]


def add_triggers(X):
    """Three hand-made handover triggers tacked onto the window:
      gap_now   = serving minus neighbour RSRP right now (small/negative -> neighbour is competitive)
      gap_close = how fast that gap is closing over the window (> 0 -> closing)
      rsrp_drop = how much serving RSRP fell across the window (> 0 -> serving getting worse)
    Needs the windowed RSRP / NRxRSRP columns to already be there."""
    X = X.copy()
    X["gap_now"] = X["RSRP_t-0"] - X["NRxRSRP_t-0"]
    X["gap_close"] = (X["RSRP_t-4"] - X["NRxRSRP_t-4"]) - (X["RSRP_t-0"] - X["NRxRSRP_t-0"])
    X["rsrp_drop"] = X["RSRP_t-4"] - X["RSRP_t-0"]
    return X
