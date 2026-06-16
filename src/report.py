# Results and KPI reporting for the trained predictor, on the held-out test drives. Two entry
# points:
#   build(results)      the scorecard main.py prints and saves: (1) the catch vs false-alarm table
#                       (calibrated cascade, after the episode gate), and (2) the recorded drive KPIs.
#   kpi_replay(regime)  the KPI replay - recorded serving-cell sequence vs an early-switch
#                       counterfactual, giving HOP / HOPP / RLF / data rate side by side.
# Run the KPI replay on its own with:  py -3.12 src/report.py [5g|lte|both]   (default both)
import os, sys
import numpy as np, pandas as pd
import model as M
import data

OUT = os.path.join(M.ROOT, "output", "tables")
os.makedirs(OUT, exist_ok=True)
OUT_MAPS = os.path.join(M.ROOT, "output", "maps")

# KPI-replay knobs
K        = 3            # cling / recovery window in seconds (same as the prediction horizon)
LEAD     = M.LEAD       # furthest ahead the model is allowed to fire (s)
MARGIN   = 1.0          # Mbps; the new cell must beat the old by this much to count as "avoidable"
DEG_RSRP = -100.0       # dBm; serving at/below this just before a switch counts as a degraded cling
PP_T, MAXG = M.PP_T, M.MAXG


# ================================================== (A) scorecard (called by main.py)
def operational(res):
    # catch rate and false-alarm rate, after the episode gate
    by, eps = res["by_trace"], res["te_eps"]

    def row(tag, r):
        return dict(regime=res["regime"].upper(), stage=tag, recall=round(r["recall"], 3),
                    caught=r["caught"], handovers=r["ho"], true_alarms=r["true"],
                    false_alarms=r["false"], precision=round(r["precision"], 3),
                    false_per_min=round(r["fa_per_min"], 3))
    gated = M.evaluate(by, eps, res["keep"])
    return pd.DataFrame([row("calibrated cascade", gated)])


def recorded_kpis(res):
    # KPIs measured straight off the recorded test traces (no model involved): handovers per minute,
    # ping-pong share, mean/median downlink, and the RLF rate
    regime = res["regime"]
    cell_column = data.cell_col(regime)
    mins = gen = raw = pp = rlf = nsec = 0
    dls = []
    for ti in res["te_ids"]:
        stream = res["streams"][ti]
        d = stream["d"]
        n = stream["n"]
        sec = stream["sec"]
        ds = np.full(n, 1.0)
        if n > 1:
            ds[1:] = np.diff(sec)
        cells = d[cell_column].astype("object").to_numpy()
        rsrp, dl = M.col(d, "RSRP"), M.col(d, "DL_bitrate")
        state = d["State"].astype(str).str.upper().to_numpy()
        # raw serving-cell changes, per gap-free segment
        r = np.zeros(n, int)
        for a, b in data._segments(ds, M.MAXG):
            r[a:b] = data.handover_events(cells[a:b])
        # RLF = throughput collapsed to zero while in data state
        rl = np.zeros(n, int)
        for a, b in data._segments(ds, M.MAXG):
            for t in range(a, b):
                if state[t] != "D":
                    continue
                rl[t] = int(np.isfinite(dl[t]) and dl[t] <= 0)
        mins += (sec[-1] - sec[0] + 1) / 60 if n else 0
        gen += int(stream["gen"].sum())
        raw += int(r.sum())
        pp += int(r.sum() - stream["gen"].sum())
        rlf += int(rl.sum())
        nsec += n
        dls.append(dl)
    dl_all = np.concatenate(dls)
    return dict(regime=regime.upper(), test_min=round(mins, 1), HOP_per_min=round(gen / mins, 3),
                HOPP_pct=round(100 * pp / max(1, raw), 1),
                DL_mean_Mbps=round(float(np.nanmean(dl_all)), 1),
                DL_median_Mbps=round(float(np.nanmedian(dl_all)), 1),
                RLF_rate=round(rlf / max(1, nsec), 4),
                gen_HO=gen, raw_HO=raw, pingpong=pp)


def build(results):
    op = pd.concat([operational(r) for r in results.values()], ignore_index=True)
    kp = pd.DataFrame([recorded_kpis(r) for r in results.values()])
    op.to_csv(os.path.join(OUT, "scorecard_operational.csv"), index=False)
    kp.to_csv(os.path.join(OUT, "scorecard_kpis.csv"), index=False)
    print("\n=== (1) operational  (calibrated cascade, held-out test) ===")
    print(op.to_string(index=False))
    print("\n=== (2) recorded KPIs  (held-out test) ===")
    print(kp.to_string(index=False))
    print(f"\nwrote scorecard_operational.csv, scorecard_kpis.csv -> "
          f"{os.path.relpath(OUT, M.ROOT)}")
    return op, kp


# ================================================== (B) KPI replay (recorded vs early-switch)
# The KPIs:
#   HOP   = N_HO / T_min        handovers per minute            (lower is better)
#   HOPP  = N_pp / N_HO         ping-pong share of handovers     (towards 0)
#   RLF   = N_RLF / N_HO        link failures per handover       (lower is better)
#   Rbar  = mean(R_DL + R_UL)   total data rate, Mbps            (higher is better)
# N_HO counts every serving-cell change (ping-pongs included); N_pp is the ping-pong ones; N_RLF is
# the number of throughput-zero link-failure events. The counterfactual is an early-switch UE: in
# the cling window of each caught avoidable handover it is moved onto the target cell early, so its
# cells / DL / UL there take the target's measured post-switch values.
def gated_survivors(res):
    # per test trace, the model's fire stream after the gate has dropped the rejected episodes
    surv = {ti: np.zeros_like(res["streams"][ti]["fire"]) for ti in res["te_ids"]}
    for (ti, i0, i1), k in zip(res["te_eps"], res["keep"]):
        if k:
            surv[ti][i0:i1 + 1] = res["streams"][ti]["fire"][i0:i1 + 1]
    return surv


def _num(d, c):
    if c in d.columns:
        return pd.to_numeric(d[c], errors="coerce").to_numpy(float)
    return np.full(len(d), np.nan)


def _dmean(x, sD, a, b):
    # mean of x over [a, b) restricted to data-state seconds, NaN if there are none
    s = x[a:b][sD[a:b]]
    return float(np.nanmean(s)) if len(s) and np.isfinite(s).any() else np.nan


def analyze(res, regime):
    """Per-handover table for the replay. For each genuine (ping-pong-cleaned) handover it records
    the model's final gated fire timing, the downlink just before vs just after, and whether the
    switch looks 'avoidable' (the new cell is durably better, i.e. the phone clung to a worse cell).
    build_streams() then uses f_idx / h_idx / post_dl to build the early-switch counterfactual."""
    cell_column = data.cell_col(regime)
    gated = gated_survivors(res)
    rows = []
    for ti in res["te_ids"]:
        stream = res["streams"][ti]
        d = stream["d"].reset_index(drop=True)
        n = stream["n"]
        cells = d[cell_column].astype(str).to_numpy()
        net = d["NetworkMode"].astype(str).str.strip().to_numpy()
        rsrp = pd.to_numeric(d["RSRP"], errors="coerce").to_numpy(float)
        dl = pd.to_numeric(d["DL_bitrate"], errors="coerce").to_numpy(float)
        stateD = (d["State"].astype(str).str.upper().to_numpy() == "D")
        speed = pd.to_numeric(d["Speed"], errors="coerce").to_numpy(float)   # km/h
        fire = gated[ti]
        for h in stream["gen_idx"]:                  # genuine (ping-pong-cleaned) handovers
            if h <= 0 or h >= n:
                continue
            a0 = max(0, h - K)
            b1 = min(n, h + K)
            pre_dl, post_dl = _dmean(dl, stateD, a0, h), _dmean(dl, stateD, h, b1)
            pre_rsrp = _dmean(rsrp, np.ones(n, bool), a0, h)
            post_rsrp = _dmean(rsrp, np.ones(n, bool), h, b1)
            if not (np.isfinite(pre_dl) and np.isfinite(post_dl)):
                continue                             # no data-state seconds either side -> can't judge it
            avoidable = post_dl > pre_dl + MARGIN    # new cell durably better -> the recorded switch was late
            preD = stateD[a0:h]
            oracle = float(np.nansum(np.clip(post_dl - dl[a0:h], 0, None)[preD]))
            # did the model fire in the LEAD s before the handover, and if so, when?
            lo = max(0, h - LEAD)
            fired = np.where(fire[lo:h] == 1)[0]
            f = (lo + int(fired[0])) if len(fired) else -1
            lead = (h - f) if f >= 0 else 0
            capD = stateD[f:h] if f >= 0 else np.zeros(0, bool)
            captured = float(np.nansum(np.clip(post_dl - dl[f:h], 0, None)[capD])) if f >= 0 else 0.0
            dist_m = (speed[h] * 1000 / 3600.0) * lead if np.isfinite(speed[h]) else np.nan
            rows.append(dict(trace=ti, sec=int(stream["sec"][h]), h_idx=int(h), f_idx=int(f),
                             inter_rat=int(net[h] != net[h - 1]), from_rat=net[h - 1], to_rat=net[h],
                             pre_dl=pre_dl, post_dl=post_dl, pre_rsrp=pre_rsrp, post_rsrp=post_rsrp,
                             avoidable=int(avoidable),
                             degraded=int(np.isfinite(pre_rsrp) and pre_rsrp < DEG_RSRP),
                             oracle_recover_mbit=oracle, model_fired=int(f >= 0), lead_s=lead,
                             model_recover_mbit=captured, ue_move_m=dist_m))
    return pd.DataFrame(rows)


def build_streams(res, df, regime):
    """Build, per test trace, the recorded stream and the counterfactual early-switch stream (cells,
    DL, UL, State, sec each). In each avoidable handover the model caught, the early-switch UE is put
    onto the target cell during the cling window, so its cells / DL / UL there take the target's
    measured post-switch values."""
    cell_column = data.cell_col(regime)
    rec, cf = [], []
    for ti in res["te_ids"]:
        d = res["streams"][ti]["d"].reset_index(drop=True)
        sec = res["streams"][ti]["sec"]
        n = len(d)
        cells = d[cell_column].astype(str).to_numpy()
        dl = _num(d, "DL_bitrate")
        ul = _num(d, "UL_bitrate")
        state = d["State"].astype(str).to_numpy()
        sD = (np.array([s.upper() for s in state]) == "D")
        rec.append(dict(cells=cells.copy(), dl=dl.copy(), ul=ul.copy(), state=state, sec=sec))

        c_cells = cells.astype(object).copy()
        c_dl = dl.copy()
        c_ul = ul.copy()
        fires = []
        ev = df[(df.trace == ti) & (df.avoidable == 1) & (df.model_fired == 1)]
        for _, e in ev.iterrows():
            f, h = int(e.f_idx), int(e.h_idx)
            if 0 <= f < h:
                post_ul = _dmean(ul, sD, h, min(n, h + K))      # target's measured post-switch UL
                c_cells[f:h] = cells[h]
                for t in range(f, h):
                    if sD[t]:
                        c_dl[t] = e.post_dl
                        if np.isfinite(post_ul):
                            c_ul[t] = post_ul
                fires.append((f, h))
        # the controller has dwell hysteresis: once it switches it commits to the target and won't
        # bounce back within PP_T s, so short back-and-forth around its own switches is suppressed
        # (3GPP TTT/hysteresis). this is applied only inside the controller's own action windows -
        # ping-pongs elsewhere on the drive are left alone.
        for f, h in fires:
            lo2, hi2 = max(0, f - PP_T), min(n, h + PP_T)
            c_cells[lo2:hi2] = np.array(data.remove_pingpong(list(c_cells[lo2:hi2]), PP_T), dtype=object)
        cf.append(dict(cells=c_cells, dl=c_dl, ul=c_ul, state=state, sec=sec))
    return rec, cf


def stream_kpis(per_trace):
    # roll a list of per-trace streams up into the drive-level KPIs
    raw = gen = pp = nrlf = nrlf_sec = nsec = 0
    tmin = 0.0
    rates = []
    for tr in per_trace:
        cells = np.asarray(tr["cells"], dtype=object)
        n = len(cells)
        sec = np.asarray(tr["sec"], float)
        dt = np.full(n, 1.0)
        if n > 1:
            dt[1:] = np.diff(sec)
        dl = np.asarray(tr["dl"], float)
        ul = np.asarray(tr["ul"], float)
        sD = (np.array([str(s).upper() for s in tr["state"]]) == "D")
        for a, b in data._segments(dt, MAXG):
            r = data.handover_events(cells[a:b])
            g = data.handover_events(data.remove_pingpong(cells[a:b], PP_T))
            raw += int(r.sum())
            gen += int(g.sum())
            pp += int(r.sum() - g.sum())
            cond = sD[a:b] & np.isfinite(dl[a:b]) & (dl[a:b] <= 0)    # link-failure seconds
            starts = cond & ~np.concatenate([[False], cond[:-1]])     # count events (run starts), not seconds
            nrlf += int(starts.sum())
            nrlf_sec += int(cond.sum())
        nsec += n
        tmin += (sec[-1] - sec[0] + 1) / 60 if n else 0
        rates.append(np.nan_to_num(dl) + np.nan_to_num(ul))          # R_DL + R_UL per second
    allr = np.concatenate(rates) if rates else np.array([np.nan])
    return dict(T_min=tmin, N_HO=raw, N_pp=pp, N_RLF=nrlf, N_RLF_sec=nrlf_sec, N_sec=nsec,
                N_HO_gen=gen,
                HOP=raw / tmin if tmin else np.nan, HOPP=pp / raw if raw else np.nan,
                RLF=nrlf / raw if raw else np.nan, RLF_persec=nrlf_sec / nsec if nsec else np.nan,
                Rbar=float(np.mean(allr)))


def kpi_replay(regime):
    res = M.predict_regime(regime)
    df = analyze(res, regime)
    rec, cf = build_streams(res, df, regime)
    rec_kpis, cf_kpis = stream_kpis(rec), stream_kpis(cf)
    print("\n" + "=" * 72)
    print(f"DRIVE-LEVEL KPIs  --  regime={regime.upper()}  "
          f"({len(res['te_ids'])} held-out traces, all measured)")
    print("=" * 72)
    print(f"  components: T={rec_kpis['T_min']:.1f} min,  N_HO={rec_kpis['N_HO']} "
          f"(genuine {rec_kpis['N_HO_gen']}, ping-pong {rec_kpis['N_pp']}),  "
          f"N_RLF={rec_kpis['N_RLF']} events / {rec_kpis['N_RLF_sec']} sec of {rec_kpis['N_sec']}")
    rows = [("HOP    (HO/min, lower better)", "HOP", "{:.3f}"),
            ("HOPP   (ping-pong share ->0)", "HOPP", "{:.3f}"),
            ("RLF/HO (failures/HO, lower)", "RLF", "{:.4f}"),
            ("RLF/s  (zero-DL sec share, lower)", "RLF_persec", "{:.5f}"),
            ("Rbar   (DL+UL Mbps, higher)", "Rbar", "{:.2f}")]
    print(f"\n  {'KPI':34s} {'Recorded':>12s} {'Model+Gate':>12s} {'change':>10s}")
    print("  " + "-" * 70)
    for lab, key, fmt in rows:
        a, b = rec_kpis[key], cf_kpis[key]
        change = ""
        if np.isfinite(a) and np.isfinite(b) and a != 0:
            change = f"{(b / a - 1) * 100:+.1f}%"
        print(f"  {lab:34s} {fmt.format(a):>12s} {fmt.format(b):>12s} {change:>10s}")
    pd.DataFrame({"Recorded": rec_kpis, "Model+Gate": cf_kpis}).to_csv(
        os.path.join(OUT_MAPS, f"kpi_report_{regime}.csv"))
    print(f"\n  note: 'Model+Gate' is the early-switch controller with dwell hysteresis (commits to")
    print(f"  its target, no back-switch within {PP_T}s) applied only inside its own action windows,")
    print(f"  so the HOPP drop is a real controller behaviour (3GPP TTT/hysteresis), not a")
    print(f"  denominator trick. N_HO counts all serving-cell changes; Rbar is over all seconds")
    print(f"  (idle included, missing DL/UL = 0). -> {os.path.relpath(OUT_MAPS, M.ROOT)}\\kpi_report_{regime}.csv")
    return rec_kpis, cf_kpis


def main():
    os.makedirs(OUT_MAPS, exist_ok=True)
    np.seterr(divide="ignore", invalid="ignore")
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "both"
    regimes = ["lte", "5g"] if arg == "both" else [arg]
    for regime in regimes:
        kpi_replay(regime)


if __name__ == "__main__":
    main()
