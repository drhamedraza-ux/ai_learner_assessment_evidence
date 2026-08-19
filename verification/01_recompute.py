#!/usr/bin/env python3
"""Layer 3 — recompute every reported RQ2/RQ3/RQ4-input table from predictions_test.npz.

Nothing here reuses the pipeline's own table code paths for the comparison: the metric
definitions are re-implemented from pcdt_p2.py's source, then the results are diffed
cell-by-cell against the CSVs the runs actually wrote.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix

ROOT = Path("/home/claude/work/P02/PCDT_P2_runs/runs")
BUCKETS = [("within", 0, 20*60), ("sameday", 20*60, 86400), ("d1", 86400, 7*86400),
           ("d7", 7*86400, 28*86400), ("d28", 28*86400, np.inf)]
REAL_HORIZONS = ["d1", "d7", "d28"]
MODEL_ORDER = ["DKT", "SAKT", "AKT", "DKVMN", "DKT-F", "PCDT", "PCDT-noPsi", "Control"]
BASELINES = ["DKT", "SAKT", "AKT", "DKVMN", "DKT-F"]
TOL = 5e-6


def ece(p, y, bins=15):
    e, edges = 0., np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    for b in range(bins):
        s = idx == b
        if s.sum():
            e += s.mean() * abs(y[s].mean() - p[s].mean())
    return float(e)


def evaluate(p, y, bins=15):
    fin = np.isfinite(p) & np.isfinite(y)
    p, y = p[fin], y[fin]
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return dict(n=int(len(y)), auc=float(roc_auc_score(y, p)),
                acc=float(((p >= .5) == y).mean()),
                rmse=float(np.sqrt(((p - y) ** 2).mean())),
                nll=float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()),
                ece=ece(p, y, bins), brier=float(brier_score_loss(y, p)))


def utility(p, y, thr):
    flag, need = (p < thr).astype(int), (y == 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(need, flag, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec) else np.nan
    return dict(threshold=thr, flagged=int(tp + fp), correctly_flagged=int(tp),
                false_alarms=int(fp), missed=int(fn), precision=prec, recall=rec,
                specificity=spec, f1=f1, balanced_accuracy=(rec + spec) / 2)


def paired_bootstrap(pa, pb, y, rows_idx, n_boot, seed):
    rg = np.random.default_rng(seed)
    learners = np.unique(rows_idx)
    by = {l: np.where(rows_idx == l)[0] for l in learners}
    d = []
    for _ in range(n_boot):
        s = rg.choice(learners, len(learners), replace=True)
        ii = np.concatenate([by[l] for l in s])
        if len(np.unique(y[ii])) < 2:
            continue
        d.append(roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    pv = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return (float(roc_auc_score(y, pa) - roc_auc_score(y, pb)), float(lo), float(hi),
            float(min(pv, 1)))


def did_retention(pa, pb, bid, M, B, Y, cfg, ref_id=0, n_boot=None):
    selh, selr = (B == bid) & M, (B == ref_id) & M
    nl, npt = int(np.unique(np.where(selh)[0]).size), int(selh.sum())
    if nl < cfg["min_learners_bucket"] or npt < cfg["min_points_bucket"]:
        return dict(adequate=False, n_learners=nl, n_points=npt)

    def auc(sel, p):
        y = Y[sel]
        return roc_auc_score(y, p[sel]) if len(np.unique(y)) > 1 else np.nan

    ra = (auc(selh, pa) - auc(selh, pb)) - (auc(selr, pa) - auc(selr, pb))
    nb = cfg["n_bootstrap"] if n_boot is None else n_boot
    rg = np.random.default_rng(cfg["seed"])
    ls = np.unique(np.where(selh | selr)[0])
    boot = []
    for _ in range(nb):
        keep = np.zeros(selh.shape[0], bool)
        keep[rg.choice(ls, len(ls), replace=True)] = True
        mh, mr = selh & keep[:, None], selr & keep[:, None]
        if mh.sum() < 50 or mr.sum() < 50:
            continue
        try:
            boot.append((auc(mh, pa) - auc(mh, pb)) - (auc(mr, pa) - auc(mr, pb)))
        except Exception:
            pass
    boot = np.asarray([b for b in boot if np.isfinite(b)])
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan))
    return dict(adequate=True, n_learners=nl, n_points=npt, retention_advantage=float(ra),
                ci_lo=float(lo), ci_hi=float(hi), auc_pcdt=float(auc(selh, pa)),
                auc_comparator=float(auc(selh, pb)))


def cmp_frames(name, got, want, keys, cols, out):
    """Cell-by-cell comparison anchored on key columns (never set membership)."""
    g = got.set_index(keys).sort_index()
    w = want.set_index(keys).sort_index()
    if list(g.index) != list(w.index):
        out.append(f"    {name}: ROW KEY MISMATCH\n      recomputed={list(g.index)}\n      onfile    ={list(w.index)}")
        return 0, 1
    ok = bad = 0
    for c in cols:
        if c not in w.columns:
            out.append(f"    {name}: column {c} absent from file")
            bad += 1
            continue
        for k in g.index:
            a, b = g.loc[k, c], w.loc[k, c]
            if isinstance(a, (np.floating, float)) and isinstance(b, (np.floating, float, int, np.integer)):
                if np.isnan(a) and (isinstance(b, float) and np.isnan(b)):
                    ok += 1
                    continue
                if np.isnan(a) != (isinstance(b, float) and np.isnan(b)):
                    bad += 1
                    out.append(f"    {name}[{k},{c}] NaN mismatch: {a} vs {b}")
                    continue
                if abs(float(a) - float(b)) > TOL:
                    bad += 1
                    out.append(f"    {name}[{k},{c}] {float(a):.8f} vs file {float(b):.8f} "
                               f"(d={float(a) - float(b):+.2e})")
                else:
                    ok += 1
            else:
                if str(a) != str(b):
                    bad += 1
                    out.append(f"    {name}[{k},{c}] {a!r} vs file {b!r}")
                else:
                    ok += 1
    return ok, bad


DO_BOOT = "--boot" in sys.argv
FILT = [a for a in sys.argv[1:] if not a.startswith("--")]
runs = sorted([d for d in ROOT.iterdir() if d.name.startswith("RQ2RQ3")
               and (not FILT or any(f in d.name for f in FILT))])
grand_ok = grand_bad = 0
report = []

for rd in runs:
    mani = json.load(open(rd / "manifest.json"))
    cfg = mani["config"]
    cfg["seed"] = mani["seed"]
    z = np.load(rd / "artifacts/predictions_test.npz")
    Y, M, B = z["targets"], z["mask"] > 0, z["bucket"]
    PRED = {k: z[k] for k in MODEL_ORDER}
    out = []
    ok = bad = 0

    # --- Table 1: bucket counts -------------------------------------------------
    rows = [dict(horizon=k, points=int(((B == i) & M).sum()),
                 learners=int(np.unique(np.where((B == i) & M)[0]).size))
            for i, (k, _, _) in enumerate(BUCKETS)]
    BC = pd.DataFrame(rows)
    a, b = cmp_frames("table1_bucket_counts", BC,
                      pd.read_csv(rd / "tables/table1_bucket_counts.csv"),
                      ["horizon"], ["points", "learners"], out)
    ok += a; bad += b

    # --- Table 2: predictive ----------------------------------------------------
    rows = []
    for k in MODEL_ORDER:
        r = evaluate(PRED[k][M], Y[M], cfg["ece_bins"])
        r["model"] = k
        rows.append(r)
    T2 = pd.DataFrame(rows)
    a, b = cmp_frames("table2_predictive", T2,
                      pd.read_csv(rd / "tables/table2_predictive.csv"),
                      ["model"], ["n", "auc", "acc", "rmse", "nll", "ece", "brier"], out)
    ok += a; bad += b

    # --- Table 9: post-gap AUC by model ----------------------------------------
    rows = []
    for name in MODEL_ORDER:
        p = PRED[name]
        rec = dict(model=name)
        for i, (k, _, _) in enumerate(BUCKETS):
            sel = (B == i) & M
            y = Y[sel]
            rec[f"auc_{k}"] = (float(roc_auc_score(y, p[sel]))
                               if sel.sum() >= cfg["min_points_bucket"] and len(np.unique(y)) > 1
                               else np.nan)
            rec[f"n_{k}"] = int(sel.sum())
        real = [rec[f"auc_{k}"] for k in REAL_HORIZONS if np.isfinite(rec.get(f"auc_{k}", np.nan))]
        rec["auc_mean_postgap"] = float(np.mean(real)) if real else np.nan
        rows.append(rec)
    T9 = pd.DataFrame(rows)
    T9["rank_within"] = T9["auc_within"].rank(ascending=False, method="min")
    T9["rank_mean_postgap"] = T9["auc_mean_postgap"].rank(ascending=False, method="min")
    cols9 = ([f"auc_{k}" for k, _, _ in BUCKETS] + [f"n_{k}" for k, _, _ in BUCKETS]
             + ["auc_mean_postgap", "rank_within", "rank_mean_postgap"])
    a, b = cmp_frames("table9_postgap", T9,
                      pd.read_csv(rd / "tables/table9_postgap_auc_by_model.csv"),
                      ["model"], cols9, out)
    ok += a; bad += b

    # --- strongest baseline / H1 ------------------------------------------------
    t2f = pd.read_csv(rd / "tables/table2_predictive.csv")
    strongest = t2f[t2f.model.isin(BASELINES)].sort_values("auc", ascending=False).iloc[0]["model"]
    if strongest != mani.get("strongest_baseline"):
        out.append(f"    STRONGEST BASELINE mismatch: recomputed {strongest} vs manifest "
                   f"{mani.get('strongest_baseline')}")
        bad += 1
    else:
        ok += 1
    COMPARATORS = list(dict.fromkeys(["Control", "DKT-F", strongest]))

    # --- Table 6: educational utility -------------------------------------------
    rows = []
    for comp in COMPARATORS:
        for bid, (k, _, _) in enumerate(BUCKETS):
            sel = (B == bid) & M
            if sel.sum() < cfg["min_points_bucket"]:
                continue
            for thr in cfg["utility_thresholds"]:
                up = utility(PRED["PCDT"][sel], Y[sel], thr)
                uc = utility(PRED[comp][sel], Y[sel], thr)
                rows.append(dict(horizon=k, comparator=comp, threshold=thr,
                                 extra_correctly_flagged=up["correctly_flagged"] - uc["correctly_flagged"],
                                 extra_false_alarms=up["false_alarms"] - uc["false_alarms"],
                                 fewer_missed=uc["missed"] - up["missed"],
                                 d_precision=up["precision"] - uc["precision"],
                                 d_recall=up["recall"] - uc["recall"], d_f1=up["f1"] - uc["f1"],
                                 strict_win=bool(up["correctly_flagged"] > uc["correctly_flagged"]
                                                 and up["false_alarms"] <= uc["false_alarms"])))
    T6 = pd.DataFrame(rows)
    a, b = cmp_frames("table6_utility", T6,
                      pd.read_csv(rd / "tables/table6_educational_utility.csv"),
                      ["horizon", "comparator", "threshold"],
                      ["extra_correctly_flagged", "extra_false_alarms", "fewer_missed",
                       "d_precision", "d_recall", "d_f1"], out)
    ok += a; bad += b

    # --- Table 5: retention (DiD) ----------------------------------------------
    rows = []
    for comp in COMPARATORS:
        for bid, (k, _, _) in enumerate(BUCKETS):
            if k not in REAL_HORIZONS:
                continue
            r = did_retention(PRED["PCDT"], PRED[comp], bid, M, B, Y, cfg,
                              n_boot=(None if DO_BOOT else 0))
            r.update(horizon=k, comparator=comp,
                     source="real" if r["adequate"] else "INADEQUATE")
            rows.append(r)
    T5 = pd.DataFrame(rows)
    c5 = ["adequate", "n_learners", "n_points", "retention_advantage",
          "auc_pcdt", "auc_comparator"] + (["ci_lo", "ci_hi"] if DO_BOOT else [])
    a, b = cmp_frames("table5_persistence", T5,
                      pd.read_csv(rd / "tables/table5_persistence.csv"),
                      ["horizon", "comparator"], [c for c in c5 if c in T5.columns], out)
    ok += a; bad += b

    # --- Table 3 + 7: paired bootstrap (expensive; opt-in) ----------------------
    if DO_BOOT:
        rows_idx = np.repeat(np.arange(PRED["PCDT"].shape[0])[:, None],
                             PRED["PCDT"].shape[1], 1)[M]
        stats = []
        for k in MODEL_ORDER:
            if k == "PCDT":
                continue
            d, lo, hi, pv = paired_bootstrap(PRED["PCDT"][M], PRED[k][M], Y[M], rows_idx,
                                             cfg["n_bootstrap"], cfg["seed"])
            stats.append(dict(comparison=f"PCDT - {k}", delta_auc=d, ci_lo=lo, ci_hi=hi, p=pv))
        T3 = pd.DataFrame(stats)
        a, b = cmp_frames("table3_statistics", T3,
                          pd.read_csv(rd / "tables/table3_statistics.csv"),
                          ["comparison"], ["delta_auc", "ci_lo", "ci_hi", "p"], out)
        ok += a; bad += b

    grand_ok += ok
    grand_bad += bad
    report.append(f"{rd.name:34s} checks={ok:5d} FAIL={bad}")
    report.extend(out)

open("recompute_log.txt","a").write("\n".join(report)+"\n")
print("\n".join(report))
print(f"\n=== TOTAL: {grand_ok} cell checks recomputed from predictions, {grand_bad} mismatches ===")
