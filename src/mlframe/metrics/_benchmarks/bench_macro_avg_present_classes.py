"""qual-7 bench: macro precision/recall/F1 over PRESENT classes vs over ALL declared classes.

Ground truth = ``sklearn.metrics.classification_report`` macro avg, whose label set is the union of classes
present in ``y_true`` or ``y_pred`` (declared-but-absent classes are excluded). mlframe's
``fast_classification_report`` divides the macro mean by ``nclasses`` -- so a class that appears in NEITHER
y_true nor y_pred contributes a zeroed precision/recall/F1, DEFLATING the macro averages. ``balanced_accuracy``
in the same kernel was already fixed to exclude absent classes (support==0); macro P/R/F1 were not.

This bench measures |mlframe_macro - sklearn_macro| for the OLD (divide-by-nclasses) and NEW (mean over classes
present in y_true OR y_pred) policies across multiple seeds and scenarios with declared-but-absent classes.

Run: python -m mlframe.metrics._benchmarks.bench_macro_avg_present_classes
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def _old_macro(yt, yp, nclasses):
    supports = np.zeros(nclasses, dtype=np.int64)
    allpreds = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)
    for t, p in zip(yt, yp):
        if 0 <= t < nclasses:
            supports[t] += 1
        if 0 <= p < nclasses:
            allpreds[p] += 1
            if p == t:
                hits[p] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        recalls = hits / supports
        precisions = hits / allpreds
        f1s = 2 * (precisions * recalls) / (precisions + recalls)
    out = []
    for arr in (precisions, recalls, f1s):
        a = np.nan_to_num(arr.copy(), nan=0.0)
        out.append(a.mean())  # divide by nclasses (OLD)
    return out


def _new_macro(yt, yp, nclasses):
    supports = np.zeros(nclasses, dtype=np.int64)
    allpreds = np.zeros(nclasses, dtype=np.int64)
    hits = np.zeros(nclasses, dtype=np.int64)
    for t, p in zip(yt, yp):
        if 0 <= t < nclasses:
            supports[t] += 1
        if 0 <= p < nclasses:
            allpreds[p] += 1
            if p == t:
                hits[p] += 1
    present = (supports > 0) | (allpreds > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        recalls = hits / supports
        precisions = hits / allpreds
        f1s = 2 * (precisions * recalls) / (precisions + recalls)
    out = []
    for arr in (precisions, recalls, f1s):
        a = np.nan_to_num(arr.copy(), nan=0.0)
        out.append(a[present].mean() if present.any() else 0.0)
    return out


def _sklearn_macro(yt, yp):
    labels = np.unique(np.concatenate([yt, yp]))
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=labels, average="macro", zero_division=0)
    return [p, r, f]


def _gen(scenario, nclasses, n, rng):
    """Generate (y_true, y_pred) where some declared classes are absent."""
    if scenario == "one_absent":
        active = nclasses - 1  # top class never appears
    elif scenario == "two_absent":
        active = max(2, nclasses - 2)
    elif scenario == "half_absent":
        active = max(2, nclasses // 2)
    else:
        raise ValueError(scenario)
    yt = rng.integers(0, active, size=n)
    # predictions: mostly correct with noise, but only among active classes
    yp = yt.copy()
    flip = rng.random(n) < 0.35
    yp[flip] = rng.integers(0, active, size=int(flip.sum()))
    return yt.astype(np.int64), yp.astype(np.int64)


def main():
    scenarios = ["one_absent", "two_absent", "half_absent"]
    seeds = list(range(8))
    configs = [(6, 400), (10, 1000)]  # (nclasses_declared, n)
    print("scen | nclasses n | seed | old|err| new|err|  (avg over P/R/F1 vs sklearn macro)")
    win_old = win_new = 0
    agg = {}
    for nclasses, n in configs:
        for scen in scenarios:
            for seed in seeds:
                rng = np.random.default_rng(seed * 131 + nclasses * 7 + len(scen))
                yt, yp = _gen(scen, nclasses, n, rng)
                gt = np.array(_sklearn_macro(yt, yp))
                old = np.array(_old_macro(yt, yp, nclasses))
                new = np.array(_new_macro(yt, yp, nclasses))
                eo = float(np.mean(np.abs(old - gt)))
                en = float(np.mean(np.abs(new - gt)))
                if en < eo:
                    win_new += 1
                elif eo < en:
                    win_old += 1
                agg.setdefault((nclasses, scen), [0.0, 0.0, 0])
                agg[(nclasses, scen)][0] += eo
                agg[(nclasses, scen)][1] += en
                agg[(nclasses, scen)][2] += 1
                print(f"{scen:11s}| {nclasses:2d} {n:5d} | {seed:2d} | {eo:.4f}  {en:.4f}")
    print("\nper-scenario mean |err| (old -> new):")
    for (nc, scen), (so, sn, c) in sorted(agg.items()):
        print(f"  nclasses={nc} {scen:11s}: {so / c:.4f} -> {sn / c:.4f}")
    total = win_old + win_new
    print(f"\nper-cell wins  new/old = {win_new}/{win_old}  (of {total} cells; ties excluded)")


if __name__ == "__main__":
    main()
