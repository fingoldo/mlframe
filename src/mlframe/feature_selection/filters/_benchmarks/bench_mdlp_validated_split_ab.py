"""A/B/C/D bench for the MDLP validated-split research prototype (2026-07-19).

Compares, on both a controlled synthetic setup (where the ground truth relationship between x and y
is known, so "does the extra resolution reflect real signal or noise" has a checkable answer) and the
real wellbore target column:

  (a) quantile-fallback baseline   -- what MDLP falls back to when it returns empty edges (pre-fix
                                       bug behaviour approximated directly, not by re-running the bug)
  (b) uncapped classic MDLP        -- max_depth=8 (classification-target default), no y-cap-aware
                                       depth reduction; the configuration that produced the 20x
                                       feature blowup / worse OOS RMSE on the real wellbore fit
  (c) depth-capped classic MDLP    -- the landed fix: max_depth = ceil(log2(max_y_classes))
  (d) validated-split MDLP         -- this module's significance-gated prototype, run at the
                                       UNCAPPED max_depth=8 (the whole point is that the significance
                                       gate should reject the noise-driven deep splits on its own,
                                       without an arbitrary depth cap)

Metric: edges/bins produced, wall time, and out-of-sample MSE of the simplest possible bin-conditional
predictor (per-bin train-y mean, scored on held-out rows) -- isolates the discretization's own
generalization quality without the cost of a full MRMR/FE/GBM pipeline (which is not reproducible in
this session's budget -- the original wellbore_train.py suite harness lives outside the repo and a
full MRMR.fit is 130-500s per config; this bench isolates the SAME phenomenon -- in-sample-only split
acceptance overfitting a noisy/pseudo-classed target -- at a cost of seconds, not fit the full pipeline).

Run: python src/mlframe/feature_selection/filters/_benchmarks/bench_mdlp_validated_split_ab.py
"""
from __future__ import annotations

import math
import time

import numpy as np

from mlframe.feature_selection.filters.supervised_binning import mdlp_bin_edges
from mlframe.feature_selection.filters._mdlp_validated_split import mdlp_bin_edges_validated
from mlframe.feature_selection.filters._adaptive_nbins import _edges_from_quantiles


def _oos_mse(x_train, y_train, x_test, y_test, edges):
    """Bin-conditional-mean predictor: fit per-bin mean(y) on TRAIN using ``edges``, score MSE on TEST."""
    inner = edges[1:-1] if edges.size >= 2 else edges
    inner = inner[np.isfinite(inner)]
    codes_train = np.searchsorted(inner, x_train, side="right")
    codes_test = np.searchsorted(inner, x_test, side="right")
    n_bins = int(inner.size) + 1
    means = np.full(n_bins, float(np.mean(y_train)))
    for b in range(n_bins):
        m = codes_train == b
        if m.any():
            means[b] = float(np.mean(y_train[m]))
    pred = means[np.clip(codes_test, 0, n_bins - 1)]
    return float(np.mean((pred - y_test) ** 2)), n_bins


def _run_one(name, fn, x_train, y_train, x_test, y_test):
    t0 = time.perf_counter()
    edges = fn()
    wall = time.perf_counter() - t0
    mse, n_bins = _oos_mse(x_train, y_train, x_test, y_test, edges)
    print(f"  {name:28s} wall={wall*1000:8.2f}ms  bins={n_bins:5d}  OOS_MSE={mse:12.4f}  OOS_RMSE={math.sqrt(mse):10.4f}")
    return dict(name=name, wall=wall, bins=n_bins, mse=mse, rmse=math.sqrt(mse))


def bench_case(label, x, y, *, max_y_classes=64):
    n = x.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_test = n // 4
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    x_tr, y_tr = x[train_idx], y[train_idx]
    x_te, y_te = x[test_idx], y[test_idx]

    print(f"\n=== {label}  (n={n}, train={train_idx.size}, test={test_idx.size}) ===")
    depth_cap = max(1, int(math.ceil(math.log2(max_y_classes))))

    results = []
    results.append(_run_one("(a) quantile fallback (5 bins)", lambda: np.concatenate([[-np.inf], _edges_from_quantiles(x_tr, 5), [np.inf]]), x_tr, y_tr, x_te, y_te))
    results.append(_run_one("(b) uncapped MDLP (depth=8)", lambda: mdlp_bin_edges(x_tr, y_tr, max_depth=8, max_y_classes=max_y_classes), x_tr, y_tr, x_te, y_te))
    results.append(_run_one(f"(c) depth-capped MDLP (depth={depth_cap})", lambda: mdlp_bin_edges(x_tr, y_tr, max_depth=depth_cap, max_y_classes=max_y_classes), x_tr, y_tr, x_te, y_te))
    results.append(_run_one("(d) validated MDLP (analytic+perm, a=.05)", lambda: mdlp_bin_edges_validated(x_tr, y_tr, max_depth=8, max_y_classes=max_y_classes, alpha=0.05, n_permutations=30), x_tr, y_tr, x_te, y_te))
    results.append(_run_one("(d') validated MDLP (bonferroni a=.05)", lambda: mdlp_bin_edges_validated(x_tr, y_tr, max_depth=8, max_y_classes=max_y_classes, alpha=0.05, n_permutations=30, bonferroni=True), x_tr, y_tr, x_te, y_te))
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # --- Synthetic 1: PURE NOISE, high-cardinality continuous "target" (the exact failure mode:
    # x has zero real relationship to y once y is pseudo-classed; a valid criterion should end up
    # close to the 1-bin / no-split answer, an overfit one keeps carving noise). ---
    n = 50_000
    x_noise = rng.standard_normal(n)
    y_noise = rng.standard_normal(n) * 1000.0 + rng.uniform(-1, 1, n)  # continuous, ~50k unique, independent of x
    bench_case("Synthetic: pure noise (x _|_ y, continuous high-card y)", x_noise, y_noise)

    # --- Synthetic 2: REAL signal, continuous target, still high-cardinality (y = f(x) + noise) so
    # max_y_classes quantization still engages -- a valid criterion should keep splitting near the
    # true breakpoints and should NOT lose accuracy vs the depth-capped baseline. ---
    x_sig = rng.uniform(-5, 5, n)
    y_sig = np.where(x_sig < -1.5, 10.0, np.where(x_sig < 2.0, 30.0, 5.0)) + rng.standard_normal(n) * 2.0
    bench_case("Synthetic: real 3-piece signal + noise (continuous y)", x_sig, y_sig)

    # --- Real wellbore data: TVT target vs a genuinely continuous, high-signal feature (GR) and a
    # near-noise engineered feature, at the same ~45-50k row scale the production bug was found at. ---
    try:
        import polars as pl
        from os.path import join

        DATA_DIR = r"C:\Users\Admin\Machine learning\data\Competitions\ROGII - Wellbore Geology Prediction"
        lf = pl.scan_parquet(join(DATA_DIR, "train_df.parquet")).filter(~pl.col("TVT").is_null())
        counts = lf.group_by("well_id").agg(pl.len().alias("n")).sort("well_id").collect()
        counts = counts.with_columns(pl.col("n").cum_sum().alias("cum"))
        keep = counts.filter(pl.col("cum") <= 50_000)["well_id"].to_list()
        df = lf.filter(pl.col("well_id").is_in(keep)).collect()
        y_real = df["TVT"].to_numpy().astype(np.float64)
        gr = df["GR"].to_numpy().astype(np.float64)
        mask = np.isfinite(gr) & np.isfinite(y_real)
        bench_case("REAL wellbore: GR (continuous feature) vs TVT", gr[mask], y_real[mask])

        # An engineered-looking near-noise-relative-to-TVT column, if present, to probe the blowup case directly.
        for cand in ("GR_diff_5", "GR_lag_10"):
            if cand in df.columns:
                xc = df[cand].to_numpy().astype(np.float64)
                m2 = np.isfinite(xc) & np.isfinite(y_real)
                bench_case(f"REAL wellbore: {cand} vs TVT", xc[m2], y_real[m2])
                break
    except Exception as exc:
        print(f"[skip real-data section] {exc!r}")
