"""Bench the uncapped OOF MDLP refit cost in ``generate_target_aware_group_bins``
(CPX5, 2026-06-23).

The OOF leak-safe binning fits one Fayyad-Irani MDLP per (fold, group) over
``n_folds x n_groups`` qualifying groups, UNCAPPED. This bench measures the wall
cost at a realistic shape (n=100k, n_groups in {50,100,200}, n_folds=5) to
document how bad the explosion is, AND quantifies how often a group's per-group
OOF edges DIFFER from the pooled global edges -- the evidence that any "cap to
global fallback for most groups" approach would CHANGE the OOF bin values (and
therefore is NOT identity- or selection-safe). See the FUTURE note at the call
site in ``_grouped_quantile_fe.py``.

Run:  python -m mlframe.feature_selection.filters._benchmarks.bench_grouped_quantile_fe
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._grouped_quantile_fe import (
    generate_target_aware_group_bins,
    _fit_group_edges,
    _MIN_GROUP_SIZE,
)


def _make_data(n: int, n_groups: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, size=n)
    # Per-group location/scale so per-group MDLP edges genuinely differ from global.
    loc = rng.normal(0, 5, size=n_groups)[g]
    scale = (0.5 + rng.random(n_groups))[g]
    x = loc + scale * rng.normal(size=n)
    # Target depends on WITHIN-group position (the signal target-aware bins capture).
    z = (x - loc) / scale
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.random(n) < p).astype(np.int64)
    X = pd.DataFrame({"grp": g, "num": x})
    return X, y


def _median_time(fn, repeats=3):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _edges_differ_fraction(X, y, n_bins=5):
    """Fraction of qualifying groups whose all-rows per-group MDLP edges differ
    from the pooled global edges -- a lower bound on how many OOF assignments a
    'use global for most groups' cap would change."""
    x = X["num"].to_numpy(np.float64)
    fin = np.isfinite(x)
    glob = _fit_group_edges(x[fin], y[fin], n_bins)
    g = X["grp"].to_numpy()
    n_qual = 0
    n_diff = 0
    for key in np.unique(g):
        rows = np.where(g == key)[0]
        if rows.size < _MIN_GROUP_SIZE:
            continue
        n_qual += 1
        e = _fit_group_edges(x[rows], y[rows], n_bins)
        if e.shape != glob.shape or not np.allclose(e, glob):
            n_diff += 1
    return n_qual, n_diff


def main():
    n = 100_000
    print(f"n={n}, n_folds=5, _MIN_GROUP_SIZE={_MIN_GROUP_SIZE}")
    print(f"{'n_groups':>9} | {'wall_s (median of 3)':>20} | {'qual_groups':>11} | {'edges_differ':>12}")
    print("-" * 64)
    for n_groups in (50, 100, 200):
        X, y = _make_data(n, n_groups)
        # warm
        generate_target_aware_group_bins(X, y, ["grp"], ["num"], n_bins=5, n_folds=5)
        t = _median_time(
            lambda: generate_target_aware_group_bins(X, y, ["grp"], ["num"], n_bins=5, n_folds=5)
        )
        n_qual, n_diff = _edges_differ_fraction(X, y)
        frac = (n_diff / n_qual) if n_qual else 0.0
        print(f"{n_groups:>9} | {t:>20.3f} | {n_qual:>11} | {n_diff:>5}/{n_qual:<3} ({frac:.0%})")


if __name__ == "__main__":
    main()
