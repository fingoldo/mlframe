"""Bench + A/B for the screen_predictors whole-matrix-copy elimination (mutate-and-restore).

Emits, for a grid of (seed, shape, n_jobs), the MRMR selected support + fit wall so OLD (baseline worktree) vs NEW can be
diffed. ``n_jobs=1`` forces the SERIAL confirm path (the one whose null-draw semantics change: restore-to-pristine each call
vs the old progressively-shuffled shared buffer); ``n_jobs>=2`` forces the PARALLEL path (already per-worker copy, unchanged).

  1. SELECTION-EQUIVALENCE: run in both trees, diff the SUPPORT lines (serial old vs new; parallel old vs new).
  2. RAM: ``--ram`` prints the O(p*n) -> O((|x|+|y|)*n) contrast at a representative shape.

Usage:
    PYTHONPATH=<tree>/src python bench_screen_mutate_restore_ram.py --grid
    PYTHONPATH=<tree>/src python bench_screen_mutate_restore_ram.py --ram
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def _make_data(n=3000, p=24, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 4, size=(n, p)).astype(np.float64)
    y = (X[:, 0] > 1).astype(int) ^ (X[:, 1] > 1).astype(int)
    y = y ^ (X[:, 5] > 2).astype(int)
    # a marginally-informative column to create borderline candidates that a null-draw change could flip
    flip = rng.random(n) < 0.35
    y2 = y.copy()
    y2[flip] = (X[flip, 9] > 2).astype(int)
    return X, y2.astype(np.int64)


def _fit_support(seed, n, p, n_jobs):
    from mlframe.feature_selection.filters import MRMR

    X, y = _make_data(n=n, p=p, seed=seed)
    Xdf = pd.DataFrame(X, columns=[f"f{i}" for i in range(p)])
    t0 = time.perf_counter()
    m = MRMR(
        full_npermutations=20,
        baseline_npermutations=10,
        extra_x_shuffling=True,
        verbose=0,
        n_jobs=n_jobs,
        random_seed=seed + 1000,
    )
    m.fit(Xdf, y)
    wall = time.perf_counter() - t0
    support = sorted(int(i) for i in np.asarray(m.support_).ravel())
    return support, wall


def run_grid():
    configs = []
    for seed in (0, 1, 2, 3, 4):
        for n, p in ((3000, 24), (2000, 18)):
            configs.append((seed, n, p, 1))  # serial confirm path
    for seed in (0, 1):
        configs.append((seed, 3000, 24, 2))  # parallel confirm path
    for seed, n, p, nj in configs:
        try:
            support, wall = _fit_support(seed, n, p, nj)
            path = "serial" if nj == 1 else "parallel"
            print(f"CFG seed={seed} n={n} p={p} path={path} SUPPORT={support} WALL={wall:.2f}")
            sys.stdout.flush()
        except Exception as e:
            print(f"CFG seed={seed} n={n} p={p} nj={nj} ERROR={type(e).__name__}: {e}")
            sys.stdout.flush()


def run_ram():
    n, p = 200_000, 500
    x_cols, y_cols = 1, 1
    bytes_per = 8
    matrix_copy = n * p * bytes_per
    per_col = n * (x_cols + y_cols) * bytes_per
    print("RAM contrast (float64):")
    print(f"  frame shape                (p={p}, n={n})")
    print(f"  OLD whole-matrix copy      {matrix_copy/1e6:.1f} MB   (O(p*n))")
    print(f"  NEW per-column save        {per_col/1e6:.3f} MB   (O((|x|+|y|)*n))")
    print(f"  reduction factor           {matrix_copy/per_col:.0f}x")


if __name__ == "__main__":
    if "--ram" in sys.argv:
        run_ram()
    else:
        run_grid()
