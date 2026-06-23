"""Bench: error_bias_per_feature densifies the WHOLE feature matrix then uses only max_features columns.

``error_bias_per_feature`` overlays the value distribution of at most ``max_features`` (default 4) feature columns
across OVER / UNDER / MAJORITY error groups. It calls ``_resolve_feature_matrix`` which column-stacks EVERY feature
column into one dense float64 matrix (O(n*cols)), then reads only ``mat[:, sel]`` for ``sel`` = the first
``max_features`` columns (or a named subset). At the diagnostics shape (100k rows x DIAG_MAX_FEATURES=200 cols) that
is ~100k*200 float64 (160 MB) built and discarded so 4 columns can be histogrammed. Pulling only the needed columns
is bit-identical by construction (same label-encoding, same finite filter, same histogram edges).

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.reporting.charts._benchmarks.bench_error_bias_column_pull
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.reporting.charts.error_analysis import error_bias_per_feature


def _make_frame(n: int, cols: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = {f"f{j}": rng.standard_normal(n) for j in range(cols)}
    X = pd.DataFrame(data)
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * 0.3
    return X, y_true, y_pred


def _bench(n: int, cols: int, repeat: int = 7):
    X, y_true, y_pred = _make_frame(n, cols)
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = error_bias_per_feature(X, y_true, y_pred, max_features=4)
        best = min(best, time.perf_counter() - t0)
    return best, out


if __name__ == "__main__":
    for n, cols in [(100_000, 200), (100_000, 50), (50_000, 200), (10_000, 200)]:
        t, out = _bench(n, cols)
        print(f"n={n:>7} cols={cols:>4}  error_bias_per_feature best-of-7 = {t*1000:8.2f} ms  panels={len(out.group_means)}")
