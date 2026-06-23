"""Bench: worst_k_table densifies the WHOLE feature matrix then uses only top_fi columns at K rows.

``worst_k_table`` finds the K worst-error rows (K<=20) and surfaces the ``top_fi`` (default 5) highest-importance
feature values for each. It calls ``_resolve_feature_matrix`` which column-stacks EVERY feature column into one dense
float64 matrix (O(n*cols)), then reads ``mat[sel, fi_cols]`` -- a K x top_fi slice. At the diagnostics row cap
(100k rows) and a few-hundred-column engineered frame, that is ~100k*200 float64 (160 MB) built and discarded so a
20x5 slice can be read. Pulling only the needed columns at the needed rows is bit-identical by construction.

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.reporting.charts._benchmarks.bench_worst_k_column_pull
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.reporting.charts.error_analysis import worst_k_table


def _make_frame(n: int, cols: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = {f"f{j}": rng.standard_normal(n) for j in range(cols)}
    X = pd.DataFrame(data)
    y_true = rng.standard_normal(n)
    y_pred = y_true + rng.standard_normal(n) * 0.3
    fi = rng.random(cols)
    return X, y_true, y_pred, fi


def _bench(n: int, cols: int, repeat: int = 7) -> float:
    X, y_true, y_pred, fi = _make_frame(n, cols)
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = worst_k_table(X, y_true, y_pred, task="regression", k=20, feature_importances=fi, top_fi=5)
        best = min(best, time.perf_counter() - t0)
    return best, out


if __name__ == "__main__":
    for n, cols in [(100_000, 200), (100_000, 50), (50_000, 200), (10_000, 200)]:
        t, out = _bench(n, cols)
        print(f"n={n:>7} cols={cols:>4}  worst_k_table best-of-7 = {t*1000:8.2f} ms  rows_returned={len(out.table)}")
