"""A/B: does materialising the top-k COLUMN NAMES dominate ``row_wise_top_k_extreme_columns``?

The suite's only caller (``_pipeline_extensions._extreme_scores_only``) keeps the ``topN_score`` columns and
throws the ``topN_column`` name columns away -- but the function builds them anyway, which on a 2.18M-row frame
means materialising millions of python string objects into an object-dtype array nobody reads. The production
log spent 45.2s in this stage.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_row_wise_extreme_scores_only``
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_engineering.row_wise_extremality import (
    _compute_extremality_matrix,
    row_wise_top_k_extreme_columns,
)


def make_frame(n_rows: int, n_cols: int, nan_frac: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """A float frame with realistic NaN density."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_rows, n_cols))
    data[rng.random(data.shape) < nan_frac] = np.nan
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(n_cols)])


def full_output(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """What the suite calls today: scores AND the name columns it then discards."""
    out = row_wise_top_k_extreme_columns(df, k=k)
    return out[[c for c in out.columns if c.endswith("_score")]]


def scores_only(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """The same scores with no name materialisation at all."""
    extremality, _cols = _compute_extremality_matrix(df, None)
    n_rows, n_cols = extremality.shape
    k = min(k, n_cols)
    sort_key = np.where(np.isnan(extremality), -1.0, extremality)
    if k < n_cols:
        candidates = np.argpartition(-sort_key, kth=k - 1, axis=1)[:, :k]
    else:
        candidates = np.tile(np.arange(n_cols), (n_rows, 1))
    candidate_scores = np.take_along_axis(sort_key, candidates, axis=1)
    local_order = np.argsort(-candidate_scores, axis=1, kind="quicksort")
    top_scores = np.take_along_axis(candidate_scores, local_order, axis=1)
    top_scores = np.where(top_scores < 0.0, np.nan, top_scores)
    return pd.DataFrame({f"top{i + 1}_score": top_scores[:, i] for i in range(k)}, index=df.index)


def bench(n_rows: int = 400_000, n_cols: int = 85, k: int = 3, repeats: int = 3) -> None:
    """Warm once, report best-of-N for both, and confirm the scores are identical."""
    df = make_frame(n_rows, n_cols)
    for fn in (full_output, scores_only):
        fn(df.head(2000), k)  # warm

    times = {}
    for name, fn in (("full (names built)", full_output), ("scores only", scores_only)):
        best = min(_time(fn, df, k) for _ in range(repeats))
        times[name] = best
        print(f"{name:20s} {best:7.3f}s")
    print(f"\nspeedup: {times['full (names built)'] / times['scores only']:.2f}x at {n_rows:,} x {n_cols}, k={k}")

    a, b = full_output(df, k), scores_only(df, k)
    diffs = [np.nanmax(np.abs(a.iloc[:, i].to_numpy() - b.iloc[:, i].to_numpy())) for i in range(k)]
    print("max|diff| per score column:", ", ".join(f"{d:.3e}" for d in diffs))


def _time(fn, df, k) -> float:
    """Wall time of one call."""
    t0 = time.perf_counter()
    fn(df, k)
    return time.perf_counter() - t0


if __name__ == "__main__":
    bench()
