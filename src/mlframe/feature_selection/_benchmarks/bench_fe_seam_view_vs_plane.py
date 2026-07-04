"""Bench: FE seed-ranking MI on ONE contiguous f32 plane (batched kernel) vs per-column zero-copy views (k dispatches).

Decides the FE seam design (matrix-native replatform, P-seam): when a polars/pandas frame feeds the FE families, do we
(A) assemble one contiguous f32 block and call the batched MI kernel once, or (B) loop per-column numpy views and call the
MI kernel k times -- avoiding a contiguous copy. Memory is ~equal (both materialise ~n_sub x k values for the subsample
decision), so this measures the SPEED delta of batched-vs-looped dispatch plus the pandas-vs-polars column-extraction cost.

Run: python -m mlframe.feature_selection._benchmarks.bench_fe_seam_view_vs_plane
Shapes mirror the wellbore geosteering fit: ~496 numeric columns, a 30k-row FE decision subsample, 773-ish groups (groups
do not affect this kernel). Reports best-of-N warm wall for each strategy on both pandas and polars sources.
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import plugin_mi_classif_batch_dispatch

N_SUB = 30_000
K_COLS = 496
N_BINS = 20
REPEATS = 7


def _synth(n, k, seed=0):
    rng = np.random.default_rng(seed)
    # float32 columns with a few informative ones (so MI ranking is non-degenerate); mirrors densified numeric FE input.
    X = rng.standard_normal((n, k)).astype(np.float32)
    y = (X[:, 0] * 1.5 + X[:, 7] * X[:, 11] + rng.standard_normal(n) * 0.3)
    y = np.digitize(y, np.quantile(y, np.linspace(0, 1, 11)[1:-1])).astype(np.int64)
    return X, y


def _best(fn, repeats=REPEATS):
    fn()  # warm (numba JIT / cache)
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def _strategy_plane(frame_cols, y, cols):
    """One contiguous f32 (n,k) block -> one batched dispatch."""
    def run():
        dense = frame_cols()  # (n, k) f32 contiguous
        return plugin_mi_classif_batch_dispatch(dense, y, N_BINS)
    return run


def _strategy_views(col_view, y, cols):
    """Per-column zero-copy view -> k dispatches on (n,1)."""
    def run():
        mis = np.empty(len(cols), dtype=np.float64)
        for j, c in enumerate(cols):
            v = col_view(c)  # zero-copy view where the column is contiguous/native-dtype
            mis[j] = plugin_mi_classif_batch_dispatch(v.reshape(-1, 1), y, N_BINS)[0]
        return mis
    return run


def main():
    import pandas as pd
    X, y = _synth(N_SUB, K_COLS)
    cols = [f"f{i}" for i in range(K_COLS)]

    pdf = pd.DataFrame(X, columns=cols)
    results = {}

    # pandas source
    results["pandas/plane"] = _best(_strategy_plane(lambda: pdf[cols].to_numpy(np.float32), y, cols))
    results["pandas/views"] = _best(_strategy_views(lambda c: pdf[c].to_numpy(), y, cols))

    try:
        import polars as pl
        ldf = pl.DataFrame({c: X[:, j] for j, c in enumerate(cols)})
        results["polars/plane"] = _best(_strategy_plane(lambda: ldf.select(cols).to_numpy().astype(np.float32, copy=False), y, cols))
        results["polars/views"] = _best(_strategy_views(lambda c: ldf[c].to_numpy(), y, cols))
    except Exception as exc:
        print(f"polars unavailable: {exc!r}")

    # parity: both strategies must produce the same MI ranking (selection-equivalence gate)
    mi_plane = plugin_mi_classif_batch_dispatch(pdf[cols].to_numpy(np.float32), y, N_BINS)
    mi_views = np.array([plugin_mi_classif_batch_dispatch(pdf[c].to_numpy().reshape(-1, 1), y, N_BINS)[0] for c in cols])
    top_plane = set(np.argsort(mi_plane)[-8:])
    top_views = set(np.argsort(mi_views)[-8:])

    # f32 vs f64 CEILING: the MI seed-ranking on the contiguous plane in each precision. This is the MAX the
    # MLFRAME_FE_SUBSAMPLE_DTYPE=float32 flag could save IF the family bodies consumed f32 (today they hardcode
    # .to_numpy(np.float64), so the realised win needs that upcast removed -- the seam). Selection must stay equivalent.
    Xf64 = pdf[cols].to_numpy(np.float64)
    Xf32 = pdf[cols].to_numpy(np.float32)
    t_f64 = _best(lambda: plugin_mi_classif_batch_dispatch(Xf64, y, N_BINS))
    t_f32 = _best(lambda: plugin_mi_classif_batch_dispatch(Xf32, y, N_BINS))
    mi_f64 = plugin_mi_classif_batch_dispatch(Xf64, y, N_BINS)
    mi_f32 = plugin_mi_classif_batch_dispatch(Xf32, y, N_BINS)
    top_f64, top_f32 = set(np.argsort(mi_f64)[-8:]), set(np.argsort(mi_f32)[-8:])

    print(f"\nFE seam bench  n_sub={N_SUB} k={K_COLS} bins={N_BINS} best-of-{REPEATS}")
    for k, v in results.items():
        print(f"  {k:16s} {v * 1e3:8.1f} ms")
    if "pandas/plane" in results and "pandas/views" in results:
        print(f"  views/plane ratio (pandas): {results['pandas/views'] / results['pandas/plane']:.2f}x")
    print(f"  top-8 seed set identical (plane vs views): {top_plane == top_views}  "
          f"max|MI_plane-MI_views|={np.abs(mi_plane - mi_views).max():.2e}")
    print(f"\n  MI-ranking f64={t_f64 * 1e3:.1f} ms  f32={t_f32 * 1e3:.1f} ms  "
          f"f32 speedup={t_f64 / t_f32:.2f}x  top-8 seed identical(f32 vs f64)={top_f32 == top_f64}")


if __name__ == "__main__":
    main()
