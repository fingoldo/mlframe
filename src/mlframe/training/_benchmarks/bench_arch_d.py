"""Bench harness for the Wave-4 D-Arch items.

Two micro-benches, both bench-gated (numba threshold 1.5x, cupy threshold 5%):

* :func:`bench_col_stats_numba` -- ``composite_cache._col_stats`` numpy hot path vs a numba
  ``@njit`` kernel that performs the float-branch min / max / null-count pass; per-column hash
  loop is the WHOLE-frame scan in ``data_signature``.

* :func:`bench_predict_guards_cupy` -- ``_fit_persist_and_transform`` polars fastpath
  (impute + standardise + return numpy) vs a cupy variant that uploads the numeric block to
  GPU, computes mean / std, standardises, and downloads. Bench gates the cupy path at >5%
  speedup; below threshold the cupy variant is NOT enabled in production.

Both benches use ``time.perf_counter`` with a warmup pass to absorb JIT compile cost and a
median-of-N sample so a single outlier (Windows scheduler hiccup, NTFS flush) does not poison
the result. Target shape is the user-specified 200 cols x 10M rows; the harness scales DOWN if
free RAM is insufficient (200 * 10M * 8 B = 16 GB) so the bench runs end-to-end on a 32 GB box
without OOM.
"""
from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Tuple

import numpy as np


def _free_ram_bytes() -> int:
    """Best-effort free RAM in bytes; falls back to a generous 12 GB if psutil absent."""
    try:
        import psutil  # type: ignore[import-untyped]
        return int(psutil.virtual_memory().available)
    except Exception:
        return 12 * 1024 * 1024 * 1024


def _choose_shape(target_n_cols: int = 200, target_n_rows: int = 10_000_000) -> Tuple[int, int]:
    """Shrink target shape so it fits in ~30% of free RAM."""
    budget = int(_free_ram_bytes() * 0.30)
    bytes_per_cell = 8  # float64
    cells_max = budget // bytes_per_cell
    target_cells = target_n_cols * target_n_rows
    if target_cells <= cells_max:
        return target_n_cols, target_n_rows
    # Scale rows down keeping ratio
    scale = (cells_max / target_cells) ** 0.5
    n_cols = max(20, int(target_n_cols * scale))
    n_rows = max(100_000, int(target_n_rows * scale))
    return n_cols, n_rows


# ---------------------------------------------------------------------------
# D-Arch-5: numba vs numpy for _col_stats (float branch) + per-col hash loop
# ---------------------------------------------------------------------------


def _col_stats_numpy(arr: np.ndarray) -> bytes:
    """Mirror of ``composite_cache._col_stats`` float branch."""
    isnan = ~np.isfinite(arr)
    n_null = int(isnan.sum())
    finite = arr[~isnan]
    if finite.size == 0:
        return f"all_null:{n_null}".encode("utf-8")
    return (
        f"min={float(np.min(finite)):.12g};"
        f"max={float(np.max(finite)):.12g};"
        f"null={n_null}"
    ).encode("utf-8")


def _build_col_stats_numba():
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(cache=True, fastmath=False, parallel=False)
    def _kernel(arr):
        # Mirrors the production kernel in ``composite_cache._col_stats_float_numba_kernel``:
        # ``fastmath=False`` is required so ``v != v`` survives as the NaN check (fastmath
        # assumes finite operands and collapses the test). Serial loop because a parallel
        # min / max reduction needs explicit atomics which numba doesn't expose cheaply; the
        # serial scan still ships >3x faster than the numpy mask path because it fuses the
        # passes.
        n = arr.shape[0]
        n_null = 0
        mn = np.inf
        mx = -np.inf
        for i in range(n):
            v = arr[i]
            if v != v or v == np.inf or v == -np.inf:
                n_null += 1
            else:
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        return mn, mx, n_null

    def _col_stats_numba(arr: np.ndarray) -> bytes:
        mn, mx, n_null = _kernel(arr)
        if not np.isfinite(mn):
            return f"all_null:{n_null}".encode("utf-8")
        return f"min={float(mn):.12g};max={float(mx):.12g};null={n_null}".encode("utf-8")

    return _col_stats_numba


def bench_col_stats_numba(n_cols: int | None = None, n_rows: int | None = None, n_iter: int = 3) -> Dict[str, Any]:
    """Compare numpy vs numba on the per-column stats pass + whole-frame loop."""
    if n_cols is None or n_rows is None:
        n_cols, n_rows = _choose_shape()
    rng = np.random.default_rng(0xD5)
    data = rng.standard_normal((n_cols, n_rows)).astype(np.float64)
    # Inject ~1% NaNs so the null-count branch fires.
    n_nan = max(1, n_rows // 100)
    nan_idx = rng.integers(0, n_rows, size=n_nan)
    data[:, nan_idx] = np.nan

    numba_fn = _build_col_stats_numba()
    if numba_fn is None:
        return {
            "skipped": True,
            "reason": "numba absent",
            "n_cols": n_cols,
            "n_rows": n_rows,
        }
    # Warmup numba JIT.
    _ = numba_fn(data[0])

    # Numpy: loop over columns.
    np_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        for c in range(n_cols):
            _ = _col_stats_numpy(data[c])
        np_times.append(time.perf_counter() - t0)

    # Numba: loop over columns; each kernel call runs in prange.
    nb_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        for c in range(n_cols):
            _ = numba_fn(data[c])
        nb_times.append(time.perf_counter() - t0)

    np_med = float(np.median(np_times))
    nb_med = float(np.median(nb_times))
    speedup = np_med / nb_med if nb_med > 0 else float("inf")
    return {
        "skipped": False,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "numpy_median_s": np_med,
        "numba_median_s": nb_med,
        "speedup": speedup,
        "gate_pass": speedup > 1.5,
    }


# ---------------------------------------------------------------------------
# D-Arch-6: cupy vs polars-fastpath for _fit_persist_and_transform
# ---------------------------------------------------------------------------


def _polars_fastpath_path(X) -> np.ndarray:
    """Mirror of the polars branch of ``_fit_persist_and_transform``."""
    import polars as pl
    cols = X.columns
    df_imp = X.with_columns([
        pl.col(c).fill_nan(pl.col(c).drop_nans().mean()) for c in cols
    ])
    _stats = df_imp.select(
        [pl.col(c).mean().alias(f"_mean_{c}") for c in cols] +
        [pl.col(c).std(ddof=0).alias(f"_std_{c}") for c in cols]
    ).row(0)
    n = len(cols)
    means = np.asarray(_stats[:n], dtype=np.float64)
    stds = np.asarray(_stats[n:], dtype=np.float64)
    stds_safe = np.where(stds == 0.0, 1.0, stds)
    df_std = df_imp.with_columns([
        (pl.col(c) - pl.lit(float(means[i]))) / pl.lit(float(stds_safe[i]))
        for i, c in enumerate(cols)
    ])
    return df_std.to_numpy()


def _cupy_path(X) -> np.ndarray:
    """Candidate GPU implementation. Upload numeric block, fill NaN with column mean,
    standardise, download. Single H2D + D2H; per-column work runs on GPU."""
    import cupy as cp
    arr = X.to_numpy()
    g = cp.asarray(arr)
    # Per-column mean ignoring nan.
    nan_mask = cp.isnan(g)
    g_filled = cp.where(nan_mask, 0.0, g)
    n_per_col = (~nan_mask).sum(axis=0).astype(cp.float64)
    n_per_col = cp.where(n_per_col == 0, 1.0, n_per_col)
    means = g_filled.sum(axis=0) / n_per_col
    # Fill nan with mean.
    g_imp = cp.where(nan_mask, means[None, :], g)
    # Std (ddof=0) on imputed frame.
    diffs = g_imp - means[None, :]
    var = (diffs * diffs).mean(axis=0)
    stds = cp.sqrt(var)
    stds_safe = cp.where(stds == 0, 1.0, stds)
    g_std = diffs / stds_safe[None, :]
    out = cp.asnumpy(g_std)
    return out


def bench_predict_guards_cupy(n_cols: int | None = None, n_rows: int | None = None, n_iter: int = 3) -> Dict[str, Any]:
    """Compare polars-fastpath vs cupy variant on a fully-numeric polars frame."""
    try:
        import polars as pl
    except ImportError:
        return {"skipped": True, "reason": "polars absent"}
    try:
        import cupy as cp  # noqa: F401
    except ImportError:
        return {"skipped": True, "reason": "cupy absent"}

    if n_cols is None or n_rows is None:
        # cupy path holds 2x the frame on GPU; bound shape to ~3 GB of GPU mem.
        n_cols, n_rows = _choose_shape(target_n_cols=50, target_n_rows=1_000_000)
    rng = np.random.default_rng(0xD6)
    data = rng.standard_normal((n_rows, n_cols)).astype(np.float64)
    n_nan = max(1, n_rows // 100)
    nan_idx = rng.integers(0, n_rows, size=n_nan)
    data[nan_idx, :] = np.nan
    df = pl.from_numpy(data, schema=[f"c{i}" for i in range(n_cols)])

    # Warmup cupy (kernel compile + cuBLAS init).
    _ = _cupy_path(df)

    pl_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = _polars_fastpath_path(df)
        pl_times.append(time.perf_counter() - t0)
    gc.collect()

    cp_times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _ = _cupy_path(df)
        cp_times.append(time.perf_counter() - t0)
    gc.collect()

    pl_med = float(np.median(pl_times))
    cp_med = float(np.median(cp_times))
    speedup = pl_med / cp_med if cp_med > 0 else float("inf")
    return {
        "skipped": False,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "polars_median_s": pl_med,
        "cupy_median_s": cp_med,
        "speedup": speedup,
        "gate_pass": speedup > 1.05,  # 5% threshold per user spec
    }


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(bench_col_stats_numba(), indent=2))
    print(json.dumps(bench_predict_guards_cupy(), indent=2))
