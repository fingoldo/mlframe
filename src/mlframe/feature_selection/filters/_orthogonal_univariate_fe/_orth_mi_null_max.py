"""Per-permutation maximum MI of a candidate matrix against permuted targets, binning the matrix once.

The family-wise null of the local MI gate scores the same candidate matrix against ``B`` permutations of the target. The CPU plug-in MI bins each column by rank, and the rank bins do not
depend on the target, so ``B`` full calls repeat the argsort of every column ``B`` times; the argsort is ~97% of a column's MI time. The columns are rank-binned once here and each
permutation only builds the class histograms, with the same expressions in the same order as ``_plugin_mi_classif_njit``, so the values are bit-identical to the repeated calls.

The shortcut applies only where ``_mi_classif_batch`` would take the CPU numba rank path (no GPU residency, no edge binning, no imbalance weighting, every column finite); any other
configuration runs the repeated calls unchanged.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable

import numpy as np
from numba import njit, prange

from .._numba_utils import _quantile_bin_njit

logger = logging.getLogger(__name__)


@njit(cache=True, parallel=True)
def _rank_codes_rows_njit(X: np.ndarray, n_bins: int) -> np.ndarray:
    """(k, n) int32 rank-bin codes of the columns of the (n, k) matrix ``X`` (each column copied contiguous before the argsort, as the batch MI kernel does)."""
    n, k = X.shape
    codes = np.empty((k, n), dtype=np.int32)
    for j in prange(k):
        codes[j, :] = _quantile_bin_njit(X[:, j].copy(), n_bins)
    return codes


@njit(cache=True, parallel=True)
def _rank_codes_cols_njit(X: np.ndarray, cols: np.ndarray, n_bins: int) -> np.ndarray:
    """:func:`_rank_codes_rows_njit` for the columns ``cols`` of ``X`` only: (len(cols), n) int32 rank-bin codes."""
    n = X.shape[0]
    codes = np.empty((cols.shape[0], n), dtype=np.int32)
    for t in prange(cols.shape[0]):
        codes[t, :] = _quantile_bin_njit(X[:, cols[t]].copy(), n_bins)
    return codes


@njit(cache=True, fastmath=True, parallel=True)
def _mi_from_codes_rows_njit(codes: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """Plug-in MI of each row of the (k, n) ``codes`` against ``y``; the arithmetic of ``_plugin_mi_classif_njit`` after its binning step."""
    k, n = codes.shape
    out = np.zeros(k, dtype=np.float64)  # Intentional float64: MI statistic vector, output contract kept f64
    if n == 0:
        return out
    y_min = y[0]
    y_max = y[0]
    for i in range(1, n):
        if y[i] < y_min:
            y_min = y[i]
        if y[i] > y_max:
            y_max = y[i]
    n_classes = (y_max - y_min) + 1
    # the class counts and the shifted labels are the same for every column: counted once, and the bin counts are the row sums of the joint table
    yc = np.empty(n, dtype=np.int64)
    hist_y = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        yc[i] = y[i] - y_min
        hist_y[yc[i]] += 1
    for j in prange(k):
        hist_xy = np.zeros((n_bins, n_classes), dtype=np.int64)
        for i in range(n):
            hist_xy[codes[j, i], yc[i]] += 1
        hist_x = np.zeros(n_bins, dtype=np.int64)
        for b in range(n_bins):
            for c in range(n_classes):
                hist_x[b] += hist_xy[b, c]
        log_n = math.log(n)
        mi = 0.0
        for b in range(n_bins):
            if hist_x[b] == 0:
                continue
            log_hx = math.log(hist_x[b])
            for c in range(n_classes):
                n_xy = hist_xy[b, c]
                if n_xy == 0 or hist_y[c] == 0:
                    continue
                mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[c]))
        if mi < 0.0:
            mi = 0.0
        out[j] = mi
    return out


def _imbalance_weighting_on() -> bool:
    """Whether the opt-in class-balanced MI would replace the plain MI (``_mi_classif_batch`` takes it before any backend)."""
    return os.environ.get("MLFRAME_FE_IMBALANCE_MI", "").strip().lower() in ("on", "1", "true", "yes", "auto")


def _cpu_rank_path_applies(X: np.ndarray, nbins: int) -> bool:
    """Whether ``_mi_classif_batch(X, y, nbins=nbins)`` would score the finite columns of ``X`` with the CPU numba rank batch kernel (no GPU route, no edge binning, resident path); the finiteness of
    the values is not part of this test, a column with a missing value is scored on its own."""
    from . import _orth_mi_backends as be

    if X.ndim != 2 or X.shape[1] == 0 or be._MI_BACKEND != "numba":
        return False
    if os.environ.get("MLFRAME_MI_BACKEND", "") == "cuda":
        return False
    n, p = int(X.shape[0]), int(X.shape[1])
    from .._fe_gpu_strict import fe_gpu_strict_enabled

    if fe_gpu_strict_enabled(n=n, p=p) or be._orth_mi_gpu_enabled(n=n, p=p) or be._fe_edge_binning_enabled():
        return False
    from .._fe_usability_signal import _crit_np_dtype

    return be._mi_path_for(n, p, _crit_np_dtype(), False) in ("resident", "cpu_blocks")


def _takes_cpu_rank_path(X: np.ndarray, nbins: int) -> bool:
    """Whether ``_mi_classif_batch(X, y, nbins=nbins)`` would run the CPU numba rank batch kernel on every column of ``X`` (the rank path applies and every value is finite)."""
    from . import _orth_mi_backends as be
    from .._fe_usability_signal import _crit_np_dtype

    return _cpu_rank_path_applies(X, nbins) and be._mi_path_for(int(X.shape[0]), int(X.shape[1]), _crit_np_dtype(), False) == "resident" and bool(np.isfinite(X).all())


def _null_max_strict_gpu(X: np.ndarray, y: np.ndarray, permutations: "list[np.ndarray]", nbins: int) -> "np.ndarray | None":
    """The STRICT-residency twin of ``_mi_classif_batch``: the candidate matrix is uploaded once and its bin edges are computed once (``candidate_cache``), every permutation then only
    re-bins nothing and counts. ``None`` when STRICT is off for this shape, cupy is absent or a device fault occurs, so the caller runs the repeated per-permutation calls."""
    from . import _orth_mi_backends as be
    from .._fe_gpu_strict import fe_gpu_strict_enabled

    if X.ndim != 2 or X.shape[1] == 0 or not fe_gpu_strict_enabled(n=int(X.shape[0]), p=int(X.shape[1])):
        return None
    try:
        import cupy as cp

        from .._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

        # Intentional float64: GPU f64 MI path, the same upload the per-call STRICT route makes.
        Xd = cp.asarray(np.ascontiguousarray(np.asarray(X, dtype=np.float64)))
        cache: dict = {}
        out = np.empty(len(permutations), dtype=np.float64)  # Intentional float64: statistic vector (null max), exact permutation ties
        y_arr = np.asarray(y)
        for b, perm in enumerate(permutations):
            yd, y_min, n_classes = be._resident_labels(y_arr[perm], cp)
            mi = _plugin_mi_classif_batch_cuda_resident(Xd, yd, int(nbins), y_min=y_min, n_classes=n_classes, candidate_cache=cache)
            out[b] = float(np.nanmax(np.asarray(mi, dtype=np.float64)))  # Intentional float64: statistic vector (MI), exact permutation ties
        return out
    except (ImportError, *be._device_error_classes()) as exc:
        logger.warning("_null_max_strict_gpu: GPU null-max unavailable, falling back to the CPU path: %s", exc)
        return None


def _null_max_streamed_path(X: np.ndarray, y: np.ndarray, permutations: "list[np.ndarray]", nbins: int) -> "np.ndarray | None":
    """The streamed-GPU twin: when ``_mi_classif_batch`` would score ``X`` through the streamed path (the matrix is too large for one resident batch), every permutation reuses each block's
    upload and bin edges. ``None`` when that route does not apply or fails, so the caller runs the repeated per-permutation calls."""
    from . import _orth_mi_backends as be

    if X.ndim != 2 or X.shape[1] == 0 or be._MI_BACKEND != "numba" or not bool(np.isfinite(X).all()):
        return None
    from .._fe_usability_signal import _crit_np_dtype

    dtype = _crit_np_dtype()
    n, p = int(X.shape[0]), int(X.shape[1])
    try:
        if be._mi_path_for(n, p, dtype, False) != "streamed":
            return None
        from ._streamed_gpu_mi import _devices, _host_free_bytes, null_max_streamed, plan_streamed_mi

        plan = plan_streamed_mi(n, _devices(), _host_free_bytes(), dtype)
        if plan["width"] <= 0:
            return None
        width = int(plan["width"])
        blocks = (X[:, a : a + width] for a in range(0, p, width))
        return null_max_streamed(blocks, y, permutations, nbins=int(nbins), n_rows=n, dtype=dtype, n_cols=p)
    except Exception:
        logger.warning("streamed GPU null maxima unavailable for this matrix; scoring each permutation separately.", exc_info=True)
        return None


def _null_max_rank_mixed(X: np.ndarray, y: np.ndarray, permutations: "list[np.ndarray]", nbins: int, fallback: Callable[..., Any]) -> np.ndarray:
    """The rank-code shortcut for a matrix that ``_mi_classif_batch`` scores in column blocks and/or whose columns may hold a missing value.

    ``_mi_classif_batch`` scores the finite columns of each block in one batch and every column with a missing value on its own, on the finite rows of that column. Here each block's finite columns are
    rank-binned once and each permutation only counts; the columns with a missing value are scored through ``fallback`` (the genuine batch MI) on their own, so each value is the one the repeated calls give.
    The maximum over a permutation is the maximum over the blocks and both groups. Only one block's codes are held at a time, the memory bound that routed the matrix to column blocks."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry
    from . import _orth_mi_backends as be
    from .._fe_usability_signal import _crit_np_dtype

    n, p = int(X.shape[0]), int(X.shape[1])
    resident = be._mi_path_for(n, p, _crit_np_dtype(), False) == "resident"
    width = p if resident else max(1, int(be._mi_chunk_cols_for(n)))
    y_i64 = np.ascontiguousarray(y, dtype=np.int64)
    y_arr = np.asarray(y)
    out = np.full(len(permutations), np.nan, dtype=np.float64)  # Intentional float64: statistic vector (null max), exact permutation ties
    with parallel_kernel_entry():
        for a in range(0, p, width):
            block = X[:, a : a + width]
            finite = np.isfinite(block).all(axis=0)
            dense = np.flatnonzero(finite).astype(np.int64)
            partial = np.flatnonzero(~finite)
            codes = _rank_codes_cols_njit(block, dense, int(nbins)) if dense.size else None
            block_partial = np.ascontiguousarray(block[:, partial]) if partial.size else None
            for b, perm in enumerate(permutations):
                if codes is not None:
                    out[b] = np.fmax(out[b], np.nanmax(_mi_from_codes_rows_njit(codes, np.ascontiguousarray(y_i64[perm]), int(nbins))))
                if block_partial is not None:
                    part = np.asarray(fallback(block_partial, y_arr[perm], nbins=nbins), dtype=np.float64)  # Intentional float64: statistic vector (MI), exact permutation ties
                    out[b] = np.fmax(out[b], np.nanmax(part))
            del codes
    return out


def null_max_mi(X: np.ndarray, y: np.ndarray, permutations: "list[np.ndarray]", nbins: int, fallback: Callable[..., Any]) -> np.ndarray:
    """``max_j MI(X_j; y[perm])`` (NaN-ignoring) for every permutation, in double precision, with the matrix binned once.

    Only the genuine ``_mi_classif_batch`` is shortcut. Runs through the STRICT-residency kernel (one upload, one edge computation) when STRICT is on, through the streamed GPU kernel when the matrix is routed there, through the rank-code kernel when ``_mi_classif_batch`` would take the CPU numba rank
    path, and otherwise through ``fallback`` (``_mi_classif_batch``) once per permutation."""
    from . import _orth_mi_backends as be

    out = np.empty(len(permutations), dtype=np.float64)  # Intentional float64: statistic vector (null max), exact permutation ties
    # The shortcuts reproduce the genuine batch MI; a substituted scorer (a test double, a caller's own estimator) is honoured by scoring through it once per permutation.
    if fallback is be._mi_classif_batch and not _imbalance_weighting_on():
        strict = _null_max_strict_gpu(X, y, permutations, nbins)
        if strict is not None:
            return strict
        streamed = _null_max_streamed_path(X, y, permutations, nbins)
        if streamed is not None:
            return streamed
        if _takes_cpu_rank_path(X, nbins):
            from mlframe._numba_parallel_guard import parallel_kernel_entry

            y_i64 = np.ascontiguousarray(y, dtype=np.int64)
            with parallel_kernel_entry():
                codes = _rank_codes_rows_njit(np.ascontiguousarray(X), int(nbins))
                for b, perm in enumerate(permutations):
                    out[b] = float(np.nanmax(_mi_from_codes_rows_njit(codes, np.ascontiguousarray(y_i64[perm]), int(nbins))))
            return out
        if _cpu_rank_path_applies(X, nbins):
            return _null_max_rank_mixed(X, y, permutations, nbins, fallback)
    for b, perm in enumerate(permutations):
        # Intentional float64: statistic vector (null max), exact permutation ties; the rank shortcut above is float64 too.
        out[b] = float(np.nanmax(np.asarray(fallback(X, np.asarray(y)[perm], nbins=nbins), dtype=np.float64)))
    return out
