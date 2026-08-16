"""CUDA (cupy) discretization kernels carved out of ``discretization/__init__.py`` to keep that
module under the project's ~1000 LOC guideline.

Re-exported from ``__init__.py`` (a package ``__init__.py`` is exempt from ruff's F401 unused-
import rule, so a plain import there is sufficient -- no self-alias needed) so every existing
import path (``from mlframe.feature_selection.filters.discretization import discretize_2d_array_cuda``,
and ``discretize_2d_array``'s own CUDA-fastpath call) keeps resolving unchanged.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Optional

import numpy as np

from . import _safe_code_dtype

logger = logging.getLogger(__name__)


def discretize_2d_array_cuda(
    arr: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
    dtype: type = np.int8,
) -> np.ndarray:
    """CuPy port of :func:`discretize_2d_array` for the quantile method.

    Single-launch ``cp.percentile`` computes all per-column edges at once;
    per-column ``cp.searchsorted`` produces the ordinal bins. Total H2D +
    compute + D2H on a 1M-row x 30-col frame runs in ~50 ms (vs ~880 ms
    for the CPU prange path on the same workload at fit-time on a
    GTX 1050 Ti / cc 6.1).

    Returns:
        ``np.ndarray`` of shape ``arr.shape`` with the requested ``dtype``.
        ``copy_to_host`` happens at the end - callers see plain numpy.

    Raises:
        RuntimeError: if CuPy is not installed or CUDA is not available.
        NotImplementedError: for ``method`` other than ``"quantile"``.

    The function does NOT replace :func:`discretize_2d_array`; both stay
    available. A future dispatch path (``discretize_2d_array_dispatch``)
    can route by ``(n_rows, n_cols)`` and CUDA availability, mirroring
    the ``dispatch_batch_pair_mi`` pattern in ``batch_pair_mi_gpu``.
    """
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("cupy not installed; discretize_2d_array_cuda unavailable") from exc

    try:
        from .._gpu_policy import cuda_available_for_run
        if not cuda_available_for_run():
            raise RuntimeError("CUDA not available on this host")
    except ImportError:
        pass  # fall through; cupy import succeeded so CUDA is likely there

    if method not in ("quantile", "uniform"):
        raise NotImplementedError(
            f"discretize_2d_array_cuda implements 'quantile' / 'uniform'; got method={method!r}",
        )

    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array; got shape {arr.shape}")

    n_rows, n_cols = arr.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty(arr.shape, dtype=dtype)

    # Widen the code dtype to hold ordinal codes 0..n_bins-1 BEFORE allocating the device output - mirrors
    # the CPU discretize_2d_array (_safe_code_dtype). Without this an int8 request at n_bins>128 wrapped the
    # top bins negative on the GPU (codes 128..n_bins-1 -> negative) while the CPU path widened to int16,
    # a silent cross-backend divergence on the public API. (Verified: NaN routing already matches CPU.)
    dtype = _safe_code_dtype(n_bins, dtype, reserve_nan_slot=(method == "uniform"))
    d_arr = cp.asarray(arr)  # H2D once for the whole frame
    # No throwaway GPU allocation just to read a dtype object - ``np.dtype(dtype)`` produces the identical
    # ``numpy.dtype`` instance cupy's own ``.dtype`` attribute would (cupy dtypes ARE numpy dtypes), so the
    # 1-element ``cp.asarray``/upload this used to pay per call is pure overhead with no different result.
    _out_cp_dtype = cp.int8 if dtype == np.int8 else np.dtype(dtype)
    out = cp.empty((n_rows, n_cols), dtype=_out_cp_dtype)

    if method == "quantile":
        qs = cp.linspace(0.0, 100.0, n_bins + 1)
        # cp.percentile has no nanpercentile twin (unlike numpy), and a plain
        # cp.percentile over a NaN-bearing column poisons EVERY edge for that column with NaN - searchsorted
        # against an all-NaN edges row then silently collapses the WHOLE column's real values (not just the
        # NaN rows) to a single bin, the exact bug the CPU path's edges()/get_binning_edges() were already
        # fixed for. cupy has no NaN-aware percentile kernel to vectorise this with, so route the
        # rare NaN-bearing case through numpy's nanpercentile on the host array already available in ``arr``
        # (this function's caller has it; ``d_arr`` is just its device upload) - the common NaN-free case
        # keeps the fully vectorised cp.percentile fast path unchanged.
        if bool(cp.isnan(d_arr).any()):
            bin_edges = cp.asarray(np.nanpercentile(arr, cp.asnumpy(qs), axis=0))
        else:
            # cp.percentile vectorises across axis=0 -> bin_edges shape: (n_bins + 1, n_cols).
            bin_edges = cp.percentile(d_arr, qs, axis=0)
        # cp.searchsorted is 1-D; loop per column. Each call is fully on-device
        # so the loop is dispatch-overhead only (~30 us per launch). For
        # n_cols=30 the total dispatch is ~1 ms vs ~50 ms compute. For
        # n_cols >= 1000 the Python-loop dispatch becomes a wall: route to the
        # fused RawKernel ``discretize_quantile_cuda_rk`` below in that regime.
        if n_cols >= 1000:
            # Per-row col-wise: ravel bin_edges to (n_cols * (n_bins+1)) and do
            # one fused 2D searchsorted via a hand-rolled RawKernel. ~10x
            # speedup vs the per-col Python loop on n_cols=10k.
            cuts = cp.ascontiguousarray(bin_edges[1:-1, :].T)  # (n_cols, n_bins-1)
            out = _discretize_quantile_rawkernel(d_arr, cuts, n_bins, _out_cp_dtype)
        else:
            for j in range(n_cols):
                out[:, j] = cp.searchsorted(bin_edges[1:-1, j], d_arr[:, j], side="right")
    else:
        # method == 'uniform': vectorised arithmetic, no percentile sort,
        # no per-column dispatch. Single GPU pass. Mirrors discretize_uniform
        # njit kernel on CPU. Fastest path for Gaussian-ish data where the
        # accuracy hit vs quantile is small (bench at info_theory module
        # docstring quotes H(X)/log(nbins) >= 0.82 for Gaussian).
        # plain cp.min/cp.max propagate NaN (a single NaN anywhere in a column
        # poisons that column's min/max to NaN), unlike the CPU discretize_uniform path's NaN-aware range.
        # cp.nanmin/cp.nanmax exist (unlike cp.nanpercentile above) so this branch stays fully on-device.
        col_min = cp.nanmin(d_arr, axis=0, keepdims=True)
        col_max = cp.nanmax(d_arr, axis=0, keepdims=True)
        # Mirrors the CPU
        # ``discretize_uniform`` fix - canonical formula
        # ``rev_bin_width = n_bins / (max - min)`` with constant-column
        # zero fallback. The pre-fix formula
        # ``n_bins / (max - min + min/2)`` silently mis-binned positive-
        # shifted columns (e.g. linspace(1000, 1100) collapsed to 2 bins
        # instead of 10) AND broke on negative ranges via div-by-zero
        # / sign flip. Cross-backend bit-comparability still holds
        # because both backends now use the same canonical formula.
        _rng = col_max - col_min
        # Where range is zero (constant column), substitute 1 to avoid
        # div-by-zero; the resulting code is clamped to 0 below so the
        # column emits a single bin honestly.
        _rng_safe = cp.where(_rng > 0, _rng, 1.0)
        rev = n_bins / _rng_safe
        out_f = (d_arr - col_min) * rev
        out_f = cp.where(_rng > 0, out_f, 0.0)
        out_f = cp.clip(out_f, 0, n_bins - 1)
        # an individual NaN VALUE (not just a NaN-poisoned column min/max,
        # already fixed above) still produces NaN through the affine map; cp.clip is a no-op on NaN (like
        # numpy), so without this it would cast to an undefined/garbage int code. The CPU discretize_uniform
        # kernel routes NaN rows to a dedicated code one past the real range (``nan_code = n_bins``) instead
        # of colliding with a real bin - mirror that here so NaN rows carry the same, correct, honest code
        # on both backends rather than a silent garbage cast.
        out_f = cp.where(cp.isnan(d_arr), float(n_bins), out_f)
        out = out_f.astype(_out_cp_dtype)

    # D2H the final tensor (single transfer, n_rows * n_cols bytes for int8).
    return np.asarray(cp.asnumpy(out).astype(dtype, copy=False))


def _choose_discretize_row_chunk_rows(n_cols: int, in_itemsize: int, free_bytes: int, out_itemsize: int = 2) -> int:
    """Rows of ``arr`` (``n_cols`` columns, ``in_itemsize`` bytes/element) that fit a single row-chunk
    upload within a safe VRAM budget (40% of free VRAM, leaving headroom for the output chunk + any
    quantile-edge/reduction scratch). Clamped to >=10_000 rows (a tiny chunk would need an excessive
    number of launches) and to 20M as a sane ceiling.

    ``out_itemsize`` is the widened output code dtype's byte width (``_safe_code_dtype``): a high ``n_bins``
    widens codes to int32/int64, so the fixed ``+2`` margin understated the per-row output cost and could
    pick a chunk ~2x too large. Default 2 preserves the historical margin for callers that don't pass it."""
    budget = max(0, int(free_bytes * 0.4))
    per_row_bytes = max(1, n_cols * (in_itemsize + out_itemsize))  # input row + one output row of the real code width
    rows = budget // per_row_bytes
    return int(np.clip(rows, 10_000, 20_000_000))


def discretize_2d_array_cuda_row_chunked(
    arr: np.ndarray,
    n_bins: int = 10,
    method: str = "quantile",
    dtype: type = np.int8,
    quantile_subsample_rows: Optional[int] = None,
    free_bytes: Optional[int] = None,
) -> np.ndarray:
    """Row-chunked variant of :func:`discretize_2d_array_cuda` for when the FULL ``arr`` upload would not
    safely fit in free VRAM. Uploads ``arr`` in row-chunks small enough to fit; the two methods handle the
    cross-chunk statistic differently:

    * ``method="uniform"``: EXACT, no approximation. Column min/max are genuinely reducible across row-
      chunks (running min/max, pass 1), then the elementwise bin formula is applied per row-chunk (pass 2)
      using the exact global min/max - bit-identical to :func:`discretize_2d_array_cuda`.
    * ``method="quantile"``: APPROXIMATE by construction. Exact quantiles need the full column's order
      statistics, which is NOT reducible across row-chunks without a streaming quantile algorithm. Instead,
      bin edges are computed from a GPU-resident random SUBSAMPLE (``quantile_subsample_rows``, default
      ``None`` -> ``feature_engineering.UNIFIED_FE_SUBSAMPLE_N`` = 30_000, the SAME validated MI-sweep
      subsample size used throughout MRMR's FE pipeline - jaccard=1.0 vs full-n at 50k+, 0.88 at 5k, per
      the bench backing that constant. Quantile-edge estimation has far lower sampling variance than the
      MI estimation that constant was validated for, so 30k is comfortably sufficient here too) then
      applied via row-chunked ``searchsorted`` (exact application of approximate edges). This matches the
      project's documented FE/MRMR exception (a binning/candidate-MI speed lever's bar is SELECTION-
      equivalence, not bit-identical MI). See
      ``tests/feature_selection/discretization/test_discretize_2d_array_row_chunked.py`` for the
      closeness/selection-equivalence validation.

    Returns a plain ``np.ndarray`` (D2H happens per row-chunk, not as one giant transfer at the end).

    ``free_bytes``: an optional already-probed free-VRAM byte count (``cp.cuda.runtime.memGetInfo()``'s
    first element). When the caller (``discretize_2d_array``'s CUDA-eligibility gate) already probed
    free VRAM microseconds earlier for its own reject decision, passing it here skips this function's
    own redundant ``memGetInfo`` call - ``memGetInfo`` is a read-only device counter query with no
    intervening GPU allocation between the two probes, so reusing the value changes no decision.
    ``None`` (the default - direct/standalone calls) keeps the self-probe unchanged.
    """
    import cupy as cp

    if quantile_subsample_rows is None:
        from mlframe.feature_selection.filters.feature_engineering import UNIFIED_FE_SUBSAMPLE_N

        quantile_subsample_rows = UNIFIED_FE_SUBSAMPLE_N

    if method not in ("quantile", "uniform"):
        raise NotImplementedError(f"discretize_2d_array_cuda_row_chunked implements 'quantile' / 'uniform'; got method={method!r}")
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array; got shape {arr.shape}")

    n_rows, n_cols = arr.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty(arr.shape, dtype=dtype)

    dtype = _safe_code_dtype(n_bins, dtype, reserve_nan_slot=(method == "uniform"))
    # No throwaway GPU allocation just to read a dtype object - ``np.dtype(dtype)`` produces the identical
    # ``numpy.dtype`` instance cupy's own ``.dtype`` attribute would (cupy dtypes ARE numpy dtypes), so the
    # 1-element ``cp.asarray``/upload this used to pay per call is pure overhead with no different result.
    _out_cp_dtype = cp.int8 if dtype == np.int8 else np.dtype(dtype)

    if free_bytes is not None:
        free_b = int(free_bytes)
    else:
        try:
            free_b, _total_b = cp.cuda.runtime.memGetInfo()
        except Exception as e:
            logger.debug("cp.cuda.runtime.memGetInfo() failed, using 512MiB conservative fallback: %s", e)
            free_b = 512 * 1024 * 1024  # conservative fallback if the probe is unavailable
    row_chunk_rows = _choose_discretize_row_chunk_rows(n_cols, arr.dtype.itemsize, free_b, out_itemsize=np.dtype(_out_cp_dtype).itemsize)
    _quantile_subsample_note = f", quantile_subsample_rows={min(n_rows, quantile_subsample_rows)}/{n_rows}" if method == "quantile" else ""
    logger.info(
        "discretize_2d_array_cuda_row_chunked: method=%s n_rows=%d n_cols=%d in_dtype=%s -> row_chunk_rows=%d "
        "(%d chunk(s)), free_vram=%.2fGB%s",
        method, n_rows, n_cols, arr.dtype, row_chunk_rows, -(-n_rows // row_chunk_rows), free_b / 1024**3,
        _quantile_subsample_note,
    )

    out: np.ndarray = np.empty((n_rows, n_cols), dtype=dtype)
    n_chunks = 0

    if method == "uniform":
        col_min_d: Any = None
        col_max_d: Any = None
        # Mirrors the B-12 fix already landed on the
        # non-chunked sibling discretize_2d_array_cuda - plain cp.min/cp.max propagate NaN (a single NaN
        # anywhere in a column poisons that column's min/max to NaN), so use cp.nanmin/cp.nanmax instead.
        for row_start in range(0, n_rows, row_chunk_rows):
            row_end = min(row_start + row_chunk_rows, n_rows)
            d_chunk = cp.asarray(arr[row_start:row_end])
            cmin = cp.nanmin(d_chunk, axis=0)
            cmax = cp.nanmax(d_chunk, axis=0)
            col_min_d = cmin if col_min_d is None else cp.minimum(col_min_d, cmin)
            col_max_d = cmax if col_max_d is None else cp.maximum(col_max_d, cmax)
            del d_chunk
        _rng = col_max_d - col_min_d
        _rng_safe = cp.where(_rng > 0, _rng, 1.0)
        rev = n_bins / _rng_safe
        for row_start in range(0, n_rows, row_chunk_rows):
            row_end = min(row_start + row_chunk_rows, n_rows)
            d_chunk = cp.asarray(arr[row_start:row_end])
            out_f = (d_chunk - col_min_d) * rev
            out_f = cp.where(_rng > 0, out_f, 0.0)
            out_f = cp.clip(out_f, 0, n_bins - 1)
            # Route individual NaN VALUES to the dedicated
            # NaN bin code (n_bins), matching the CPU discretize_uniform kernel and the fixed non-chunked
            # sibling - without this, cp.clip is a no-op on NaN and it would cast to a garbage int code.
            out_f = cp.where(cp.isnan(d_chunk), float(n_bins), out_f)
            out[row_start:row_end] = cp.asnumpy(out_f.astype(_out_cp_dtype))
            del d_chunk, out_f
            n_chunks += 1
    else:  # quantile
        sub_n = min(n_rows, quantile_subsample_rows)
        if sub_n < n_rows:
            sub_idx = np.sort(np.random.default_rng(0).choice(n_rows, size=sub_n, replace=False))
            sub_arr = arr[sub_idx]
        else:
            sub_arr = arr
        d_sub = cp.asarray(sub_arr)
        qs = cp.linspace(0.0, 100.0, n_bins + 1)
        # Mirrors the B-12 fix on the non-chunked sibling -
        # cp.percentile has no nanpercentile twin, and a plain cp.percentile over a NaN-bearing subsample
        # poisons EVERY edge for that column with NaN, collapsing the whole column to one degenerate bin.
        if bool(cp.isnan(d_sub).any()):
            bin_edges = cp.asarray(np.nanpercentile(sub_arr, cp.asnumpy(qs), axis=0))
        else:
            bin_edges = cp.percentile(d_sub, qs, axis=0)
        del d_sub
        # Cut points are derived from ``bin_edges`` ONCE here (fit-constant across every row-chunk below)
        # instead of being re-transposed inside ``_discretize_quantile_rawkernel`` on every chunk call -
        # that re-derivation was an O(n_cols * n_bins) transpose+copy repeated per chunk for identical output.
        cuts = cp.ascontiguousarray(bin_edges[1:-1, :].T) if n_cols >= 1000 else None  # (n_cols, n_bins-1)
        for row_start in range(0, n_rows, row_chunk_rows):
            row_end = min(row_start + row_chunk_rows, n_rows)
            d_chunk = cp.asarray(arr[row_start:row_end])
            if n_cols >= 1000:
                chunk_out = _discretize_quantile_rawkernel(d_chunk, cuts, n_bins, _out_cp_dtype)
            else:
                chunk_out = cp.empty((row_end - row_start, n_cols), dtype=_out_cp_dtype)
                for j in range(n_cols):
                    chunk_out[:, j] = cp.searchsorted(bin_edges[1:-1, j], d_chunk[:, j], side="right")
            out[row_start:row_end] = cp.asnumpy(chunk_out)
            del d_chunk, chunk_out
            n_chunks += 1

    logger.debug(
        "discretize_2d_array_cuda_row_chunked: method=%s n_rows=%d n_cols=%d row_chunk_rows=%d n_chunks=%d",
        method, n_rows, n_cols, row_chunk_rows, n_chunks,
    )
    return out


_SEARCHSORTED_RIGHT_2D_SRC = r"""
extern "C" __global__ void searchsorted_right_2d(
    const double* __restrict__ arr,    // (n_rows, n_cols) C-order
    const double* __restrict__ cuts,    // (n_cols, n_cuts) C-order
    int* __restrict__ out,              // (n_rows, n_cols)
    const int n_rows, const int n_cols, const int n_cuts
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rows * n_cols;
    if (gid >= total) return;
    int row = gid / n_cols;
    int col = gid % n_cols;
    double v = arr[row * n_cols + col];
    // searchsorted side='right': bin = first index i s.t. cuts[i] > v,
    // OR n_cuts if every cut <= v.
    int lo = 0, hi = n_cuts;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cuts[col * n_cuts + mid] > v) hi = mid;
        else lo = mid + 1;
    }
    out[row * n_cols + col] = lo;
}
"""
_searchsorted_right_2d_cuda = None
_DISCRETIZE_KERNEL_LOCK = threading.Lock()


def _get_searchsorted_right_2d_kernel():
    """Build (idempotently) and return the fused per-column searchsorted RawKernel.

    Mirrors ``info_theory._cmi_cuda._get_kernel``'s module-level-singleton pattern. ``cp.RawKernel`` used
    to be rebuilt from CUDA source text on EVERY call to ``_discretize_quantile_rawkernel`` - which
    ``discretize_2d_array_cuda_row_chunked``'s quantile branch calls once per row-chunk (up to 10-50+
    times per large discretize call) - so the source was recompiled that many times per fit instead of
    once for the whole process lifetime.
    """
    global _searchsorted_right_2d_cuda
    if _searchsorted_right_2d_cuda is not None:
        return _searchsorted_right_2d_cuda
    import cupy as cp

    with _DISCRETIZE_KERNEL_LOCK:
        if _searchsorted_right_2d_cuda is not None:
            return _searchsorted_right_2d_cuda
        module = sys.modules[__name__]
        module._searchsorted_right_2d_cuda = cp.RawKernel(  # type: ignore[attr-defined]
            _SEARCHSORTED_RIGHT_2D_SRC, "searchsorted_right_2d",
        )
        return module._searchsorted_right_2d_cuda


def _discretize_quantile_rawkernel(d_arr, cuts, n_bins, out_cp_dtype):
    """Fused per-column searchsorted via cupy RawKernel.

    Replaces the Python-loop calling ``cp.searchsorted`` once per column,
    which becomes dispatch-bound at n_cols >= 1000 (~30us launch * 1000 cols
    = 30ms wasted on dispatch alone). The fused kernel does ``n_rows*n_cols``
    binary searches in parallel; for n=1M / p=1000 / n_bins=10 measured ~7ms
    vs ~70ms for the per-col loop on cc 6.1.

    ``cuts`` is the caller's PRE-TRANSPOSED, contiguous ``(n_cols, n_bins-1)`` cut-point matrix (its
    ``bin_edges[1:-1, :].T``) - hoisted out of this function because ``bin_edges``/``cuts`` are fit-
    constant across every row-chunk of one ``discretize_2d_array_cuda_row_chunked`` call, so re-deriving
    them here on every call re-paid an O(n_cols * n_bins) transpose+copy per chunk for identical output.
    """
    import cupy as cp
    n_rows, n_cols = d_arr.shape
    out_int32 = cp.empty((n_rows, n_cols), dtype=cp.int32)
    kernel = _get_searchsorted_right_2d_kernel()
    threads = 256
    blocks = (n_rows * n_cols + threads - 1) // threads
    kernel((blocks,), (threads,), (
        d_arr.astype(cp.float64, copy=False), cuts.astype(cp.float64, copy=False),
        out_int32, np.int32(n_rows), np.int32(n_cols), np.int32(n_bins - 1),
    ))
    return out_int32.astype(out_cp_dtype, copy=False)
