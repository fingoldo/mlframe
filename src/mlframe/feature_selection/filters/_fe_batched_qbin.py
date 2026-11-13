"""Born-on-device equi-frequency quantile binning for the FE candidate pool.

Carved out of ``_fe_batched_mi`` for file size; ``_fe_batched_mi`` re-exports the public names.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._device_sort_safe import device_percentile

logger = logging.getLogger(__name__)


_QBIN_CODERS: dict = {}

# One coder source for both column types. {REAL} is the column type, {KEY} the signed integer type of its order-preserving key, {AS_INT} the bit reinterpretation and {SIGN} the mask that
# flips the magnitude bits of a negative: for double the key is 64-bit, for float it is 32-bit, so a float32 column is read at half the width and compared on the same integer pipeline.
_QBIN_CODER_TEMPLATE = r"""
__device__ __forceinline__ {KEY} mono_key({REAL} v) {
    v = v + 0.0;                                 // normalize -0.0 -> +0.0
    {KEY} b = {AS_INT}(v);
    return b >= 0 ? b : (b ^ {SIGN});            // signed-compare monotonic map
}
extern "C" __global__ void qbin_code(const {REAL}* __restrict__ x, const {KEY}* __restrict__ edge_keys,
                                     const long long n, const long long K, const int ne1,
                                     long long* __restrict__ codes) {
    // edge_keys is (ne1, K) with non-interior slots pre-set to the key maximum host-side (compare always false)
    // and the binary-column top-edge duplicated into the extra last row - so the whole coder is one
    // branch-free strided int-compare loop.
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * K;
    if (t >= total) return;
    {KEY} vk = mono_key(x[t]);
    long long idx = t % K + K;                  // start at j=1: j=0 is never interior (== column min)
    long long c = 0;
    #pragma unroll 7
    for (int j = 1; j < ne1; ++j, idx += K) {
        c += (long long)(vk >= edge_keys[idx]);
    }
    codes[t] = c;
}
"""
_QBIN_VARIANTS = {
    "double": {"KEY": "long long", "AS_INT": "__double_as_longlong", "SIGN": "0x7FFFFFFFFFFFFFFFLL"},
    "float": {"KEY": "int", "AS_INT": "__float_as_int", "SIGN": "0x7FFFFFFF"},
}


def _get_qbin_coder_kernel(real: str = "double"):
    """Fused one-pass coder for batched_quantile_bin_gpu: per element, count the distinct interior edge
    values <= x (plus the 2-distinct special case) in a single (n, K) read/write, instead of the
    ~3*(nbins+1) full-matrix elementwise passes the broadcast loop costs (nsys on the cupy polynom
    search: that loop was ~76%% of kernel GPU time - add 41.3%% + greater 15.8%% + bool-copy 14.1%%).
    Bit-identical: same interior/first-occurrence/ndistinct==2 terms, just fused.

    FP64-compare avoidance (ncu, cc 8.9: the double-compare loop pinned the FP64 pipeline at 84.4%% while memory sat at 24.5%% - consumer Ada runs FP64 at 1:64). Finite IEEE values order
    identically (signed compare) to key = b >= 0 ? b : b ^ 0x7FFF..F (flip magnitude bits of negatives), so convert x ONCE per element and the edges once per call, then all ne
    comparisons run on the integer pipeline. Signed zeros normalized via +0.0; non-finite values only occur in columns the caller discards wholesale. ``real`` is ``"double"`` or
    ``"float"``: a float32 column is coded without widening it (see ``batched_quantile_bin_gpu``)."""
    kern = _QBIN_CODERS.get(real)
    if kern is None:
        import cupy as cp

        src = _QBIN_CODER_TEMPLATE.replace("{REAL}", real)
        for key, value in _QBIN_VARIANTS[real].items():
            src = src.replace("{" + key + "}", value)
        kern = _QBIN_CODERS[real] = cp.RawKernel(src, "qbin_code")
    return kern


def _edge_keys(cp: Any, edges_all: Any, interior: Any, ndistinct: Any, real: str) -> Any:
    """The ``(nbins + 2, K)`` integer key table of the edges for the coder of type ``real``: non-interior slots hold the key maximum, the last row the binary-column top edge."""
    if real == "float":
        e64 = cp.ascontiguousarray(edges_all.astype(cp.float64, copy=False))
        e32 = e64.astype(cp.float32)
        e32 = cp.where(e32.astype(cp.float64) < e64, cp.nextafter(e32, cp.float32(np.inf)), e32)  # x >= edge <=> x >= the smallest float32 not below the edge
        eb = (e32 + cp.float32(0.0)).view(cp.int32)
        sign, imax, kdt = np.int32(0x7FFFFFFF), np.int32(np.iinfo(np.int32).max), np.int32
    else:
        eb = (cp.ascontiguousarray(edges_all) + 0.0).view(cp.int64)  # normalize -0.0, reinterpret bits
        sign, imax, kdt = np.int64(0x7FFFFFFFFFFFFFFF), np.int64(np.iinfo(np.int64).max), np.int64
    keys = cp.where(eb >= 0, eb, eb ^ sign)
    keys = cp.where(interior, keys, imax)  # fold the interior mask into the key table (false -> never counted)
    top = cp.where(eb[-1] >= 0, eb[-1], eb[-1] ^ sign)
    extra = cp.where(ndistinct == 2, top, imax)  # fold the ndistinct==2 extra count into one appended row
    return cp.ascontiguousarray(cp.concatenate([keys, extra[None, :]], axis=0).astype(kdt, copy=False))


def batched_quantile_bin_gpu(x_cols: Any, nbins: int) -> Any:
    """Born-on-device equi-frequency binning of an (n,K) float matrix -> (n,K) int codes, RESIDENT on GPU.

    Device twin of ``_mi_greedy_cmi_fe._quantile_bin`` applied per column. ONE batched ``cp.percentile``
    (axis=0) computes all K columns' edges in a single device sort - replacing K host ``np.quantile``
    (introselect partition) calls - then per-column ``cp.searchsorted`` on the (deduped) inner edges. The
    per-column ``cp.unique`` is over only (nbins+1) edge values (negligible), so the n-sized sort stays
    batched. Returns a cupy int64 (n,K) array (kept on device to feed ``batched_cmi_gpu`` without a code
    H2D). Selection-equivalent to the host binning: same equi-frequency partition (value-edge, rank-based).
    Assumes all-finite input (the production nan-scrubbed case); the caller falls back to the host path
    otherwise.
    """
    import cupy as cp

    from ._device_upload_f64 import upload_as_float64  # Intentional float64: the quantile-bin kernel reads a float64 column

    # The working type is the INPUT dtype: a float32 column stays float32 (no (n, K) float64 copy, half the bytes read by the edge select and the coder, and the coder compares 32-bit keys
    # against the edges rounded UP to float32: x >= edge <=> x >= ceil32(edge) for a float32 x, so the codes equal the float64 route's), every other input is binned in float64 as
    # before. Narrowing a float64 column to the criterion dtype would change its codes for no gain, so the criterion dtype is not consulted here.
    single = (x_cols.dtype == cp.float32) if isinstance(x_cols, cp.ndarray) else (getattr(x_cols, "dtype", None) == np.float32)
    if isinstance(x_cols, cp.ndarray):
        Xd = x_cols
    elif single:
        Xd = cp.asarray(np.ascontiguousarray(x_cols))
    else:
        Xd = upload_as_float64(cp, x_cols)  # Intentional float64: float64 operand
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    if not single:
        Xd = Xd.astype(cp.float64, copy=False)  # Intentional float64: GPU MI operand upload; the coder's float64 route
    n, K = int(Xd.shape[0]), int(Xd.shape[1])
    # Sort-free radix-select edges first (default ON): cp.percentile(axis=0) implements the batched edge
    # computation as ONE FULL MERGE SORT of the whole (n, K) matrix - nsys on the wellbore-100k strict fit
    # showed exactly this cub DeviceMergeSort at 74% of ALL GPU time (143.5s, 180 launches, ~2.4s each at
    # 61M elements). _radix_select_interior_edges produces bit-identical interior edges (documented maxdiff
    # 0 in the resulting codes) via rank-select without sorting; q=0/q=100 are exact column min/max (plain
    # reductions). cp.percentile stays the exact fallback when radix is inapplicable (R/shared-mem caps) or
    # gated off via MLFRAME_FE_GPU_RADIX_EDGES=0.
    edges_all = None
    try:
        from ._gpu_resident_select import _radix_select_interior_edges, fe_gpu_radix_edges_enabled
        if fe_gpu_radix_edges_enabled():
            # with_extremes=True: min/max ride along as 2 extra exact order statistics in the SAME select
            # pass (ncu/nsys-driven) instead of two standalone Xd.min(axis=0)/Xd.max(axis=0)
            # reduction kernels - returns the full (nbins+1, K) edge matrix directly, no concatenate.
            edges_all = _radix_select_interior_edges(cp.ascontiguousarray(Xd), int(nbins), with_extremes=True)
    except Exception as e:
        logger.warning("_radix_select_interior_edges failed, falling back to the host edge computation: %s", e)
        edges_all = None
    if edges_all is None:
        qs = cp.linspace(0.0, 100.0, nbins + 1)
        edges_all = device_percentile(Xd.astype(cp.float64) if single else Xd, qs, axis=0)  # (nbins+1, K) - one batched device sort; float32 input interpolates in double like the radix route  # Intentional float64: edge interpolation precision
    # Vectorized per-column coding (replaces a K-iteration Python loop of cp.unique + cp.searchsorted +
    # per-column copy - ~3 kernel launches PER COLUMN, 617 columns = ~1850 launches per call on the
    # wellbore-100k trace). The searchsorted-on-deduped-interior-edges semantics reduce to counting, per
    # row value x, the DISTINCT interior edge values <= x: with the ascending (nbins+1, K) edge matrix,
    # a first-occurrence mask kills duplicate edges, and excluding each column's min/max edge leaves the
    # interior. The <=2-distinct-edges branches fold in exactly: 1 distinct -> no interior, all-zero codes;
    # 2 distinct -> the legacy special case codes = (x >= max_edge), reproduced via the ndistinct==2 term.
    # Total launches: ~nbins+3 fused elementwise ops over (n, K), independent of K.
    e_first = edges_all[0, :]  # (K,) column min edge (q=0)
    e_last = edges_all[-1, :]  # (K,) column max edge (q=100)
    first_occ = cp.ones(edges_all.shape, dtype=cp.bool_)
    first_occ[1:, :] = edges_all[1:, :] != edges_all[:-1, :]
    interior = first_occ & (edges_all > e_first[None, :]) & (edges_all < e_last[None, :])
    ndistinct = first_occ.sum(axis=0).astype(cp.int64)  # (K,)
    try:
        # Fused one-pass coder (see _get_qbin_coder_kernel) - bit-identical to the broadcast loop below.
        real = "float" if single else "double"
        kern = _get_qbin_coder_kernel(real)
        codes = cp.empty((n, K), dtype=cp.int64)
        total = n * K
        threads = block_size_for(kern, work_items=total)
        blocks = (total + threads - 1) // threads
        keys = _edge_keys(cp, edges_all, interior, ndistinct, real)
        kern((int(blocks),), (threads,), (
            cp.ascontiguousarray(Xd), keys,
            np.int64(n), np.int64(K), np.int32(keys.shape[0]), codes,
        ))
        return codes
    except Exception:
        logger.debug("qbin fused coder failed; broadcast-loop fallback", exc_info=True)
    codes = cp.zeros((n, K), dtype=cp.int64)
    for j in range(1, int(edges_all.shape[0])):  # j=0 is never interior (== column min)
        codes += (interior[j, :][None, :] & (Xd >= edges_all[j, :][None, :])).astype(cp.int64)
    codes += ((ndistinct == 2)[None, :] & (Xd >= e_last[None, :])).astype(cp.int64)
    return codes
