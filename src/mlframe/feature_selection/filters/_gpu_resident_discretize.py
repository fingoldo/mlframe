"""GPU-resident FE: fused per-column binning + resident discretize codes (carve sibling of _gpu_resident_select).

Carved VERBATIM out of ``_gpu_resident_select.py`` (sibling re-export pattern) to bring the parent under the
1k-LOC ceiling. Holds the fused ``bin_codes`` RawKernel + its lazy getter, ``_searchsorted_codes``, and the
resident ``_gpu_resident_discretize_codes`` path. The parent re-exports every name moved here so all existing
``from ._gpu_resident_select import X`` / ``from .._gpu_resident_fe import X`` paths still resolve byte-for-byte.

The radix-select edge primitives (``_radix_select_interior_edges`` / ``fe_gpu_radix_edges_enabled``) stay in
the PARENT and are LAZY-imported inside the function bodies to avoid an import cycle. No kernel-source,
dispatch-threshold, or selection behavior changed.
"""
from __future__ import annotations

import os

import numpy as np

# Parent-of-the-FE-block name consumed at module scope (defined in _gpu_resident_fe before it imports the
# select parent, so this top import resolves during the partial-init import chain -- same pattern as the parent).
from ._gpu_resident_fe import _quantile_levels_dev


# FUSED PER-COLUMN BINNING (2026-06-20, nvprof-driven). The per-column ``for j in range(K): out[:,j] =
# cp.searchsorted(edges[:,j], col, 'right')`` loop fired K separate searchsorted launches PLUS K int64->
# int32 cast-copies (searchsorted returns int64, ``out`` is int32). nvprof on the n=100k/300k binning path:
# cupy_copy__int64_int32 = 19.2% of GPU time (2304 calls) + cupy_searchsorted_kernel = 11.7% (2304 calls)
# -- ~31% of GPU time in launch overhead + a needless dtype cast. This ONE kernel bins the whole (n,K)
# matrix: each thread takes one element, binary-searches its column's nbins-1 interior edges (upper_bound
# = count of edges <= value = EXACTLY cp.searchsorted(.., side='right')), and writes the int32 code
# directly -- coalesced cand/out (row-major) + coalesced strided edge reads (consecutive threads = adjacent
# columns). BIT-IDENTICAL to the per-column searchsorted; one launch, no int64 intermediate.
#
# coalescing-audit (2026-06-23): ALREADY COALESCED -- at floor. CUDA-event A/B at the production shape
# (n=100k, K=583, nbins=10, GTX 1050 Ti): 11.96ms (thread ``idx`` reads cand[idx] coalesced, col=idx%K, and
# writes out[idx] coalesced). A (K,n) column-major-input variant was tried (to mirror the materialise/radix
# wins): 70.3ms = 5.9x SLOWER -- the (n,K) row-major OUTPUT write out[row*K+col] becomes stride-K
# uncoalesced when threads are laid out column-major. The row-major layout is the coalesced one for BOTH the
# value read and the code write here; do not re-audit.
# Edges are ALWAYS float64 (cp.percentile and the radix-select both produce f64 edges). The value is
# promoted to double for the compare -- EXACTLY what cp.searchsorted(f64_edges, f32_value) does (it
# upcasts the value to the edges' dtype). Comparing in the value's f32 instead would 1-off at boundaries
# (a downcast of the f64 edge loses precision) -- the bug that broke bit-identity on the first cut.
#
# bench-attempt-rejected (2026-06-27): nsys re-profile (iter 2) made bin_codes_f32 the #3 GPU kernel (171ms/
# 12 inst); nvprof showed gld_efficiency=51.7% (the divergent f64 edge reads drag the f32 cand read down)
# while gst=100%, achieved_occupancy=0.92, DRAM ~38GB/s (40% of the card's ~96GB/s). Two BIT-IDENTICAL
# (maxdiff 0) alternatives to lift gld, both SLOWER at the production shape (n=100k, K=583, nbins=10, GTX
# 1050 Ti, interleaved-min >=20 reps): (a) unrolled LINEAR scan of all ne edges (uniform coalesced row order,
# no branch divergence) 11.6->16.1ms = SLOWER (ne=9 extra edge reads outweigh the divergence saving); (b)
# stage the whole (ne,K) edge table (~40KB) into SHARED so divergent edge reads hit shared not global
# 11.6->82.7ms = 7x SLOWER (40KB shared caps the block to 1/SM, occupancy collapses). The gld inefficiency is
# the intrinsic f64-vs-f32 mixed-width access of per-element binning; lifting it costs more than it saves on
# Pascal. VERDICT: at its practical floor for this card.
_BIN_CODES_SRC = r"""
extern "C" __global__
void bin_codes_TYPENAME(const TYPE* __restrict__ cand, const double* __restrict__ edges,
                        const long long n, const int K, const int ne, OUTTYPE* __restrict__ out) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (idx >= total) return;
    int col = (int)(idx % (long long)K);
    double v = (double)cand[idx];
    int lo = 0, hi = ne;                       // upper_bound over this column's interior edges
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (edges[(long long)mid * K + col] <= v) lo = mid + 1; else hi = mid;
    }
    out[idx] = (OUTTYPE)lo;                     // = #(edges <= v) = searchsorted(.., 'right')
}
"""

# LAUNCH-FUSION (2026-06-27, bit-identical): emit the NARROW code dtype (int8/int16) straight from the
# binning kernel instead of int32-then-astype. The resident chunk path discretized to int32 (this kernel)
# then immediately cast int32->int8 per VRAM chunk (a separate launch + a full (n,K) int32 intermediate,
# ~12-24x/fit). Instantiating the kernel's OUTPUT type directly (OUTTYPE) drops that cast launch + the int32
# buffer. nbins<=255 -> int8 cannot overflow (codes are in [0, nbins-1]); the on-device write is the same
# upper_bound value, only narrower -- BIT-IDENTICAL to int32-then-astype. The default out_dtype stays int32
# so every other caller (basis/hermite) is byte-for-byte unchanged.
_BIN_CODES_OUTTYPE = {"int8": "signed char", "int16": "short", "int32": "int", "int64": "long long"}

_BIN_CODES_KERNELS: dict = {}


def _get_bin_codes_kernel(dtype, out_name: str = "int32"):
    """Lazy-compiled (pickle-safe, module-level cache) fused binning RawKernel for f32 / f64 input
    and a templated narrow OUTPUT dtype (int8/int16/int32/int64). Keyed on (input, output)."""
    import cupy as cp

    in_key = "f64" if dtype == cp.float64 else "f32"
    cache_key = in_key + "_" + out_name
    k = _BIN_CODES_KERNELS.get(cache_key)
    if k is None:
        ctype = "double" if in_key == "f64" else "float"
        out_ctype = _BIN_CODES_OUTTYPE.get(out_name, "int")
        # TYPENAME is part of the C symbol name (must be a valid identifier); fold the output dtype into it
        # so the int8/int16/int32 kernels get distinct symbols within one translation unit per instantiation.
        sym = "bin_codes_" + in_key + "_" + out_name
        src = (_BIN_CODES_SRC
               .replace("bin_codes_TYPENAME", sym)
               .replace("OUTTYPE", out_ctype)     # MUST precede the TYPE replace (TYPE is a substring of OUTTYPE)
               .replace("TYPE", ctype))
        k = cp.RawKernel(src, sym)
        _BIN_CODES_KERNELS[cache_key] = k
    return k


def _searchsorted_codes(cand_gpu, interior_edges, out_dtype=None):
    """Bin (n,K) ``cand_gpu`` against per-column ascending ``interior_edges`` (ne,K) -> int32 (n,K) codes,
    code = #(interior edges <= value) (== per-column cp.searchsorted side='right'). One fused kernel
    launch (no K searchsorted launches, no int64->int32 cast). Falls back to the per-column loop on any
    kernel failure -- bit-identical either way."""
    import cupy as cp

    n, K = cand_gpu.shape
    try:
        # cand_gpu is already C-contiguous f32 on the production path (RawKernel cp.empty output / cp.asarray
        # of a C-contiguous host slice); cp.ascontiguousarray would still alloc+copy the whole (n,K) matrix
        # (the nvprof cupy_copy__float32_float32 hotspot, 19.7%). The kernel only needs C-order memory -> reuse
        # the buffer when already contiguous (bit-identical bytes); a strided view still gets the safety copy.
        cand_c = cand_gpu if cand_gpu.flags.c_contiguous else cp.ascontiguousarray(cand_gpu)
        edges_c = cp.ascontiguousarray(interior_edges, dtype=cp.float64)  # edges f64 (match cp.searchsorted promotion)
        ne = int(edges_c.shape[0])
        # out_dtype lets the resident chunk path emit the NARROW code dtype (int8/int16) DIRECTLY (launch-fusion:
        # drops the downstream int32->int8 astype launch + the int32 (n,K) intermediate). Default int32 keeps
        # every other caller byte-identical. nbins<=255 -> int8 cannot overflow (codes in [0, nbins-1]).
        _od = cp.dtype(out_dtype) if out_dtype is not None else cp.dtype(cp.int32)
        out = cp.empty((n, K), dtype=_od)
        total = n * K
        threads = 256
        blocks = (total + threads - 1) // threads
        _get_bin_codes_kernel(cand_c.dtype, _od.name)(
            (blocks,), (threads,),
            (cand_c, edges_c, np.int64(n), np.int32(K), np.int32(ne), out),
        )
        return out
    except Exception:
        import logging
        logging.getLogger(__name__).debug("fused bin-codes kernel failed; per-column searchsorted fallback", exc_info=True)
        _od = cp.dtype(out_dtype) if out_dtype is not None else cp.dtype(cp.int32)
        out = cp.empty((n, K), dtype=_od)
        ec = cp.ascontiguousarray(interior_edges)
        for j in range(K):
            out[:, j] = cp.searchsorted(ec[:, j], cand_gpu[:, j], side="right")
        return out


def _gpu_resident_discretize_codes(cand_gpu, nbins: int, out_dtype=None, cm_hint=None):
    """Quantile-bin a RESIDENT (n, K) cupy candidate matrix to ordinal codes ON the GPU. Mirrors
    ``discretize_2d_array_cuda`` -- ``cp.percentile(.., linspace(0,100,nbins+1), axis=0)`` for per-column
    edges + per-column ``cp.searchsorted(edges[1:-1], col, side='right')`` -- but keeps the input + output
    on-device (no H2D of the big candidate matrix, no D2H of codes here), so it chains gen -> discretize ->
    noise-gate without round-trips. Returns a cupy int32 (n, K) codes array (resident).

    DTYPE: the percentile + searchsorted run in the INPUT's native dtype by default -- so the float32 FE
    candidate buffer stays float32 (no up-cast; float32 halves the bandwidth of the dominant full sort
    cp.percentile does and preserves the FE selection, the acceptance bar) while the float64 grand-fused
    MI path stays float64 (bit-identical). ``MLFRAME_FE_GPU_BINNING_DTYPE=float64`` forces the exact f64
    path host-wide (bit-identical to the CPU ``discretize_2d_quantile_batch``, whose ``np.percentile``
    upcasts float32 to float64); ``=float32`` forces f32."""
    import cupy as cp

    # Bin in the input's NATIVE dtype by default (the float32 FE candidate buffer stays float32 -- no
    # up-cast, half the sort bandwidth; the float64 grand-fused MI path stays float64 -- bit-identical).
    # MLFRAME_FE_GPU_BINNING_DTYPE forces a specific working dtype (float64 = the exact CPU-parity fallback).
    # cross-sibling (parent) radix-select edge primitives: lazy-imported to avoid an import cycle.
    from ._gpu_resident_select import _radix_select_interior_edges, fe_gpu_radix_edges_enabled
    forced = os.environ.get("MLFRAME_FE_GPU_BINNING_DTYPE", "").strip().lower()
    if forced in ("float64", "f64", "double"):
        work = cp.float64
    elif forced in ("float32", "f32", "single"):
        work = cp.float32
    else:
        work = cand_gpu.dtype
    if cand_gpu.dtype != work:
        cand_gpu = cand_gpu.astype(work, copy=False)
    n, K = cand_gpu.shape

    # RANK-EXACT SORT-FREE EDGES (roadmap #2): extract just the nbins-1 interior quantile edges via the
    # radix-select kernel instead of cp.percentile's full sort. Bit-identical codes (verified maxdiff 0),
    # faster (win grows with n). Returns None -> cp.percentile fallback (R over cap / shared-mem over the
    # device limit); any kernel exception also falls back. cp.percentile's interior edges are bin_edges[1:-1].
    if fe_gpu_radix_edges_enabled() and n > 0:
        try:
            # Already C-contiguous here (see _searchsorted_codes note); _radix_select_interior_edges does its
            # OWN coalescing transpose internally and only needs C-order input, so skip the redundant full
            # (n,K) f32 copy when contiguous (bit-identical edges -> codes). The KEEP transpose stays inside it.
            _cand_c = cand_gpu if cand_gpu.flags.c_contiguous else cp.ascontiguousarray(cand_gpu)
            # cm_hint only valid when cand wasn't re-copied (else its shape/values may not match _cand_c bytes).
            _hint = cm_hint if _cand_c is cand_gpu else None
            interior = _radix_select_interior_edges(_cand_c, int(nbins), cm_hint=_hint)
        except Exception:
            import logging
            logging.getLogger(__name__).debug("radix-select edges failed; cp.percentile fallback", exc_info=True)
            interior = None
        if interior is not None:
            return _searchsorted_codes(cand_gpu, interior, out_dtype=out_dtype)

    qs = _quantile_levels_dev(cp, nbins, work)
    if K == 1:
        # CUPY BUG GUARD: cp.percentile(X, axis=0) returns WRONG edges for a single-column (n, 1) array
        # (verified maxdiff ~23 vs numpy; multi-column is exact). A K==1 chunk occurs whenever the last
        # candidate block holds one column, which would silently corrupt that column's codes (breaking the
        # discretize bit-identity). Ravel to 1D where cp.percentile is correct, then restore the shape.
        bin_edges = cp.percentile(cand_gpu.ravel(), qs).reshape(-1, 1)  # (nbins+1, 1)
    else:
        bin_edges = cp.percentile(cand_gpu, qs, axis=0)  # (nbins+1, K)
    return _searchsorted_codes(cand_gpu, bin_edges[1:-1], out_dtype=out_dtype)
