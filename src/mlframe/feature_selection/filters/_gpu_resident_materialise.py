"""GPU-resident FE: chunk-materialise kernels + operand-table residency + host fast paths (carve sibling).

Carved VERBATIM out of ``_gpu_resident_select.py`` (sibling re-export pattern) to bring the parent under the
1k-LOC ceiling. Holds the row-major + coalesced column-major ``fe_materialise`` RawKernels, the operand-table
residency caches/build helpers (incl. the batched plain-unary op-code kernel + ``build_resident_operand_table``),
the pinned-D2H staging buffer, and the GPU materialise/discretize codes host fast paths.

The transpose kernels (``_transpose_to_cm`` / ``_transpose_cm_to_rm``) and the resident discretize path
(``_gpu_resident_discretize_codes``) stay in the PARENT / the discretize sibling and are LAZY-imported inside
the function bodies to avoid import cycles (mirroring the prior in-file layout); ``_gpu_apply_prewarp`` is
lazy-imported from ``_gpu_resident_fe`` as before. The parent re-exports every name moved here so all existing
import paths still resolve byte-for-byte. No kernel-source, residency, or selection behavior changed.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any

import numpy as np
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# Parent-of-the-FE-block names consumed by the moved host fast paths (defined in _gpu_resident_fe before it
# imports the select parent -> resolve during the partial-init import chain, same pattern as the parent).
from ._gpu_resident_shared import (
    _gpu_k_chunk,
    _stash_deferred_host_fill,
    _stash_resident_codes,
    fe_gpu_defer_host_codes_enabled,
    fe_gpu_resident_codes_enabled,
)

# CHUNK-MATERIALISE CUDA RawKernel. The FE chunk path's #1 CPU hotspot is
# ``_materialise_chunk_njit`` - it builds the (n, K) float32 candidate matrix by gathering strided
# operand columns ``tv[r, ai]`` / ``tv[r, bi]`` out of a row-major operand table and applying the
# binary op-code table (mlframe.feature_selection.filters._feature_engineering_pairs._pairs_materialise
# ._NJIT_BINARY_OP_CODES). It is MEMORY-BANDWIDTH bound on those gathers, not compute. This kernel does
# the IDENTICAL work on the GPU: each thread owns one (row, candidate) cell, gathers its two operand
# columns by op-code index, applies the binary op, scrubs non-finite -> 0, and writes float32 row-major.
#
# BIT-IDENTICAL to ``_materialise_chunk_njit``: operands are read as float32 (the ``tv`` dtype); mul/add/
# sub/abs_diff are plain float32 ops; max/min/signed propagate NaN exactly (``a+b`` when either is NaN);
# div (op 3) and ratio_abs (op 8) are FLOAT64-PROMOTED then cast back to float32 (matching the njit
# kernel's ``np.float32(np.float64(a)/...)`` - numba/numpy promote the float64 ``1e-9`` / ``1.0``
# literals); the final nan_to_num(nan=0, +-inf=0) is the same predicate. The op-code numbering is the
# njit table: 0=mul 1=add 2=sub 3=div 4=max 5=min 6=abs_diff 7=signed 8=ratio_abs. ``tv`` is the
# (n, n_operands) row-major float32 operand table; the kernel addresses operand column ``c`` of row
# ``i`` via ``tv[i*n_operands + c]`` (so NO transpose is needed - it mirrors the njit ``tv[r, ai]``).
_FE_MATERIALISE_SRC = r"""
extern "C" __global__
void fe_materialise(const float* __restrict__ tv,
                    const long long* __restrict__ a_cols,
                    const long long* __restrict__ b_cols,
                    const signed char* __restrict__ ops,
                    const long long n, const long long n_operands, const int K,
                    float* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (tid >= total) return;
    int k = (int)(tid % (long long)K);
    long long i = tid / (long long)K;
    long long ai = a_cols[k];
    long long bi = b_cols[k];
    float a = tv[i * n_operands + ai];
    float b = tv[i * n_operands + bi];
    int op = (int)ops[k];
    float v;
    if (op == 0) {            // mul
        v = a * b;
    } else if (op == 1) {     // add
        v = a + b;
    } else if (op == 2) {     // sub
        v = a - b;
    } else if (op == 3) {     // div = _safe_div (2026-06-13 form): exact x/y for y!=0, eps floor only on exact-zero
        v = (float)((double)a / ((b == 0.0f) ? 1e-9 : (double)b));
    } else if (op == 4) {     // max = np.maximum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a > b) ? a : b;
    } else if (op == 5) {     // min = np.minimum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a < b) ? a : b;
    } else if (op == 6) {     // abs_diff = |a - b|
        v = fabsf(a - b);
    } else if (op == 7) {     // signed = sign(a)*|b| (nan-propagating)
        if (a != a || b != b) {
            v = a + b;
        } else {
            float sgn = (a == 0.0f) ? 0.0f : ((a > 0.0f) ? 1.0f : -1.0f);
            v = sgn * fabsf(b);
        }
    } else {                  // op == 8: ratio_abs = double precision-promoted a/(|b|+1)
        v = (float)((double)a / ((double)fabsf(b) + 1.0));
    }
    // np.nan_to_num(nan=0, posinf=0, neginf=0)
    if (isnan(v) || isinf(v)) v = 0.0f;
    out[i * (long long)K + k] = v;
}
"""
_FE_MATERIALISE_KERNEL = None  # module-level singleton (lazy-compiled; never on an instance -> pickle-safe)


def _get_fe_materialise_kernel():
    """Lazily compile (once) and return the module-level row-major ``fe_materialise`` RawKernel singleton."""
    global _FE_MATERIALISE_KERNEL
    if _FE_MATERIALISE_KERNEL is None:
        import cupy as cp

        _FE_MATERIALISE_KERNEL = cp.RawKernel(_FE_MATERIALISE_SRC, "fe_materialise")
    return _FE_MATERIALISE_KERNEL


# COALESCED COLUMN-MAJOR fe_materialise (coalescing audit - the SAME stride-uncoalesced-read
# lever that won 5.59x on the noise-gate hist kernel, applied to the materialise). CUDA-event decomposition
# at the production block shape (n=100k, K=1200, n_operands=64, GTX 1050 Ti) showed the row-major kernel
# above runs at only ~20 GB/s vs the card's ~94 GB/s coalesced floor (write-only baseline 5.1ms @ 94 GB/s
# vs the full kernel 72ms): with thread ``tid -> (i = tid//K, k = tid%K)`` consecutive threads share row
# ``i`` and read ``tv[i*n_operands + a_cols[k]]`` - a SCATTERED per-candidate operand-column gather (each
# warp touches 32 unrelated operand columns of the same row) -> ~1/5 effective bandwidth.
#
# This variant flips the thread mapping to ``tid -> (k = tid//n, i = tid%n)`` and reads from a COLUMN-MAJOR
# operand table ``tv_cm`` (n_operands, n): consecutive threads (consecutive ``i`` within a fixed candidate
# ``k``) read ``tv_cm[ai*n + i]`` = CONSECUTIVE memory -> fully coalesced operand loads, and write the
# (K, n) column-major output ``out[k*n + i]`` coalesced too. The per-element math (op-code table, float64-
# promoted div/ratio_abs, NaN/inf scrub) is BYTE-FOR-BYTE the row-major kernel's (verified array_equal vs
# fe_materialise across n in {3k,10k,40k} x K in {1,50,257,583} incl. zeros/negatives/+-inf). Caller
# transposes the (n, K) operand table to (K=n_operands, n) ONCE per step (cached, ~0.7ms at n=100k) and
# transposes the (K, n) result back to the (n, K) row-major layout the downstream bin/D2H expect via the
# coalesced tiled-transpose kernel. NET (interleaved-min CUDA-event A/B, 2x-confirmed, GTX 1050 Ti):
# n=100k K=583 36.1->18.4ms = 1.96x; n=100k K=1200 73.5->36.1ms = 2.03x (incl. the tv-transpose + the
# result transpose-back). Gated ON; ``MLFRAME_FE_GPU_MATERIALISE_CM=0`` forces the row-major kernel; any
# transpose/compile/launch failure falls back to it -> CPU / no-CUDA path byte-unchanged.
_FE_MATERIALISE_CM_SRC = r"""
extern "C" __global__
void fe_materialise_cm(const float* __restrict__ tv_cm,
                       const long long* __restrict__ a_cols,
                       const long long* __restrict__ b_cols,
                       const signed char* __restrict__ ops,
                       const long long n, const long long n_operands, const int K,
                       float* __restrict__ out) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (tid >= total) return;
    long long k = tid / n;                 // candidate index (consecutive threads share k)
    long long i = tid - k * n;             // row index (consecutive -> coalesced over n)
    long long ai = a_cols[k];
    long long bi = b_cols[k];
    float a = tv_cm[ai * n + i];           // COLUMN-MAJOR: consecutive threads read consecutive memory
    float b = tv_cm[bi * n + i];
    int op = (int)ops[k];
    float v;
    if (op == 0) {            // mul
        v = a * b;
    } else if (op == 1) {     // add
        v = a + b;
    } else if (op == 2) {     // sub
        v = a - b;
    } else if (op == 3) {     // div = _safe_div (exact x/y for y!=0, eps floor only on exact-zero)
        v = (float)((double)a / ((b == 0.0f) ? 1e-9 : (double)b));
    } else if (op == 4) {     // max = np.maximum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a > b) ? a : b;
    } else if (op == 5) {     // min = np.minimum (nan-propagating)
        if (a != a || b != b) v = a + b; else v = (a < b) ? a : b;
    } else if (op == 6) {     // abs_diff = |a - b|
        v = fabsf(a - b);
    } else if (op == 7) {     // signed = sign(a)*|b| (nan-propagating)
        if (a != a || b != b) {
            v = a + b;
        } else {
            float sgn = (a == 0.0f) ? 0.0f : ((a > 0.0f) ? 1.0f : -1.0f);
            v = sgn * fabsf(b);
        }
    } else {                  // op == 8: ratio_abs = double precision-promoted a/(|b|+1)
        v = (float)((double)a / ((double)fabsf(b) + 1.0));
    }
    if (isnan(v) || isinf(v)) v = 0.0f;
    out[k * n + i] = v;                     // COLUMN-MAJOR (K, n) output -> coalesced
}
"""
_FE_MATERIALISE_CM_KERNEL = None  # module-level singleton (lazy-compiled; pickle-safe)

# tv -> (n_operands, n) column-major copy cache (weakref-identity, mirrors _OPERAND_TABLE_CACHE): the
# operand table is the SAME device array across a step's blocks/chunks, so transpose it ONCE per step.
_OPERAND_TABLE_CM_CACHE: dict = {"ref": None, "cm": None}
# Same concurrent-chunk-pipeline race as _OPERAND_TABLE_CACHE_LOCK above: two threads racing the same
# cache-miss branch would both pay the transpose and the loser's result silently clobbers the winner's.
_OPERAND_TABLE_CM_CACHE_LOCK = threading.Lock()


def fe_gpu_materialise_cm_enabled() -> bool:
    """Whether the COALESCED column-major fe_materialise (coalescing audit, ~2x net) is active. DEFAULT ON
    (opt-out ``MLFRAME_FE_GPU_MATERIALISE_CM=0``). Bit-identical (array_equal) to the row-major kernel; the
    row-major kernel stays the fallback on any transpose/compile/launch failure (CPU / no-CUDA unchanged)."""
    return os.environ.get("MLFRAME_FE_GPU_MATERIALISE_CM", "1").strip().lower() not in ("0", "false", "no", "off")


def _get_fe_materialise_cm_kernel():
    """Lazily compile (once) and return the module-level coalesced column-major ``fe_materialise_cm`` RawKernel singleton."""
    global _FE_MATERIALISE_CM_KERNEL
    if _FE_MATERIALISE_CM_KERNEL is None:
        import cupy as cp

        _FE_MATERIALISE_CM_KERNEL = cp.RawKernel(_FE_MATERIALISE_CM_SRC, "fe_materialise_cm")
    return _FE_MATERIALISE_CM_KERNEL


def _operand_table_cm(cp, tv_gpu):
    """(n_operands, n) column-major (= C-contiguous transpose of the (n, n_operands) row-major ``tv_gpu``)
    copy, cached by weakref identity of ``tv_gpu`` so the transpose is paid ONCE per step (the operand table
    is the same device object across the step's materialise blocks). Uses the coalesced tiled-transpose
    kernel; falls back to ``cp.ascontiguousarray(tv_gpu.T)`` (bit-identical) for non-f32 / non-contiguous /
    any kernel failure."""
    import weakref
    from ._gpu_resident_select import _transpose_to_cm  # parent transpose kernel (lazy: avoid cycle)

    c = _OPERAND_TABLE_CM_CACHE
    with _OPERAND_TABLE_CM_CACHE_LOCK:
        ref = c["ref"]
        if ref is not None and ref() is tv_gpu and c["cm"] is not None:
            return c["cm"]
        cm = _transpose_to_cm(tv_gpu)  # (n_operands, n) C-order
        try:
            c["ref"] = weakref.ref(tv_gpu)
            c["cm"] = cm
        except TypeError:
            c["ref"] = None
            c["cm"] = None
        return cm


def _fe_materialise_block_gpu(tv_gpu, a_cols_block, b_cols_block, ops_block, return_cm=False):
    """Generate the (n, len(ops_block)) float32 candidate matrix for the given column blocks in ONE kernel
    launch, RESIDENT on the GPU. ``tv_gpu`` is the (n, n_operands) row-major float32 operand table already
    on the device. ``a_cols_block`` / ``b_cols_block`` (int64) / ``ops_block`` (int8) are host or device
    arrays of length K. Returns a row-major (n, K) cupy float32 matrix, BIT-EQUAL to
    ``_materialise_chunk_njit`` (same float32 ops, same double precision-promoted div/ratio_abs, same nan_to_num)."""
    import cupy as cp
    from ._gpu_resident_select import _transpose_cm_to_rm  # parent transpose kernel (lazy: avoid cycle)

    n = int(tv_gpu.shape[0])
    n_operands = int(tv_gpu.shape[1])
    K = len(ops_block)
    a_g = cp.asarray(a_cols_block, dtype=cp.int64)
    b_g = cp.asarray(b_cols_block, dtype=cp.int64)
    ops_g = cp.asarray(ops_block, dtype=cp.int8)
    total = n * K
    threads = 256
    blocks = (total + threads - 1) // threads

    # COALESCED column-major path (coalescing audit, ~2x net): materialise into a (K, n) column-major buffer
    # from the (n_operands, n) column-major operand table (coalesced operand gathers + coalesced write), then
    # transpose the result back to the (n, K) row-major layout the downstream bin/D2H expect. Bit-identical
    # (array_equal) to the row-major kernel; falls back to it on any transpose/compile/launch failure.
    if fe_gpu_materialise_cm_enabled() and n > 0 and K > 0:
        try:
            tv_cm = _operand_table_cm(cp, tv_gpu)  # (n_operands, n) C-order, once/step (cached)
            cm_out = cp.empty((K, n), dtype=cp.float32)
            _get_fe_materialise_cm_kernel()(
                (blocks,),
                (threads,),
                (tv_cm, a_g, b_g, ops_g, np.int64(n), np.int64(n_operands), np.int32(K), cm_out),
            )
            rm = _transpose_cm_to_rm(cm_out)  # (K, n) -> (n, K) row-major (coalesced)
            # LAUNCH-FUSION: hand the cm buffer back so the radix-select edges step reuses it
            # instead of transposing rm back to cm (the inverse of the transpose just done). Bit-identical.
            return (rm, cm_out) if return_cm else rm
        except Exception:
            import logging

            logging.getLogger(__name__).debug("column-major fe_materialise failed; row-major fallback", exc_info=True)

    out = cp.empty((n, K), dtype=cp.float32)
    _get_fe_materialise_kernel()(
        (blocks,),
        (threads,),
        (tv_gpu, a_g, b_g, ops_g, np.int64(n), np.int64(n_operands), np.int32(K), out),
    )
    # No cm buffer on the row-major fallback path (cm_hint=None -> discretize transposes as before).
    return (out, None) if return_cm else out


# PINNED D2H STAGING for the out_cand float buffer (nvprof+paired-microbench driven).
# The downstream survivor/usability reads need the (n,K) float candidate matrix on host, so out_cand is
# unavoidable - but ``cp.asnumpy(cand)`` copies into the caller's PAGEABLE buffer, which makes cupy stage
# the D2H through an internal pinned bounce buffer at PAGEABLE PCIe bandwidth (the #1 production wall:
# cProfile cupy.get = 9.07s, 321 blocking syncs). DMA'ing the chunk into a PERSISTENT PINNED host buffer
# first, then a plain host->host memcpy into the caller's pageable slice, runs the device transfer at full
# pinned bandwidth. MEASURED GTX 1050 Ti, (100k, blk=1200) f32 = 480MB: the device D2H 143ms->75ms (1.9x);
# end-to-end into a pageable slice incl. the added host memcpy 209ms->130ms (1.6x); the whole materialise+
# bin+codes call (K=1200) 696ms->~575ms with the float path on. The buffer is a module-level singleton
# (never on an instance -> pickle-safe), grown on demand and reused across the 15 canonical chunks.
# bench-attempt-rejected (2026-06-21, prior): DEFERRING the float D2H entirely (out_cand=None + downstream
# recompute) was a 0.98x fit-level WASH because removing an overlapped transfer cuts no wall - but here we
# do NOT remove it, we make the SAME bytes move faster (pinned DMA), which DOES cut the blocking-sync wall.
# Thread-local so two concurrent GPU callers never share (and clobber each other's) the same pinned DMA
# staging buffer: each thread DMAs into its OWN pinned allocation. A single module-level singleton would
# have two threads' ``get(out=view)`` writing the same host bytes, corrupting both transfers.
_PINNED_D2H_TLS = threading.local()


def clear_pinned_d2h() -> bool:
    """Release the calling thread's pinned D2H staging buffer so page-locked host memory is freed (e.g. at fit completion).

    The staging buffer is thread-local; this clears only the current thread's allocation (the only one it can safely
    reach). Returns True if a buffer was present and dropped, False otherwise.
    """
    had = bool(getattr(_PINNED_D2H_TLS, "bufs", None))
    _PINNED_D2H_TLS.bufs = None
    return had


def _pinned_view(n_bytes: int, shape, dtype, slot: int = 0):
    """A pinned-host numpy view of at least ``n_bytes``, reshaped to ``shape`` (``dtype``). Reuses a
    THREAD-LOCAL pinned allocation, growing it on demand. Lets ``cupy.ndarray.get(out=...)`` DMA at full
    pinned PCIe bandwidth instead of cp.asnumpy's pageable bounce-buffer path. Thread-local (not a shared
    singleton) so concurrent GPU callers don't clobber each other's staging; module-level (not on an
    estimator instance) -> never reachable from pickled state.

    ``slot`` selects an independent buffer (default 0 = the historical single buffer). The async
    DOUBLE-BUFFERED D2H pipeline alternates slots 0/1 so block k+1's copy can stream into one buffer while
    block k's is still being drained into the caller's pageable array."""
    import cupy as cp

    bufs = getattr(_PINNED_D2H_TLS, "bufs", None)
    if bufs is None:
        bufs = {}
        _PINNED_D2H_TLS.bufs = bufs
    buf = bufs.get(slot)
    if buf is None or buf.mem.size < n_bytes:
        buf = cp.cuda.alloc_pinned_memory(int(n_bytes))
        bufs[slot] = buf
    count = int(np.prod(shape))
    return np.frombuffer(buf, dtype=dtype, count=count).reshape(shape)


# Operand-table H2D cache: the FE step's operand table ``transformed_vars`` is the SAME
# array object across all ~15 chunks of a step, but was re-uploaded to the GPU per chunk (and again per
# survivor re-materialise). Cache the device copy by WEAKREF IDENTITY of the host array: reuse while the
# same object is alive (across the step's chunks), re-upload when the step swaps in a new operand table
# (the weakref breaks). NOT keyed on id() - id reuse after free would false-hit on a different table.
# Pickle-safe (module-global, never on an instance). The data is identical -> candidates/codes/MI/
# selection bit-identical; this only moves the H2D from per-chunk to once-per-step.
# Per-host-object device cache (was a single slot, which two interleaved steps clobbered: step B's upload
# overwrote step A's device table, so A's still-running chunks read B's bytes). Keyed by id() but each entry
# carries a WEAKREF to the host array, so an id-recycle after free can never false-hit (the weakref must
# resolve to the SAME live object). Bounded FIFO so distinct operand tables across a long fit don't grow it.
_OPERAND_TABLE_CACHE: "OrderedDict[int, tuple]" = OrderedDict()  # id(host) -> (weakref(host), gpu)
_OPERAND_TABLE_CACHE_MAX = 8
# The chunk-pipeline feature runs chunk k+1's
# production (in a ThreadPoolExecutor(max_workers=1)) concurrently with the main thread consuming chunk k,
# and both threads can reach into this cache (and _PREBUILT_OPERAND_TABLE below) for the SAME
# transformed_vars object with no lock - under the GIL this cannot corrupt memory, but a cupy call that
# releases the GIL mid-lookup (H2D transfer) lets both threads race the same cache-miss branch (duplicate
# uploads) or interleave an eviction with a concurrent move_to_end. Narrow blast radius today (gated behind
# default-OFF fe_gpu_strict_enabled()), but this lock closes the race regardless.
_OPERAND_TABLE_CACHE_LOCK = threading.Lock()


# GPU-RESIDENT OPERAND TABLE (phase 1 of the 100%-GPU-resident MRMR FE rewrite, gated).
# The operand table ``transformed_vars`` (n, n_operands) float32 is built on the CPU in
# ``check_prospective_fe_pairs`` (one column per (var, unary)), then ``_resident_operand_table`` H2Ds it to
# the device ONCE per step. Phase 1 removes even that single H2D by building the device mirror's columns ON
# the GPU directly from the resident raw operand inputs (via ``_unary_apply`` - the same math as the CPU
# ``unary_transformations``), so the materialise consumes a DEVICE array with NO host->device transfer of
# the bulk operand bytes. The CPU ``transformed_vars`` is STILL built (the pair-search inner loops /
# discretize read it on the host - those move to the GPU in later phases); phase 1 only kills the
# materialise H2D. Operand transforms that are NOT plain GPU unaries (prewarp / gate_med / hermite-poly -
# fitted/special, no straightforward cupy form) are built on the CPU and copied into the resident mirror (a
# few columns); the bulk plain-unary columns are GPU-built. The PREBUILT mirror is registered here by
# weakref-identity of the host ``transformed_vars`` so ``_resident_operand_table`` returns it WITHOUT the
# H2D. Module-global -> never reachable from pickled estimator state. Gated OFF by default
# (``MLFRAME_FE_GPU_RESIDENT_OPERANDS``) until proven 11-green; the CPU / no-CUDA path is unchanged.
# Per-host-object prebuilt-mirror registry (was a single slot, clobbered when two concurrent steps each
# registered their own GPU-resident mirror). Keyed by id() with a co-validating weakref so an id-recycle
# can't return a stale/wrong-table mirror; bounded FIFO. ``device_table=None`` clears the entry for that host.
_PREBUILT_OPERAND_TABLE: "OrderedDict[int, tuple]" = OrderedDict()  # id(host) -> (weakref(host), gpu)
_PREBUILT_OPERAND_TABLE_MAX = 8
_PREBUILT_OPERAND_TABLE_LOCK = threading.Lock()  # - see _OPERAND_TABLE_CACHE_LOCK's note


_OPERAND_CACHE_VRAM_FRACTION = 0.25


def _operand_cache_budget_bytes() -> int:
    """Byte cap for the device operand-table caches: ``MLFRAME_FE_OPERAND_CACHE_MAX_MB`` when set, else a quarter of the device's total VRAM
    (0 = unknown device, caller then falls back to the entry-count cap only)."""
    try:
        env = os.environ.get("MLFRAME_FE_OPERAND_CACHE_MAX_MB")
        if env:
            return int(float(env) * (1 << 20))
        import cupy as cp

        return int(cp.cuda.runtime.memGetInfo()[1] * _OPERAND_CACHE_VRAM_FRACTION)
    except Exception as e:  # no cupy / no device / bad env value: entry-count cap only
        logger.debug("operand cache byte budget unavailable (%s: %s)", type(e).__name__, e)
        return 0


def _trim_operand_cache(cache: "OrderedDict[int, tuple]", max_entries: int) -> None:
    """Under the cache's lock: drop entries whose host array died, then evict oldest-first while over the entry or byte cap. The newest entry stays."""
    for k in [k for k, v in cache.items() if v[0]() is None]:
        del cache[k]
    budget = _operand_cache_budget_bytes()
    total = sum(int(getattr(v[1], "nbytes", 0)) for v in cache.values())
    while len(cache) > 1 and (len(cache) > max_entries or (budget > 0 and total > budget)):
        # evict-ok: a miss re-uploads the operand table
        _, (_ref, _dev) = cache.popitem(last=False)
        total -= int(getattr(_dev, "nbytes", 0))


def clear_gpu_operand_table_caches() -> None:
    """Drop every device operand-table cache (uploaded, prebuilt and the column-major transpose) so the VRAM can be reclaimed at fit teardown."""
    with _OPERAND_TABLE_CACHE_LOCK:
        _OPERAND_TABLE_CACHE.clear()
    with _PREBUILT_OPERAND_TABLE_LOCK:
        _PREBUILT_OPERAND_TABLE.clear()
    with _OPERAND_TABLE_CM_CACHE_LOCK:
        _OPERAND_TABLE_CM_CACHE["ref"] = None
        _OPERAND_TABLE_CM_CACHE["cm"] = None


def fe_gpu_resident_operands_enabled() -> bool:
    """Whether the GPU-RESIDENT operand-table build (phase 1) is active. DEFAULT ON (opt-out
    ``MLFRAME_FE_GPU_RESIDENT_OPERANDS=0``). When on (and CUDA present - the caller guards this and
    falls back on any failure) the operand table's bulk plain-unary columns are produced ON the GPU and
    the materialise consumes the device array with no H2D re-upload; the CPU / no-CUDA path is byte-for-
    byte unchanged (operand table H2D'd as before)."""
    return os.environ.get("MLFRAME_FE_GPU_RESIDENT_OPERANDS", "1").strip().lower() not in ("0", "false", "no", "off")


def register_prebuilt_operand_table(transformed_vars: np.ndarray, device_table: Any) -> None:
    """Register a GPU-RESIDENT device mirror ``device_table`` for the host operand table ``transformed_vars``
    (keyed on the host array's weakref identity). ``_resident_operand_table`` then returns ``device_table``
    for that exact host object WITHOUT re-uploading. Pass ``device_table=None`` to clear. The device array
    MUST be a row-major (n, n_operands) C-contiguous float32 cupy array matching ``transformed_vars``'s
    shape (the layout ``_fe_materialise_block_gpu``'s kernel addresses); a mismatch is ignored at lookup."""
    import weakref

    c = _PREBUILT_OPERAND_TABLE
    key = id(transformed_vars)
    with _PREBUILT_OPERAND_TABLE_LOCK:
        if device_table is None:
            c.pop(key, None)
            return
        try:
            c[key] = (weakref.ref(transformed_vars), device_table)
            c.move_to_end(key)
            _trim_operand_cache(c, _PREBUILT_OPERAND_TABLE_MAX)
        except TypeError:
            c.pop(key, None)


def _prebuilt_operand_table(transformed_vars):
    """The registered GPU-resident device mirror for ``transformed_vars`` iff it matches the host array by
    weakref identity AND shape (n, n_operands); else None. Shape guard so a stale/mismatched mirror can
    never feed the materialise kernel a wrong-width table (out-of-bounds operand-column reads)."""
    c = _PREBUILT_OPERAND_TABLE
    with _PREBUILT_OPERAND_TABLE_LOCK:
        hit = c.get(id(transformed_vars))
        if hit is None:
            return None
        ref, g = hit
    if g is None or ref() is not transformed_vars:
        return None
    if tuple(g.shape) != tuple(transformed_vars.shape):
        return None
    return g


def _resident_operand_table(cp, transformed_vars):
    """Device (n, n_operands) float32 copy of ``transformed_vars``. When a GPU-RESIDENT mirror was prebuilt
    for this exact host object (phase 1, ``register_prebuilt_operand_table``) it is returned WITH NO H2D -
    the bulk operand bytes were produced on the device. Otherwise the host array is uploaded once per
    distinct object (weakref-identity cache) and reused across a step's chunks; falls back to a plain
    upload if the array is not weakref-able."""
    import weakref

    pre = _prebuilt_operand_table(transformed_vars)
    if pre is not None:
        return pre
    c = _OPERAND_TABLE_CACHE
    key = id(transformed_vars)
    with _OPERAND_TABLE_CACHE_LOCK:
        hit = c.get(key)
        if hit is not None:
            ref, g = hit
            if ref() is transformed_vars and g is not None:
                c.move_to_end(key)
                return g
            c.pop(key, None)  # weakref dead (id recycled onto a different object) -> drop the stale entry
    # cp.asarray (H2D, may release the GIL) runs OUTSIDE the lock so a concurrent cache lookup for a
    # DIFFERENT object is never blocked on this upload; a duplicate concurrent upload for the SAME object
    # is a harmless (if wasteful) recompute, never a correctness issue - the final cache write below wins.
    g = cp.asarray(np.ascontiguousarray(transformed_vars, dtype=np.float32))
    with _OPERAND_TABLE_CACHE_LOCK:
        try:
            c[key] = (weakref.ref(transformed_vars), g)
            c.move_to_end(key)
            _trim_operand_cache(c, _OPERAND_TABLE_CACHE_MAX)
        except TypeError:
            c.pop(key, None)
    return g


from ._gpu_resident_operand_build import (  # noqa: F401  -- carved sibling, re-exported
    _BATCH_UNARY_OPS,
    _BATCH_UNARY_SRC,
    _BATCH_UNARY_KERNEL,
    _get_batch_unary_kernel,
    build_resident_operand_table,
)


def _validate_materialise_codes_dtype(dtype: Any, nbins: int) -> None:
    """Reject a code dtype too narrow for ``nbins`` before any GPU work. Pure numpy/dtype arithmetic, so it
    runs before ``import cupy``: a cupy-less host gets this ``ValueError`` rather than ``ModuleNotFoundError``."""
    # Validate BEFORE importing cupy: this check is pure numpy/dtype arithmetic, unrelated to whether a
    # GPU/cupy is even present, but used to sit after ``import cupy as cp`` -- a caller on a cupy-less host
    # got ``ModuleNotFoundError: No module named 'cupy'`` instead of this function's own intended
    # ``ValueError`` (CI: cupy is not installed on some py-version legs, breaking
    # ``test_regression_gpu_materialise_discretize_rejects_narrow_dtype_for_nbins``, which exists purely to
    # pin this validation and has nothing to do with GPU execution).
    _dt = np.dtype(dtype)
    if np.issubdtype(_dt, np.integer) and np.iinfo(_dt).max < nbins - 1:
        # a future direct caller passing a dtype too narrow for
        # nbins would otherwise silently wrap around instead of raising.
        raise ValueError(f"dtype {_dt} cannot represent codes up to nbins-1={nbins - 1} (max={np.iinfo(_dt).max})")


def gpu_materialise_discretize_codes_host(
    transformed_vars: np.ndarray,
    a_cols: np.ndarray,
    b_cols: np.ndarray,
    op_codes: np.ndarray,
    nbins: int,
    *,
    dtype: Any = np.int8,
    out_cand: np.ndarray | None = None,
) -> np.ndarray:
    """GPU fast path for the FE chunk's MATERIALISE + BINNING. Uploads the operand table
    ``transformed_vars`` (n, n_operands) float32 ONCE, then for each VRAM-bounded column block: generates
    the float32 candidate matrix on the GPU (``_fe_materialise_block_gpu`` - bit-equal to
    ``_materialise_chunk_njit``) and quantile-bins it RESIDENT (``_gpu_resident_discretize_codes``,
    bit-equal to ``discretize_2d_quantile_batch``). Returns the (n, K) ``dtype`` codes (BIT-IDENTICAL to
    the CPU njit-materialise -> ``gpu_discretize_codes_host`` pipeline, verified maxdiff 0).

    The candidate matrix is generated + binned RESIDENT (the int codes are the only mandatory D2H). But the
    downstream FE survivor / usability / ext-val stages read the CONTINUOUS candidate columns out of the
    chunk buffer, so the caller passes ``out_cand`` (the ``chunk_buffer[:, :K]`` float32 view) to receive
    the materialised float candidate matrix as well - this replaces the CPU njit materialise with the GPU
    one (the bandwidth-bound strided-gather op the GPU is good at) while keeping the buffer the rest of the
    pipeline expects. Pass ``out_cand=None`` to skip the float D2H (codes-only, when no downstream
    continuous read is needed). Inputs are finite by construction (the kernel scrubs NaN/inf inline)."""
    cp, _gpu_resident_discretize_codes, a_cols, b_cols, op_codes, K, tv_gpu, out, dev_codes, _defer_host_codes, k_chunk, _db_slot, _copy_stream = _gpu_materialise_discretize_codes_host_part1(
        dtype, nbins, transformed_vars, a_cols, b_cols, op_codes, out_cand,
    )

    _db_pending = None  # (copy_done_event, pinned_view, col_slice, cand_ref)

    def _drain_pending():
        """Block on the pending async D2H copy's event (if any) and flush it into ``out_cand``, letting the previous block's transfer overlap the current block's compute."""
        nonlocal _db_pending
        if _db_pending is not None:
            _pev, _phv, _psl, _pc = _db_pending
            _pev.synchronize()
            out_cand[:, _psl] = _phv
            _db_pending = None

    for start in range(0, K, k_chunk):
        stop = min(start + k_chunk, K)
        # return_cm: capture the materialise kernel's pre-transpose (K, n) cm buffer so the binning step's
        # radix-select edges reuse it (launch-fusion: skip the rm->cm transpose that inverts materialise's
        # cm->rm). cand_cm is None on the row-major fallback -> discretize transposes as before (bit-identical).
        cand, cand_cm = _fe_materialise_block_gpu(
            tv_gpu, a_cols[start:stop], b_cols[start:stop], op_codes[start:stop], return_cm=True
        )  # resident (n, blk) float32 - bit-equal to _materialise_chunk_njit
        if out_cand is not None:
            # Float candidate D2H for the downstream survivor/usability reads. Stage through a PERSISTENT
            # PINNED host buffer (full PCIe bandwidth) then host->host memcpy into the caller's pageable
            # slice - 1.6x faster than cp.asnumpy's pageable bounce-buffer path even WITH the added memcpy
            # (see _pinned_view note). Bit-identical bytes. ASYNC double-buffered when the copy stream is up
            # (see the loop-head note): enqueue this block's copy, drain the previous one, keep going - the
            # binning below overlaps the transfer. Falls back to the synchronous path on any fault.
            _done_async = False
            if _copy_stream is not None:
                try:
                    hv = _pinned_view(cand.nbytes, cand.shape, cand.dtype, slot=_db_slot)
                    _mat_done = cp.cuda.Event(disable_timing=True)
                    _mat_done.record()  # materialise finished on the default stream
                    _copy_stream.wait_event(_mat_done)  # copy starts only after cand is fully written
                    cand.get(out=hv, stream=_copy_stream, blocking=False)
                    _cp_done = cp.cuda.Event(disable_timing=True)
                    _cp_done.record(_copy_stream)
                    _drain_pending()  # previous block's copy -> out_cand (overlaps GPU)
                    _db_pending = (_cp_done, hv, slice(start, stop), cand)  # cand kept alive until drained
                    _db_slot ^= 1
                    _done_async = True
                except Exception:
                    log_throttle(
                        logger,
                        "gpu_materialise.async_d2h",
                        logging.WARNING,
                        "candidate block copy: the async D2H pipeline failed, using synchronous copies",
                        exc_info=True,
                    )
                    _copy_stream = None
                    _drain_pending()
            if not _done_async:
                try:
                    hv = _pinned_view(cand.nbytes, cand.shape, cand.dtype)
                    cand.get(out=hv)
                    out_cand[:, start:stop] = hv
                # best-effort: cp.asnumpy returns the same values as the pinned staging path, only slower
                except Exception:
                    logger.debug("pinned D2H staging failed; cp.asnumpy fallback", exc_info=True)
                    out_cand[:, start:stop] = cp.asnumpy(cand)
        # Bin the candidate RESIDENT at its native float32 (the FE buffer dtype) - no f64 up-cast: the
        # cand already IS float32 (bit-equal to _materialise_chunk_njit), so binning in f32 removes a needless
        # cast AND halves the bandwidth-bound percentile sort, while preserving the FE selection. The exact
        # f64 fallback (bit-identical to the CPU pipeline) is one env flip away (MLFRAME_FE_GPU_BINNING_DTYPE
        # =float64). _gpu_resident_discretize_codes applies the working dtype internally.
        # LAUNCH-FUSION: bin DIRECTLY into the target narrow ``dtype`` (int8/int16) - the binning
        # kernel writes the narrow code dtype itself (OUTTYPE-templated), so the separate int32->narrow astype
        # launch + the int32 (n,K) intermediate are gone (was: discretize int32 then astype). nbins<=128 -> the
        # codes are in [0, nbins-1] -> int8 (signed, -128..127) cannot overflow (
        # ); BIT-IDENTICAL to int32-then-astype. The D2H still
        # moves 1/4 (int8) the bytes of int32 codes. (Prior bench GTX 1050 Ti n=100k K=384: int32-D2H+host-cast
        # 170ms -> gpu-narrow+D2H 25ms = 6.7x; this fusion additionally removes the astype launch + int32 buffer.)
        _cd = np.dtype(dtype)
        # MLFRAME_FE_FUSION_AB=0 disables the f1 (narrow-emit) + f2 (cm reuse) fusions for nsys/wall A/B ONLY
        # (default fused). When off: bin int32 with an internal transpose, then astype to narrow (the pre-fusion
        # path) - BIT-IDENTICAL output, just the extra cast launch + transpose for the launch-count baseline.
        if os.environ.get("MLFRAME_FE_FUSION_AB", "1").strip() == "0":
            codes_gpu = _gpu_resident_discretize_codes(cand, int(nbins))  # int32, internal transpose
            codes_out = codes_gpu.astype(cp.dtype(_cd), copy=False) if codes_gpu.dtype != _cd else codes_gpu
        else:
            codes_gpu = _gpu_resident_discretize_codes(cand, int(nbins), out_dtype=_cd, cm_hint=cand_cm)
            codes_out = codes_gpu
        if dev_codes is not None:
            # Keep this block's narrow codes RESIDENT (the EXACT bytes we D2H below). Bit-identical to the
            # host ``out`` slice -> selection-equivalent when the resident gate consumes the device copy.
            dev_codes[:, start:stop] = codes_out
        if not _defer_host_codes:
            # Eager host fill (deferral off, or no resident copy): D2H this block's codes into ``out`` now.
            out[:, start:stop] = cp.asnumpy(codes_out)
        del cand, cand_cm, codes_gpu, codes_out
    _drain_pending()  # last block's async float copy -> out_cand
    if dev_codes is not None:
        # Stash by the returned host array's identity so the dispatch can pick the device codes up without
        # the chunk path threading a new argument (see _RESIDENT_CODES_HANDOFF). Any consumer that is NOT
        # the resident CUDA gate simply ignores it + reads ``out`` (host) as before.
        _stash_resident_codes(out, dev_codes)
    if _defer_host_codes:
        # ``out`` is UNFILLED - register the lazy device->host fill so a host-reading consumer (analytic /
        # CPU / non-resident GPU) can materialise it on demand via ensure_host_codes_filled. The eager
        # per-block D2H above was skipped; the resident gate reads the device codes directly (no host read).
        _stash_deferred_host_fill(out, dev_codes)
    _out_final: np.ndarray = out
    return _out_final


def _gpu_materialise_discretize_codes_host_part1(
    dtype: Any, nbins: int, transformed_vars: np.ndarray, a_cols: np.ndarray, b_cols: np.ndarray, op_codes: np.ndarray, out_cand: np.ndarray | None,
) -> tuple[Any, ...]:
    """Extracted block of ``gpu_materialise_discretize_codes_host`` (part 1); returns the names the caller still uses."""
    _validate_materialise_codes_dtype(dtype, nbins)
    import cupy as cp

    # This used to call clear_resident_codes_handoff with NO
    # argument - the blanket, whole-dict-clear form - which silently dropped another concurrent thread's
    # still-pending deferred-fill entry under joblib threading (each chunk's dispatch should already have
    # consumed/cleared its OWN entry; this was meant only as a dead-man's-switch, not a normal-path event).
    # Rely on the bounded-FIFO eviction (_DEFERRED_HOST_FILL_MAX) to age out any truly-abandoned entry instead.
    from ._gpu_resident_discretize import _gpu_resident_discretize_codes  # carve sibling (lazy: avoid cycle)

    tv = np.ascontiguousarray(transformed_vars, dtype=np.float32)
    a_cols = np.ascontiguousarray(a_cols, dtype=np.int64)
    b_cols = np.ascontiguousarray(b_cols, dtype=np.int64)
    op_codes = np.ascontiguousarray(op_codes, dtype=np.int8)
    n = int(tv.shape[0])
    K = int(a_cols.shape[0])
    # Operand table H2D cached per-step by weakref identity (same transformed_vars across the step's
    # chunks -> uploaded ONCE, not per chunk). Pass the ORIGINAL array so the weakref tracks it.
    tv_gpu = _resident_operand_table(cp, transformed_vars)
    out = np.empty((n, K), dtype=dtype)
    # RESIDENT-CODES HANDOFF (gated, default OFF): keep the on-device int codes in ONE (n, K) resident
    # cupy array so the noise-gate's resident-CUDA path can consume them DIRECTLY - skipping the codes'
    # GPU->host (here) ->GPU (the gate's H2D) round-trip. The host ``out`` is STILL filled (the CPU /
    # analytic / opt-out / SU / any-failure dispatch branches need it and it is the safe fallback), so this
    # only ADDS a resident copy when the gate is on; the round-trip is skipped only when the resident gate
    # is the actual consumer (it matches ``out`` by identity via the module handoff).
    _resident_codes_on = fe_gpu_resident_codes_enabled()
    dev_codes = cp.empty((n, K), dtype=cp.dtype(np.dtype(dtype))) if _resident_codes_on else None
    # DEFER the host-codes D2H when the resident handoff is on: the host ``out`` is filled LAZILY (only if a
    # host consumer reads it - see ensure_host_codes_filled) instead of eagerly per block. This skips the
    # (n, K) codes D2H (the canonical fit's single largest D2H) whenever the resident gate consumes the
    # device codes. Needs dev_codes (the resident copy) to fill from, so it is only active with it.
    _defer_host_codes = bool(dev_codes is not None and fe_gpu_defer_host_codes_enabled())
    # CODES path footprint is f32 (cand + transpose + int32 codes + narrow out), ~4B x ~4 working copies -
    # NOT the f64 MI prototype's 8x5. Budget for that so the VRAM sub-chunk is ~3x wider -> ~3x fewer
    # radix/bin/materialise launches (cuts the launch+sync+GPU-idle overhead). working_multiple=6 keeps a
    # safe margin over the honest ~4 on the 4GB card; still 0.25*free VRAM-governed; per-column-independent
    # so codes are bit-identical regardless of chunk boundary.
    k_chunk = _gpu_k_chunk(n, bytes_per_elem=4, working_multiple=6, max_cols=K)
    # ASYNC DOUBLE-BUFFERED float D2H (max-GPU phase): the float-candidate readback used a
    # SYNCHRONOUS cand.get(out=pinned) - the host blocked until materialise AND the copy finished, THEN ran
    # the 8-25 MB host memcpy, THEN launched the block's binning kernels; every block serialised
    # [materialise -> D2H -> memcpy -> bin]. Now the copy is enqueued on a dedicated non-blocking COPY STREAM
    # (after an event marking materialise completion) into one of two alternating pinned buffers, the binning
    # kernels launch IMMEDIATELY (compute overlaps the PCIe transfer), and the PREVIOUS block's finished copy
    # is drained into ``out_cand`` while THIS block's kernels run. Bit-identical bytes (same DMA, same
    # memcpy); only the schedule changes. Falls back to the synchronous path on any stream/event fault.
    _db_slot = 0
    _copy_stream = None
    if out_cand is not None:
        try:
            _copy_stream = cp.cuda.Stream(non_blocking=True)
        # best-effort: a synchronous copy gives the same bytes, only without overlapping the GPU work
        except Exception as e:
            logger.debug("non-blocking copy stream creation failed, falling back to synchronous copy: %s", e)
            _copy_stream = None
    return cp, _gpu_resident_discretize_codes, a_cols, b_cols, op_codes, K, tv_gpu, out, dev_codes, _defer_host_codes, k_chunk, _db_slot, _copy_stream


def gpu_discretize_codes_host(cand: np.ndarray, nbins: int, *, dtype: Any = np.int8, defer_host_fill: bool = False) -> np.ndarray:
    """Quantile-bin a host (n, K) float candidate matrix to ordinal codes via the GPU, returning a host
    ``(n, K)`` array of ``dtype``. The FE candidate buffer is ALREADY float32, so the matrix is kept at
    its native dtype (NO f64 up-cast) and binned in float32 (the input's native dtype) - removing a
    needless cast AND halving the bandwidth-bound cp.percentile sort, while preserving the FE selection
    (the acceptance bar; f32-vs-f64 codes agree ~100%). Set ``MLFRAME_FE_GPU_BINNING_DTYPE=double precision`` for
    the bit-identical fallback matching the CPU ``discretize_2d_quantile_batch`` (np.percentile upcasts
    float32 to double precision). Feeding the result into the UNCHANGED ``_dispatch_batch_mi_with_noise_gate``
    keeps the FE selection equivalent - this only moves the binning (CPU partition+searchsorted, the
    dominant per-pair cost at large n) onto the GPU. Inputs are assumed finite (caller scrubs NaN/inf).

    VRAM-chunked over columns so a wide candidate block never over-allocates device memory."""
    # Validate BEFORE importing cupy: this check is pure numpy/dtype arithmetic, unrelated to whether a
    # GPU/cupy is even present, but used to sit after ``import cupy as cp`` -- a caller on a cupy-less host
    # got ``ModuleNotFoundError: No module named 'cupy'`` instead of this function's own intended
    # ``ValueError`` (CI: cupy is not installed on some py-version legs, breaking
    # ``test_regression_gpu_materialise_discretize_rejects_narrow_dtype_for_nbins``, which exists purely to
    # pin this validation and has nothing to do with GPU execution).
    _dt = np.dtype(dtype)
    if np.issubdtype(_dt, np.integer) and np.iinfo(_dt).max < nbins - 1:
        # a future direct caller passing a dtype too narrow for
        # nbins would otherwise silently wrap around instead of raising.
        raise ValueError(f"dtype {_dt} cannot represent codes up to nbins-1={nbins - 1} (max={np.iinfo(_dt).max})")
    import cupy as cp

    from ._gpu_resident_discretize import _gpu_resident_discretize_codes  # carve sibling (lazy: avoid cycle)

    cand = np.ascontiguousarray(cand)  # keep native dtype (float32 FE buffer) - no f64 up-cast
    n, K = cand.shape
    out = np.empty((n, K), dtype=dtype)
    # Removed the unconditional clear_resident_codes_handoff
    # blanket call here - see the matching note in gpu_materialise_discretize_codes_host above.
    # RESIDENT-CODES HANDOFF (gated, default ON when CUDA present): this is the SECOND codes leg - the
    # binning-only path the canonical FE chunk takes when the candidate buffer is materialised on the CPU
    # (the default minimal preset's numpy-fallback materialise) then binned on the GPU. It produces the
    # SAME on-device int codes as the fused materialise path, so keep them RESIDENT (one (n, K) cupy array
    # in the narrow code dtype) and stash them by the returned host array's identity - the noise-gate
    # dispatch then consumes the device codes IN PLACE, skipping the codes' GPU->host (here) ->GPU (the
    # gate's H2D) round-trip. The host ``out`` is STILL filled (the CPU / analytic / opt-out / any-failure
    # branches read it, and it is the safe fallback), so this only ADDS a resident copy when the gate is on;
    # the round-trip is skipped only when the resident CUDA gate is the actual consumer (it matches ``out``
    # by identity). Bit-identical to the host codes -> selection unchanged.
    _resident_codes_on = fe_gpu_resident_codes_enabled()
    dev_codes = cp.empty((n, K), dtype=cp.dtype(np.dtype(dtype))) if _resident_codes_on else None
    # HOST-CODES D2H DEFERRAL. Direct callers (gpu_pairs_fe_mi's analytic path + the bit-identity
    # tests) read the returned host array IMMEDIATELY, so this leg's DEFAULT stays eager (defer_host_fill=False)
    # to keep that contract. But the per-pair FE-score leg (_pairs_score._score_one_pair) hands the return
    # straight to ``_dispatch_batch_mi_with_noise_gate``, whose resident-CUDA gate consumes the DEVICE codes in
    # place (via take_resident_codes) and never reads host ``disc_2d`` - so under the strict-resident path the
    # eager (n, K) codes D2H here is the single largest D2H of the fit and is PURE WASTE (measured n=100k: 24/24
    # dispatches hit the resident handoff; the host buffer was filled but never read). When that caller opts in
    # (defer_host_fill=True) AND the deferral is enabled, return an UNFILLED host buffer + register a lazy
    # device->host fill (ensure_host_codes_filled) keyed on the buffer id, exactly like the fused
    # gpu_materialise_discretize_codes_host leg. Bit-identical: the host buffer, if any consumer ever reads it,
    # is device_codes.get() - the exact bytes the eager D2H produced -> FE selection unchanged.
    # f32 codes-path footprint (see gpu_materialise_discretize_codes_host) -> wider VRAM sub-chunk, ~3x
    # fewer bin/edge launches; per-column-independent -> bit-identical codes.
    _defer_host_codes = bool(defer_host_fill and dev_codes is not None and fe_gpu_defer_host_codes_enabled())
    k_chunk = _gpu_k_chunk(n, bytes_per_elem=4, working_multiple=6, max_cols=K)
    for start in range(0, K, k_chunk):
        block = cand[:, start : start + k_chunk]
        stop = start + block.shape[1]
        codes_gpu = _gpu_resident_discretize_codes(cp.asarray(block), int(nbins))
        # Narrow int32->dtype ON the GPU before D2H (1/4 the bytes for int8, no host astype copy) -
        # same 6.7x codes-export win as gpu_materialise_discretize_codes_host, BIT-IDENTICAL.
        _cd = np.dtype(dtype)
        codes_out = codes_gpu.astype(cp.dtype(_cd), copy=False) if codes_gpu.dtype != _cd else codes_gpu
        if dev_codes is not None:
            # Keep this block's narrow codes RESIDENT (the EXACT bytes we D2H below) for the gate consumer.
            dev_codes[:, start:stop] = codes_out
        if not _defer_host_codes:
            out[:, start:stop] = cp.asnumpy(codes_out)
        del codes_gpu, codes_out
    if dev_codes is not None:
        _stash_resident_codes(out, dev_codes)
    if _defer_host_codes:
        # ``out`` is UNFILLED - register the lazy device->host fill so a host-reading consumer (analytic /
        # CPU / non-resident GPU) can materialise it on demand via ensure_host_codes_filled. The eager
        # per-block D2H above was skipped; the resident gate reads the device codes directly (no host read).
        _stash_deferred_host_fill(out, dev_codes)
    return out
