"""GPU-resident FE: residency-buffer + radix-select order-statistic API (Tier E carve).

Carved VERBATIM out of ``_gpu_resident_fe.py`` (sibling re-export pattern) to bring the parent under the
1k-LOC ceiling. Holds the top-level order-statistic API (``_radix_select_interior_edges`` /
``_radix_quantiles``) that dispatches to the rank-EXACT sort-free radix-select RawKernels. The kernel
sources, their lazy-compiled singletons, the coalesced-tiled-transpose kernels, and the per-host KTC
dispatch helpers were carved further into the sibling ``_gpu_resident_select_kernels.py`` (LOC budget) and
are re-exported below so every existing ``from ._gpu_resident_select import X`` name still resolves.

The gate helpers (``_cuda_present`` / ``_env_gpu_default_on`` / ``fe_gpu_*_enabled``) and the candidate-
grid primitives stay in the PARENT and are imported below; the parent re-exports every public/used name
moved here so all ``from .._gpu_resident_fe import X`` paths still resolve byte-for-byte. The few
cross-sibling references (``_gpu_apply_prewarp`` in ``_gpu_resident_basis``) are LAZY-imported inside the
function bodies to avoid an import cycle. No kernel-source, dispatch-threshold, residency, or selection
behavior changed.
"""
from __future__ import annotations

import logging

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# Fraction of the free VRAM a scratch buffer may take before the leaner kernel is used instead.
_SCRATCH_FREE_FRACTION = 0.5


def _free_vram_bytes() -> int | None:
    """Free device memory in bytes, or None when it cannot be read."""
    try:
        import cupy as cp

        return int(cp.cuda.Device().mem_info[0])
    except Exception as exc:  # best-effort probe: an unreadable device means "unknown", and the caller then tries the allocation
        log_throttle(logger, "radix_free_vram_probe_failed", logging.DEBUG, "free-VRAM probe failed (%s: %s); the scratch allocation is tried without a budget check.", type(exc).__name__, exc)
        return None


def _scratch_fits(need_bytes: int) -> bool:
    """Whether a scratch buffer of ``need_bytes`` stays within ``_SCRATCH_FREE_FRACTION`` of the free VRAM (True when the free size is unknown)."""
    free = _free_vram_bytes()
    return free is None or need_bytes <= free * _SCRATCH_FREE_FRACTION

from ._gpu_resident_select_kernels import (
    _RADIX_INTERP_CACHE,  # noqa: F401 -- tests clear this cache through the carve-out module's namespace
    _RADIX_SELECT_MAXR,
    _RADIX_STATIC_SHARED_BYTES,
    _get_radix_interp_kernel,
    radix_interp_get_or_build,
    _get_radix_select_f32_dispatch,
    _get_radix_select_interp_f64_kernel,
    _get_radix_select_interp_f64_v3_kernel,
    _get_radix_select_kernel,
    _resolve_radix_threads,
    _transpose_to_cm,
)
from ._gpu_resident_select_kernels import _transpose_cm_to_rm  # noqa: F401 - re-export only, see module docstring
from ._gpu_resident_select_kernels import fe_gpu_radix_edges_enabled  # noqa: F401 - re-export only, see module docstring
from ._gpu_resident_select_kernels import transpose_codes_to_cm  # noqa: F401 - re-export only, see module docstring


def _radix_cm_view(cand_gpu, cm_hint, data_is_cm: bool, n: int, K: int):
    """The (K, n) C-order column-major view the radix kernels read, chosen by the first applicable source.

    The order is the same as in the caller before this was split out: the block itself, then the
    materialise kernel's pre-transpose buffer, then the STRICT-f64 upcast of the f32 hint, then a transpose.
    """
    import cupy as cp

    if data_is_cm:
        data_cm = cand_gpu if cand_gpu.flags.c_contiguous else cp.ascontiguousarray(cand_gpu)
    elif cm_hint is not None and cm_hint.shape == (K, n) and cm_hint.flags.c_contiguous and cm_hint.dtype == cand_gpu.dtype:
        data_cm = cm_hint
    # Intentional float64: dtype identity check on the candidate matrix, not a compute dtype
    elif cm_hint is not None and cm_hint.shape == (K, n) and cm_hint.flags.c_contiguous and cm_hint.dtype == cp.float32 and cand_gpu.dtype == cp.float64:
        # STRICT-f64 fusion recovery (nsys-driven): under MLFRAME_FE_GPU_BINNING_DTYPE=float64 the
        # discretize sibling upcasts the f32 materialise output to f64 (cand_gpu.astype(f64)) BEFORE calling in,
        # so the f32 cm_hint's dtype no longer equals cand_gpu's and the launch-fusion above was SILENTLY
        # defeated -> a full transpose_f64 of the (n,K) f64 cand ran EVERY block (nsys F2 1M STRICT: transpose_f64
        # = 8.04% of GPU time, 242 calls / 1.20s). But cand_gpu here == cm_hint.astype(f64) EXACTLY: cand is the
        # elementwise f32->f64 upcast of the SAME materialise output the (K,n) f32 hint holds, and f32->f64 is
        # LOSSLESS and COMMUTES with transpose (upcast(transpose(x)) == transpose(upcast(x))). So upcast the
        # already-computed hint (12 B/elem, coalesced elementwise) instead of re-transposing the f64 cand
        # (16 B/elem tiled transpose) -> BIT-IDENTICAL data_cm (same order statistics -> same edges -> same
        # codes), restoring the intended fusion and dropping the transpose_f64 kernel on this path.
        # Intentional float64: bit-identical to cp.percentile's linear interpolation, which computes in float64
        data_cm = cm_hint.astype(cp.float64)
    else:
        data_cm = _transpose_to_cm(cand_gpu)  # (K, n) C-order = column-major (coalesced tiled-transpose kernel)
    return data_cm


def _launch_radix_f32_v4(cp, data_cm, n: int, K: int, ranks_g, R: int, osv, threads: int, shmem: int) -> bool:
    """Launch the compaction v4 f32 select into ``osv``; False (nothing launched) when it is off, did not compile or its scratch does not fit the VRAM budget."""
    from ._gpu_resident_select_f32_compact import _get_radix_select_f32_v4_kernel

    kern = _get_radix_select_f32_v4_kernel()
    if kern is None:
        return False
    cap = max(1024, int(n) // 4)
    if not _scratch_fits(K * cap * 4):
        return False
    try:
        scratch = cp.empty((K, cap), dtype=cp.uint32)
        kern((K,), (threads,), (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R), osv, scratch, np.int64(cap)), shared_mem=shmem)
        return True
    except Exception as exc:
        log_throttle(logger, "radix_f32_v4_launch_failed", logging.WARNING, "radix select: the f32 v4 compaction launch failed (%s: %s); using v3.", type(exc).__name__, exc)
        return False


_CANONICAL_NAN_KERNELS: dict = {}


def _canonical_nan(cand_gpu):
    """``cand_gpu`` with every NaN rewritten, in place, to the positive quiet NaN, so the radix key transform puts it last as sorting does.

    One elementwise pass over the matrix and no second matrix: the first version built ``cp.where(isnan(x), nan, x)``, a full copy of the candidate block on the device, which
    went over the memory pool's limit when the FE stages run together (a 406 MB block next to the 865 MB already held, limit 1.06 GB) and sent the edge computation to the
    CPU path. Only the payload of a NaN changes, never a value, so a caller that holds the same array sees the same numbers.
    """
    import cupy as cp

    if cand_gpu.dtype.kind != "f":
        return cand_gpu
    key = cand_gpu.dtype.name
    kern = _CANONICAL_NAN_KERNELS.get(key)
    if kern is None:
        quiet = {"float32": "__int_as_float(0x7fc00000)", "float64": "__longlong_as_double(0x7ff8000000000000LL)", "float16": "x"}.get(key)
        if quiet is None or key == "float16":
            return cand_gpu
        kern = cp.ElementwiseKernel(f"{key} x", f"{key} y", f"y = isnan(x) ? {quiet} : x", f"canonical_nan_{key}")
        _CANONICAL_NAN_KERNELS[key] = kern
    kern(cand_gpu, cand_gpu)
    return cand_gpu


def _radix_select_interior_edges(cand_gpu, nbins: int, cm_hint=None, with_extremes: bool = False, data_is_cm: bool = False):
    """Return the (nbins-1, K) INTERIOR quantile edges of the resident (n, K) cupy ``cand_gpu`` via the
    sort-free radix-select kernel + cupy's exact 'linear' interpolation (reproduced in double). The edges
    are BIT-IDENTICAL (in the resulting codes) to ``cp.percentile(cand, linspace(0,100,nbins+1))[1:-1]``.
    Returns ``None`` if the radix path is inapplicable (R over the kernel cap, shared-mem over the device
    limit) so the caller uses the cp.percentile fallback. ``cand_gpu`` must be C-contiguous (n, K).

    ``with_extremes`` (ncu/nsys-driven, 2026-07-15): also emit the column min/max as extra EXACT order
    statistics (rank 0 and rank n-1, interp weight 0 -> the interp formula degenerates to the order stat
    itself, no special-casing needed) computed in the SAME select pass, returning the FULL (nbins+1, K)
    edge matrix (min, interior..., max) instead of interior-only. This lets ``batched_quantile_bin_gpu``
    skip its separate ``Xd.min(axis=0)``/``Xd.max(axis=0)`` reduction kernels (nsys: ~18%% of GPU time in the
    cupy-search microbench) for the price of +2 order statistics in an already-shared-mem-bounded kernel -
    a no-op when nbins-1 interior ranks already include ranks near 0/n-1 (union dedups), and at worst +2
    slots against the R<=64 cap. Cached under a SEPARATE key from the interior-only path - the two modes
    never share or corrupt each other's cache entries.

    ``cm_hint`` (LAUNCH-FUSION 2026-06-27): an OPTIONAL (K, n) C-order column-major view of the SAME data as
    ``cand_gpu`` (e.g. the materialise kernel's pre-transpose cm buffer). When supplied, the internal
    ``_transpose_to_cm(cand_gpu)`` is SKIPPED - the chunk path already produced this exact buffer (the
    materialise kernel wrote cm then transposed it to the rm ``cand_gpu``), so the round-trip transpose pair
    (rm->cm here + cm->rm in materialise) collapses to ONE. The order statistics read the same values ->
    BIT-IDENTICAL edges. Validated shape (K, n) == (cand_gpu.shape[1], cand_gpu.shape[0]); any mismatch
    ignores the hint and transposes (safe).

    ``data_is_cm``: ``cand_gpu`` is itself the ``(K, n)`` C-order column-major block (what ``fused_gen_cm`` writes), so there is nothing to transpose and no copy is made."""
    cand_gpu = _canonical_nan(cand_gpu)  # a sign-bit NaN keys below -inf in the radix transform; sorting puts every NaN last
    import cupy as cp

    if data_is_cm:
        K, n = cand_gpu.shape
    else:
        n, K = cand_gpu.shape
    is_f32 = cand_gpu.dtype == cp.float32
    # ALL of the order-statistic geometry (ranks, R, shared-mem gate, the cupy 'linear' interp gathers bi/ai/w,
    # the device rank vector ranks_g) depends ONLY on (n, nbins) - NOT the candidate data - so compute it ONCE
    # per (n, nbins) and cache it. Was recomputed on EVERY per-candidate call (~11.6k x on the FE pair scan): the
    # np.unique host sort (cProfile: 12,149 calls / 4.05s = the top host-CPU cost on this path), the
    # linspace/floor, the per-call device-attribute lookup, AND the ranks H2D - all data-independent. A cached
    # None records the "radix inapplicable" verdict (R over the kernel cap / shared-mem over the device limit) so
    # the caller's cp.percentile fallback is taken without recomputing. Selection-IDENTICAL: same ranks -> same
    # order statistics -> bit-identical edges.
    _ik = (int(n), int(nbins), bool(with_extremes))
    def _build_interp_entry():
        """Order-statistic geometry + device gather vectors for (n, nbins), or None when the radix path does not apply."""
        # cupy 'linear' positions for the nbins-1 interior quantiles (q in (0,1)), float64 throughout.
        qfr = np.linspace(0.0, 100.0, int(nbins) + 1)[1:-1] / 100.0  # (nbins-1,) fractions
        idx = qfr * (n - 1)
        bel = np.floor(idx).astype(np.int64)
        abv = np.minimum(bel + 1, n - 1)
        if with_extremes:
            # exact order stats: rank 0 (min) and rank n-1 (max), interp weight 0 (bel==abv==that rank).
            idx = np.concatenate([[0.0], idx, [float(n - 1)]])
            bel = np.concatenate([[0], bel, [n - 1]]).astype(np.int64)
            abv = np.concatenate([[0], abv, [n - 1]]).astype(np.int64)
        uniq = np.unique(np.concatenate([bel, abv]))  # the order-statistic ranks to extract
        R = int(uniq.size)
        # shared-mem budget: R*256 uint32 histogram (host gate vs the device's per-block shared limit). The gate
        # must reserve the kernels' STATIC __shared__ footprint too - dynamic ``shmem`` ALONE near the limit, plus
        # the static arrays, overflows the per-block budget and cuLaunchKernel raises CUDA_ERROR_INVALID_VALUE.
        shmem = R * 256 * 4
        try:
            dev = cp.cuda.Device()
            sh_limit = int(dev.attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
        except Exception as e:
            logger.warning("querying MaxSharedMemoryPerBlock failed, using the 48KiB default: %s", e)
            sh_limit = 48 * 1024
        if R > _RADIX_SELECT_MAXR or shmem > sh_limit - _RADIX_STATIC_SHARED_BYTES:
            return None  # radix path inapplicable for this (n, nbins)
        else:
            # cupy 'linear' interp gather indices + weight (bi/ai/w) - needed BEFORE the kernel for the fused f64
            # path, which interpolates in-kernel.
            ranks_g = cp.asarray(uniq, dtype=cp.int64)
            pos = {int(r): i for i, r in enumerate(uniq)}
            bi = cp.asarray(np.asarray([pos[int(b)] for b in bel], dtype=np.int64))
            ai = cp.asarray(np.asarray([pos[int(a)] for a in abv], dtype=np.int64))
            w = cp.asarray(np.ascontiguousarray(idx - bel))  # double weight_above = idx - floor(idx)
            return (bi, ai, w, ranks_g, int(R), int(shmem))

    _ic = radix_interp_get_or_build(_ik, _build_interp_entry)
    if _ic is None:
        return None  # cached: radix inapplicable -> cp.percentile
    bi, ai, w, ranks_g, R, shmem = _ic
    # COLUMN-MAJOR input (nvprof-driven): one block/column previously read data[i*K+col] from
    # the (n,K) row-major buffer -> stride-K, gld_efficiency 12.5% (1/8 coalesced) on the dominant n-loop
    # (4 byte-passes x n reads). Transpose to (K,n) C-order so consecutive threads read consecutive memory
    # (data[col*n+i]) - one transpose pass buys ~8x coalescing across the 4 passes. Values unchanged ->
    # bit-identical order statistics. (The bin_codes step still uses the original (n,K) cand_gpu.)
    # Reuse the materialise kernel's pre-transpose (K, n) cm buffer when handed in (launch-fusion: skip the
    # rm->cm transpose that exactly inverts materialise's cm->rm). Validate shape/contiguity/dtype; else transpose.
    data_cm = _radix_cm_view(cand_gpu, cm_hint, data_is_cm, n, K)
    threads = _resolve_radix_threads(n)  # Lever B: per-host KTC-tuned block size (bit-identical edges)
    ne_rows = int(bi.shape[0])
    if not is_f32:
        # FUSED select+interp (launch-reduction): the f64 radix select keeps its R order statistics in shared
        # memory and emits the (ne, K) interior edges directly - ONE launch, no osv global, no separate
        # radix_interp launch. Bit-identical to the two-kernel f64 path (same select body + same f64 interp).
        try:
            # Intentional float64: bit-identical to cp.percentile's linear interpolation, which computes in float64
            edges = cp.empty((ne_rows, K), dtype=cp.float64)
            kern3 = _get_radix_select_interp_f64_v3_kernel()
            if kern3 is not None:
                try:
                    cap = max(1024, int(n) // 4)
                    if not _scratch_fits(K * cap * 8):
                        log_throttle(
                            logger, "radix_v3_scratch_skipped", logging.INFO,
                            "radix select: the compaction scratch (%.2f GiB for n=%d, K=%d) would take over %d%% of the free VRAM; using the v2 kernel.",
                            K * cap * 8 / (1 << 30), int(n), int(K), int(_SCRATCH_FREE_FRACTION * 100),
                        )
                        raise MemoryError("scratch over budget")
                    cand = cp.empty((K, cap), dtype=cp.uint64)
                    kern3((K,), (threads,),
                        (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R),
                         bi, ai, cp.ascontiguousarray(w), np.int32(ne_rows),
                         cand, np.int64(cap), edges), shared_mem=shmem)
                    return edges          # (nbins-1, K) double
                except MemoryError:
                    pass  # over budget: already logged above, v2 follows
                except Exception as exc:
                    log_throttle(
                        logger, "radix_v3_launch_failed", logging.WARNING,
                        "radix select: the v3 compaction launch failed (%s: %s) for n=%d, K=%d; using the v2 kernel.", type(exc).__name__, exc, int(n), int(K),
                    )
            _get_radix_select_interp_f64_kernel()((K,), (threads,),
                (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R),
                 bi, ai, cp.ascontiguousarray(w), np.int32(ne_rows), edges), shared_mem=shmem)
            return edges                  # (nbins-1, K) double
        except Exception as exc:
            log_throttle(
                logger, "radix_fused_interp_failed", logging.WARNING,
                "radix select: the fused f64 select+interp failed (%s: %s) for n=%d, K=%d; using the two-kernel path.", type(exc).__name__, exc, int(n), int(K),
            )
    osv = cp.empty((R, K), dtype=cand_gpu.dtype)
    # Lever C: dispatch the f32 path to the binary-search window-match variant where the per-host KTC selects
    # it (bit-identical order statistics, less warp divergence; base linear-scan = fallback). f64 unchanged.
    if not (is_f32 and _launch_radix_f32_v4(cp, data_cm, n, K, ranks_g, R, osv, threads, shmem)):
        ker = _get_radix_select_f32_dispatch(n) if is_f32 else _get_radix_select_kernel(is_f32)
        ker((K,), (threads,), (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R), osv), shared_mem=shmem)
    # FUSED interpolation (launch-reduction): the two fancy-index gathers + diff + cp.where
    # linear-interp collapse into ONE RawKernel that reads the two order-statistic rows from ``osv`` and writes
    # the (nbins-1, K) interior edges directly. Bit-identical to the cupy 'linear' interp (same f64 formula).
    # Intentional float64: bit-identical to cp.percentile's linear interpolation, which computes in float64
    osv64 = osv if osv.dtype == cp.float64 else osv.astype(cp.float64)
    # Intentional float64: bit-identical to cp.percentile's linear interpolation, which computes in float64
    edges = cp.empty((ne_rows, K), dtype=cp.float64)
    _ker_interp = _get_radix_interp_kernel()
    _total = ne_rows * K
    _threads = block_size_for(_ker_interp, work_items=_total)
    _ker_interp(((_total + _threads - 1) // _threads,), (_threads,), (osv64, bi, ai, cp.ascontiguousarray(w), np.int32(K), np.int32(ne_rows), edges))
    return edges  # (nbins-1, K) double


def _radix_quantiles(cand_gpu, q_fracs):
    """Per-column quantiles at ``q_fracs`` (fractions in [0,1]) over the resident (n, K) cupy ``cand_gpu``
    via the SAME sort-free radix-select kernel + cupy 'linear' interpolation as ``_radix_select_interior_edges``.
    Returns ``(len(q_fracs), K)`` double, BIT-IDENTICAL to ``cp.percentile(cand, [q*100 ...], axis=0)``
    (and to ``cp.median`` at q=0.5 - even-n linear interp == mean of the two middle order statistics).
    Returns ``None`` when the radix path is inapplicable (R over the kernel cap / shared-mem over the device
    limit) so the caller takes the cp.percentile fallback. ``cand_gpu`` must be a finite (n, K) cupy array.

    This generalises ``_radix_select_interior_edges`` (which is locked to ``linspace(0,100,nbins+1)`` binning
    edges) to ARBITRARY quantiles so the robust-scale / heavy-tail path (median, MAD-median, q25/q75) reuses
    the radix machinery instead of cp.median/cp.percentile's full per-call sort - one select+interp launch
    pair per quantile-set vs the sort's ~6-8."""
    cand_gpu = _canonical_nan(cand_gpu)  # a sign-bit NaN keys below -inf in the radix transform; sorting puts every NaN last
    import cupy as cp

    cand_gpu = cp.ascontiguousarray(cand_gpu)
    n, K = cand_gpu.shape
    is_f32 = cand_gpu.dtype == cp.float32
    # Intentional float64: bit-identical to cp.percentile's linear interpolation, which computes in float64
    qfr = np.asarray(q_fracs, dtype=np.float64)
    idx = qfr * (n - 1)
    bel = np.floor(idx).astype(np.int64)
    abv = np.minimum(bel + 1, n - 1)
    uniq = np.unique(np.concatenate([bel, abv]))  # order-statistic ranks to extract
    R = int(uniq.size)
    if R > _RADIX_SELECT_MAXR:
        return None
    shmem = R * 256 * 4
    try:
        dev = cp.cuda.Device()
        sh_limit = int(dev.attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
    except Exception as e:
        logger.warning("querying MaxSharedMemoryPerBlock failed, using the 48KiB default: %s", e)
        sh_limit = 48 * 1024
    # Reserve the kernels' STATIC __shared__ footprint (see _RADIX_STATIC_SHARED_BYTES): dynamic + static must
    # fit the per-block limit, else cuLaunchKernel raises CUDA_ERROR_INVALID_VALUE.
    if shmem > sh_limit - _RADIX_STATIC_SHARED_BYTES:
        return None
    ranks_g = cp.asarray(uniq, dtype=cp.int64)
    data_cm = _transpose_to_cm(cand_gpu)  # (K, n) coalesced (same as edges path)
    threads = _resolve_radix_threads(n)
    pos = {int(r): i for i, r in enumerate(uniq)}
    bi = cp.asarray(np.asarray([pos[int(b)] for b in bel], dtype=np.int64))
    ai = cp.asarray(np.asarray([pos[int(a)] for a in abv], dtype=np.int64))
    w = cp.ascontiguousarray(cp.asarray(idx - bel))  # double weight_above
    nq = int(qfr.size)
    # Intentional float64: bit-identical to cp.percentile's linear interpolation, which computes in float64
    out = cp.empty((nq, K), dtype=cp.float64)
    if not is_f32:
        # FUSED select+interp (launch-reduction): the f64 radix select keeps its order statistics in shared
        # memory and emits the nq interior quantiles directly - ONE launch, no osv global, no separate interp.
        try:
            _get_radix_select_interp_f64_kernel()(
                (K,), (threads,), (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R), bi, ai, w, np.int32(nq), out), shared_mem=shmem
            )
            return out
        except Exception as exc:
            log_throttle(
                logger, "radix_quantiles_fused_failed", logging.WARNING,
                "radix quantiles: the fused f64 kernel failed (%s: %s) for n=%d, K=%d; using the two-kernel path.", type(exc).__name__, exc, int(n), int(K),
            )
    osv = cp.empty((R, K), dtype=cand_gpu.dtype)
    ker = _get_radix_select_f32_dispatch(n) if is_f32 else _get_radix_select_kernel(is_f32)
    ker((K,), (threads,), (data_cm, np.int64(n), np.int32(K), ranks_g, np.int32(R), osv), shared_mem=shmem)
    # Intentional float64: bit-identical to cp.percentile's linear interpolation, which computes in float64
    osv64 = osv if osv.dtype == cp.float64 else osv.astype(cp.float64)
    _ker = _get_radix_interp_kernel()
    total = nq * K
    t = block_size_for(_ker, work_items=total)
    _ker(((total + t - 1) // t,), (t,), (osv64, bi, ai, w, np.int32(K), np.int32(nq), out))
    return out  # (len(q_fracs), K) double


def radix_order_statistics(cand_gpu, ranks):
    """``(len(ranks), K)`` order statistics (0-based ranks, values in the input dtype) of every column of the finite ``(n, K)`` cupy matrix: the values a column
    sort puts at those ranks, from the sort-free radix select. ``None`` when the select does not apply (too many ranks for the shared histograms, another dtype)."""
    cand_gpu = _canonical_nan(cand_gpu)  # a sign-bit NaN keys below -inf in the radix transform; sorting puts every NaN last
    import cupy as cp

    from ._gpu_resident_select_kernels import _get_radix_select_f32_dispatch, _get_radix_select_kernel, _resolve_radix_threads

    if cand_gpu.ndim != 2 or cand_gpu.dtype not in (cp.float32, cp.float64):
        return None
    n, K = int(cand_gpu.shape[0]), int(cand_gpu.shape[1])
    uniq = np.unique(np.asarray(ranks, dtype=np.int64))
    R = int(uniq.size)
    if n < 1 or K < 1 or R < 1 or int(uniq[0]) < 0 or int(uniq[-1]) >= n or R > _RADIX_SELECT_MAXR:
        return None
    shmem = R * 256 * 4
    if shmem > int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024)) - _RADIX_STATIC_SHARED_BYTES:
        return None
    data_cm = _transpose_to_cm(cp.ascontiguousarray(cand_gpu))
    osv = cp.empty((R, K), dtype=cand_gpu.dtype)
    ker = _get_radix_select_f32_dispatch(n) if cand_gpu.dtype == cp.float32 else _get_radix_select_kernel(False)
    ker((K,), (_resolve_radix_threads(n),), (data_cm, np.int64(n), np.int32(K), cp.asarray(uniq, dtype=cp.int64), np.int32(R), osv), shared_mem=shmem)
    row = {int(r): i for i, r in enumerate(uniq)}
    return osv[cp.asarray([row[int(r)] for r in np.asarray(ranks, dtype=np.int64)], dtype=cp.int64)]


def radix_column_medians(cand_gpu):
    """``cp.median`` of every column of the finite ``(n, K)`` matrix, in its dtype: the two middle order statistics from the radix select and their mean as
    ``cp.median`` forms it (``(a + b) / 2`` in the input dtype). ``None`` when the select does not apply."""
    cand_gpu = _canonical_nan(cand_gpu)  # a sign-bit NaN keys below -inf in the radix transform; sorting puts every NaN last
    n = int(cand_gpu.shape[0])
    st = radix_order_statistics(cand_gpu, [(n - 1) // 2, n // 2])
    if st is None:
        return None
    return (st[0] + st[1]) / cand_gpu.dtype.type(2)


def radix_quantiles_grouped(cand_gpu, q_fracs, group: int = 16):
    """:func:`_radix_quantiles` for more fractions than one radix launch holds: the fractions in groups of ``group`` (each order statistic is selected
    independently, so the stacked groups equal one launch over all of them, and ``cp.percentile``). ``None`` when a group is inapplicable or ``MLFRAME_RADIX_GROUPED=0``."""
    cand_gpu = _canonical_nan(cand_gpu)  # a sign-bit NaN keys below -inf in the radix transform; sorting puts every NaN last
    import os

    import cupy as cp

    if os.environ.get("MLFRAME_RADIX_GROUPED", "1").strip() == "0":
        return None
    fr = np.asarray(q_fracs, dtype=np.float64)  # Intentional float64: quantile fractions, as cp.percentile forms them
    if fr.size == 0:
        return None
    parts = []
    for s in range(0, int(fr.size), int(group)):
        part = _radix_quantiles(cand_gpu, fr[s : s + int(group)])
        if part is None:
            return None
        parts.append(part)
    return parts[0] if len(parts) == 1 else cp.concatenate(parts, axis=0)


# --- Carve re-exports (sibling pattern): the kernel-source / lazy-getter / KTC-dispatch block ->
# _gpu_resident_select_kernels.py, the fused-binning / resident-discretize block ->
# _gpu_resident_discretize.py, the materialise / operand-table / host-fast-path block ->
# _gpu_resident_materialise.py (all carved VERBATIM under the 1k ceiling). Rebind EVERY moved name
# (public AND underscore-private) into THIS namespace so every existing ``from ._gpu_resident_select import X``
# and ``_gpu_resident_select.X`` path still resolves byte-for-byte. At the BOTTOM so the siblings' top-level
# back-imports (from ._gpu_resident_fe import ...) resolve during the partial-init import chain.
from . import _gpu_resident_select_kernels as _grsk
from . import _gpu_resident_discretize as _grd
from . import _gpu_resident_materialise as _grm
for _m in (_grsk, _grd, _grm):
    for _n in dir(_m):
        if not _n.startswith("__") and _n not in globals():
            globals()[_n] = getattr(_m, _n)
del _m, _n
