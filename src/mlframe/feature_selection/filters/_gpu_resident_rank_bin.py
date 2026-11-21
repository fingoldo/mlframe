"""GPU-resident RANK (argsort equi-frequency) binner + rank-binned plug-in MI for the CONDITIONAL-GATE MI
Path under STRICT residency.

WHY THIS EXISTS
---------------
The STRICT-residency MI core ``_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident`` bins by PERCENTILE
EDGES (the radix-edge fast path - 71e31818 / 6b7370b4). Edge and the CPU catalog's RANK binning
(``hermite_fe._quantile_bin_njit`` argsort equi-frequency) agree bit-for-bit ONLY on tie-free columns; on
heavily-tied columns they diverge (edge lumps all tied values into one bin; rank splits the ties across bin
boundaries). The conditional-gate operator ``gate_mask`` emits ``1[c>0]*a`` - ~50% EXACT zeros - so under
STRICT the gate MI was edge-binned and did NOT byte-match the CPU rank MI the gate scoring uses (commit
89dd47c7 pinned the biz-value test to the rank estimator as a stopgap).

This module provides a cupy ``argsort``-based RANK binner that reproduces ``_quantile_bin_njit``'s recipe
(``base = n // n_bins`` rows per bin, first ``n % n_bins`` bins get one extra), and a rank-binned plug-in MI
that the GATE opts into so its STRICT MI matches the CPU rank MI.

SCOPE - GATE MI ONLY
---------------------
This is wired ONLY into the gate MI (``_orth_mi_backends._mi_classif_batch(..., rank_binning=True)`` reached
from ``_conditional_gate_fe`` / ``_pairwise_modular_fe._mi``), gated behind the resident opt-in
(``MLFRAME_FE_GPU_STRICT_RESIDENT``). The FE-candidate resident MI stays on the edge path (F2 is already
selection-equivalent with edge, and the radix-edge binning + its iter-1 v2 speedups must NOT be touched).
Default flag-off is byte-for-byte unchanged.

BIT-IDENTITY CONTRACT vs ``_quantile_bin_njit``
-----------------------------------------------
* TIE-FREE columns: codes are BIT-IDENTICAL (maxdiff 0) - the sort order is unambiguous, so cupy's stable
  ``argsort`` and numba's ``argsort`` assign the SAME rank -> bin to every row.
* TIED columns: numba's ``argsort`` breaks ties with an UNREPRODUCIBLE introsort artifact (it matches neither
  numpy ``kind='stable'`` nor ``kind='quicksort'``), so the per-row CODES cannot be made bit-identical to it
  on a GPU. BUT the GATE MI is bit-identical anyway: the gate's tied groups are exact zeros whose target label
  is (near-)homogeneous, so whichever bin a tied row lands in, its (bin, class) contribution is invariant ->
  the plug-in MI is the SAME (verified diff 0.0 across the biz-value seeds). The rank binner therefore
  delivers the gate MI byte-match the edge path could not, WITHOUT claiming code bit-identity on ties (a
  property the numba reference itself does not pin reproducibly).

CPU FALLBACK / NO-CUPY: every entry point returns ``None`` on any cupy failure so the caller takes the exact
CPU njit rank path - byte-for-byte the legacy behavior.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import numpy as np
from mlframe.utils.log_throttle import log_throttle
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._device_sort_safe import device_argsort, device_sort  # noqa: F401  (thrust aborts the process on a short pool)

logger = logging.getLogger(__name__)


def _bin_boundaries(n: int, n_bins: int) -> np.ndarray:
    """Cumulative bin-size boundaries matching ``_quantile_bin_njit``: ``base = n // n_bins`` rows per bin,
    the first ``n % n_bins`` bins get one extra row. Returns the (n_bins,) cumulative-count vector; sorted
    position ``p`` belongs to bin ``searchsorted(bnd, p, side='right')`` - the exact rank->bin assignment of
    the njit loop (which fills bin 0 with the first ``size`` sorted positions, bin 1 with the next, ...)."""
    base = n // n_bins
    rem = n % n_bins
    sizes = np.full(n_bins, base, dtype=np.int64)
    if rem:
        sizes[:rem] += 1
    return np.cumsum(sizes)


# (n, n_bins) -> cp.asarray(_bin_boundaries(n, n_bins)) device vector; read-only, shared. ``n`` (dataset row
# count) and ``n_bins`` are fit-constant across a whole FE scan's rank-bin calls, yet the boundary vector was
# rebuilt (host cumsum) AND re-uploaded (H2D) on EVERY call - mirrors ``_gpu_resident_fe._QLEVELS_CACHE`` /
# ``_quantile_levels_dev``. searchsorted never mutates ``bnd``, so sharing the device array across calls is
# safe; the cached array is byte-identical to the inline ``cp.asarray(_bin_boundaries(...))`` (deterministic
# in its args).
_BIN_BOUNDARIES_CACHE: dict = {}
_BIN_BOUNDARIES_LOCK = threading.Lock()


def _bin_boundaries_dev(cp, n: int, n_bins: int):
    """Cached device (n_bins,) cumulative bin-size boundary vector for ``(n, n_bins)``; builds + uploads via
    :func:`_bin_boundaries` on a miss, returns the shared device array on a hit."""
    # Keyed by the current device as well: a device array lives on one GPU, and another device must not receive it.
    key = (int(cp.cuda.get_device_id()), int(n), int(n_bins))
    with _BIN_BOUNDARIES_LOCK:
        hit = _BIN_BOUNDARIES_CACHE.get(key)
    if hit is None:
        hit = cp.asarray(_bin_boundaries(int(n), int(n_bins)))
        with _BIN_BOUNDARIES_LOCK:
            hit = _BIN_BOUNDARIES_CACHE.setdefault(key, hit)
    return hit


def rank_bin_codes_gpu_resident(x_gpu: Any, n_bins: int) -> Any:
    """RANK (argsort equi-frequency) bin codes for a RESIDENT 1-D cupy column. Returns an (n,) cupy int32 code
    vector, or ``None`` on any cupy failure.

    Reproduces ``hermite_fe._quantile_bin_njit``: ``argsort`` the column, then assign the first ``n//n_bins``
    (+1 for the first ``n%n_bins`` bins) sorted positions to bin 0, the next block to bin 1, etc. cupy's
    ``argsort`` is stable -> on TIE-FREE columns the codes are bit-identical to the njit reference (maxdiff 0).
    """
    try:
        import cupy as cp
    except ImportError as e:
        logger.debug("cupy import failed: %s", e)
        return None
    try:
        xg = x_gpu if isinstance(x_gpu, cp.ndarray) else cp.asarray(x_gpu)
        xg = xg.ravel()
        n = int(xg.shape[0])
        if n == 0:
            return cp.empty(0, dtype=cp.int32)
        nb = int(n_bins)
        if nb <= 1:
            return cp.zeros(n, dtype=cp.int32)
        si = device_argsort(xg)  # stable sort; the tie rule below makes the codes order-independent on ties too
        bnd = _bin_boundaries_dev(cp, n, nb)
        pos = cp.arange(n, dtype=cp.int64)
        binid = cp.searchsorted(bnd, pos, side="right").astype(cp.int32)  # bin of each sorted position
        xs = xg[si]
        same = (xs[1:] == xs[:-1]) | (cp.isnan(xs[1:]) & cp.isnan(xs[:-1]))
        is_start = cp.concatenate([cp.ones(1, dtype=cp.bool_), ~same])
        # first sorted position of each row's tie group: the group number (cumulative count of group starts) indexes the start positions
        group_start = cp.nonzero(is_start)[0][cp.cumsum(is_start, dtype=cp.int64) - 1]
        out = cp.empty(n, dtype=cp.int32)
        out[si] = binid[group_start]
        return out
    except Exception as _exc:
        logger.warning("rank_bin_codes_gpu_resident: cupy failed (%s)", _exc)
        return None


_RANK_CODES_SRC = r"""
extern "C" __global__ void rank_codes_from_sorted(const VT_* __restrict__ X, const long long* __restrict__ si, const long long n, const int k, const int nb, int* __restrict__ out) {
    // thread (p, j): sorted position p of column j. The rank code of the row at that position is the bin of the FIRST position of its tie group (n // nb rows per bin, the first n % nb bins one more)
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * (long long)k) return;
    const long long p = t / k;
    const int j = (int)(t - p * k);
#define XS(q) X[si[(long long)(q) * k + j] * k + j]
    const VT_ xp = XS(p);
    long long g = p;
    if (p > 0) {
        const VT_ xq = XS(p - 1);
        if (xp == xq || (isnan(xp) && isnan(xq))) {           // inside a tie group: its first position by bisection (equal values are contiguous, NaN last)
            long long lo = 0, hi = p;
            const bool pn = isnan(xp);
            while (lo < hi) {
                const long long mid = (lo + hi) >> 1;
                const VT_ xm = XS(mid);
                const bool before = pn ? !isnan(xm) : (xm < xp);
                if (before) lo = mid + 1; else hi = mid;
            }
            g = lo;
        }
    }
    const long long base = n / nb, rem = n % nb, cut = rem * (base + 1);
    const int b = (g < cut) ? (int)(g / (base + 1)) : (int)(rem + (g - cut) / base);
    out[si[p * k + j] * k + j] = b;
}
"""
_RANK_CODES_KERNELS: dict = {}


def kernel_rank_codes_from_sorted(cp: Any, real: Any = np.float64) -> Any:  # Intentional float64: the double-precision value matrix is the default instance
    """``rank_codes_from_sorted`` for a float32 or float64 value matrix; compile point of :func:`rank_bin_codes_batch_gpu_resident` and of the register-pressure registry."""
    key = np.dtype(real).name
    with _BIN_BOUNDARIES_LOCK:
        kern = _RANK_CODES_KERNELS.get(key)
        if kern is None:
            kern = _RANK_CODES_KERNELS[key] = cp.RawKernel(_RANK_CODES_SRC.replace("VT_", "float" if key == "float32" else "double"), "rank_codes_from_sorted")
    return kern


def rank_bin_codes_batch_gpu_resident(X_gpu: Any, n_bins: int) -> Any:
    """:func:`rank_bin_codes_batch_gpu_resident_reference` with everything after the per-column argsort in ONE launch (``rank_codes_from_sorted``: the sorted values, the tie groups, the bin of
    each group start and the scatter back to row order were about 40 launches); the codes are the same integers (the bin of a sorted position is computed from ``n`` and ``n_bins`` instead of
    searching the cumulative bin sizes). Other dtypes or layouts take the reference form."""
    try:
        import cupy as cp
    except ImportError as e:
        logger.debug("cupy import failed: %s", e)
        return None
    try:
        Xg = X_gpu if isinstance(X_gpu, cp.ndarray) else cp.asarray(X_gpu)
        if Xg.ndim == 1:
            Xg = Xg[:, None]
        if Xg.dtype not in (cp.float32, cp.float64) or int(n_bins) <= 1 or Xg.size == 0:  # Intentional float64: the kernel reads a float32 or double matrix
            return rank_bin_codes_batch_gpu_resident_reference(Xg, n_bins)
        Xg = cp.ascontiguousarray(Xg)
        n, k = int(Xg.shape[0]), int(Xg.shape[1])
        si = cp.ascontiguousarray(device_argsort(Xg, axis=0).astype(cp.int64, copy=False))  # (n, k) stable per-column sort indices
        out = cp.empty((n, k), dtype=cp.int32)
        kern = kernel_rank_codes_from_sorted(cp, Xg.dtype)
        from pyutilz.performance.gpu_hw_launch import block_size_for

        total = n * k
        threads = block_size_for(kern, work_items=total)
        kern(((total + threads - 1) // threads,), (threads,), (Xg, si, np.int64(n), np.int32(k), np.int32(int(n_bins)), out))
        return out
    except Exception as _exc:
        logger.warning("rank_bin_codes_batch_gpu_resident: fused path failed (%s); the cupy chain runs", _exc)
        return rank_bin_codes_batch_gpu_resident_reference(X_gpu, n_bins)


def rank_bin_codes_batch_gpu_resident_reference(X_gpu: Any, n_bins: int) -> Any:
    """Batched RANK bin codes for a RESIDENT (n, k) cupy matrix - per-column argsort equi-frequency. Returns
    an (n, k) cupy int32 code matrix, or ``None`` on any cupy failure. Each column is binned INDEPENDENTLY,
    bit-identical (per-column) to :func:`rank_bin_codes_gpu_resident`."""
    try:
        import cupy as cp
    except ImportError as e:
        logger.debug("cupy import failed: %s", e)
        return None
    try:
        Xg = X_gpu if isinstance(X_gpu, cp.ndarray) else cp.asarray(X_gpu)
        if Xg.ndim == 1:
            Xg = Xg[:, None]
        n, k = int(Xg.shape[0]), int(Xg.shape[1])
        if n == 0 or k == 0:
            return cp.empty((n, k), dtype=cp.int32)
        nb = int(n_bins)
        if nb <= 1:
            return cp.zeros((n, k), dtype=cp.int32)
        # ARGSORT IS IRREDUCIBLE for the rank byte-match (FIX2 investigation). The equi-frequency
        # RANK code needs every row's GLOBAL sorted position (rank) so the n//nb-row block assignment matches
        # _quantile_bin_njit; that is a total order == a full sort. No partition / edge / quantile-cut shortcut
        # reproduces it: an edge/searchsorted cut (the radix-edge resident path) assigns by VALUE not by RANK and
        # is NOT bit-identical on tie-free columns - it is exactly the divergence this rank binner exists to
        # avoid (89dd47c7). argpartition gives only the nb-1 cut elements, not each row's bin, so it cannot bin
        # the rest without a further sort. The argsort is ~85% of the rank-MI cost (wall_gate_rank_ab.py: rank
        # 5.2x edge) and stays.
        # bench-attempt-rejected (2026-06-28): argsort an f32 VIEW of an f64 operand to halve the sort width.
        # Byte-identical codes ONLY when the input is natively f32 (fix2_f32_byte_match.py maxdiff 0); for general
        # f64 input an f32 downcast can COLLAPSE distinct f64 values, reordering what were tie-free columns ->
        # breaks the bit-identity contract. _mi_classif_batch upcasts the operand to f64 before the binner, so the
        # safe-f32 case is not reachable in-scope. Rejected (correctness).
        # bench-attempt-rejected (2026-06-28): flat-index scatter (si*k+coloff) instead of the 2D fancy scatter.
        # Byte-identical but only ~1% faster (scatter is ~12% of the path; argsort dominates) - not worth the
        # less-readable flat indexing. Kept the 2D scatter.
        si = device_argsort(Xg, axis=0)  # (n, k) stable per-column sort indices
        bnd = _bin_boundaries_dev(cp, n, nb)
        pos = cp.arange(n, dtype=cp.int64)
        binid = cp.searchsorted(bnd, pos, side="right").astype(cp.int32)  # (n,) rank->bin, shared across cols
        xs = cp.take_along_axis(Xg, si, axis=0)
        same = (xs[1:] == xs[:-1]) | (cp.isnan(xs[1:]) & cp.isnan(xs[:-1]))
        is_start = cp.concatenate([cp.ones((1, k), dtype=cp.bool_), ~same], axis=0)
        # first sorted position of each row's tie group, per column: starts are listed column by column, then a row's group
        # number within its column offsets into that list
        start_row = cp.nonzero(is_start.T)[1]
        starts_per_col = is_start.sum(axis=0, dtype=cp.int64)
        col_offset = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(starts_per_col)[:-1]])
        group_no = cp.cumsum(is_start, axis=0, dtype=cp.int64) - 1
        group_start = start_row[col_offset[None, :] + group_no]
        out = cp.empty((n, k), dtype=cp.int32)
        # scatter per column: out[si[:, j], j] = bin of the tie group's first position. Vectorised via broadcast indices.
        cols = cp.broadcast_to(cp.arange(k, dtype=cp.int64), (n, k))
        out[si, cols] = binid[group_start]
        return out
    except Exception as _exc:
        logger.warning("rank_bin_codes_batch_gpu_resident: cupy failed (%s)", _exc)
        try:
            import cupy as _cp

            if isinstance(_exc, _cp.cuda.memory.OutOfMemoryError):
                from ._fe_gpu_vram import note_gpu_oom

                note_gpu_oom()  # this call site has no cushion pre-check; arm the shared cooldown so other callers skip their own doomed attempt.
        except Exception:  # nosec B110 - best-effort cooldown note, never let it mask the original failure
            logger.debug("GPU out-of-memory cooldown note failed", exc_info=True)
        return None


#: Device bytes per element of a candidate block of the rank route at its peak: the float64 block, its argsort (int64), the rank codes (int64) and the sort scratch.
_RANK_MI_BYTES_PER_ELEMENT = 48


def plugin_mi_classif_batch_rank_cuda_resident(
    X_gpu: Any, y_gpu: Any, n_bins: int = 20, *, y_min: int | None = None, n_classes: int | None = None,
) -> Optional[np.ndarray]:
    """:func:`_plugin_mi_classif_batch_rank_block` over column blocks sized from the room of the pool the thread allocates from (inside a stage slot a few hundred MB), so a wide candidate matrix is not
    allocated whole into a pool that cannot hold it. Each column's MI depends on that column and the labels only: the blocks give the values of one call, and a matrix that fits is one call. ``None`` when a block fails."""
    shape = getattr(X_gpu, "shape", None)
    if shape is None or len(shape) != 2 or int(shape[1]) < 2:
        return _plugin_mi_classif_batch_rank_block(X_gpu, y_gpu, n_bins, y_min=y_min, n_classes=n_classes)
    try:
        import cupy as cp

        from ._fe_gpu_vram import pool_column_block

        k = int(shape[1])
        width = pool_column_block(cp, int(shape[0]), bytes_per_column_row=_RANK_MI_BYTES_PER_ELEMENT, budget_fraction=0.5, min_block=16, fallback=k, what="rank resident MI")
    except Exception as exc:  # no cupy or no readable pool: one call
        logger.debug("rank resident MI: block width unknown (%r); one call", exc)
        return _plugin_mi_classif_batch_rank_block(X_gpu, y_gpu, n_bins, y_min=y_min, n_classes=n_classes)
    if width >= k:
        return _plugin_mi_classif_batch_rank_block(X_gpu, y_gpu, n_bins, y_min=y_min, n_classes=n_classes)
    parts = []
    for c0 in range(0, k, width):
        part = _plugin_mi_classif_batch_rank_block(X_gpu[:, c0 : c0 + width], y_gpu, n_bins, y_min=y_min, n_classes=n_classes)
        if part is None:
            return None
        parts.append(np.asarray(part))
    return np.concatenate(parts)


def _plugin_mi_classif_batch_rank_block(
    X_gpu: Any, y_gpu: Any, n_bins: int = 20, *, y_min: int | None = None, n_classes: int | None = None,
) -> Optional[np.ndarray]:
    """RANK-binned plug-in MI on ALREADY-RESIDENT cupy arrays - the rank twin of
    ``_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident`` (which bins by percentile EDGES). Bins each
    column of ``X_gpu`` (n, k) by argsort equi-frequency RANK, then computes plug-in MI vs ``y_gpu`` (n,)
    through the SAME fused MI-from-codes kernel the edge path uses, so the ONLY difference vs the edge path is
    the binning recipe (rank, matching the CPU njit gate scoring). Returns a host (k,) array, or
    ``None`` on any cupy failure (caller takes the CPU njit rank path).

    ``y_min`` / ``n_classes`` may be passed pre-computed (y is a fit-constant) to skip the per-call min/max.
    Only the bounded (k,) MI scalars cross D2H; the operand columns stay resident.
    """
    try:
        import cupy as cp
    except ImportError as e:
        logger.debug("cupy import failed: %s", e)
        return None
    try:
        Xg = X_gpu if isinstance(X_gpu, cp.ndarray) else cp.asarray(X_gpu)
        if Xg.ndim == 1:
            Xg = Xg[:, None]
        n, k = int(Xg.shape[0]), int(Xg.shape[1])
        if n == 0 or k == 0:
            return np.zeros(k, dtype=_crit_np_dtype())
        yg = y_gpu if isinstance(y_gpu, cp.ndarray) else cp.asarray(np.ascontiguousarray(y_gpu).astype(np.int64))
        if yg.dtype != cp.int64:
            yg = yg.astype(cp.int64)
        yg = yg.ravel()
        # y is a fit-constant: derive y_min / n_classes once (cheap) when not supplied, then shift so the
        # bincount index never underflows on negative / non-dense labels - mirrors the edge resident core.
        if y_min is None or n_classes is None:
            _ymm = cp.asnumpy(cp.stack((cp.min(yg), cp.max(yg))))
            y_min = int(_ymm[0])
            n_classes = int(_ymm[1]) - y_min + 1
        if y_min != 0:
            yg = yg - y_min
        codes = rank_bin_codes_batch_gpu_resident(Xg, int(n_bins))
        if codes is None:
            return None
        codes = codes.astype(cp.int64, copy=False)
        from ._fe_batched_mi import binned_mi_from_codes_gpu
        # codes_trusted: rank_bin_codes_batch_gpu_resident emits dense 0..n_bins-1 codes and yg was shifted to
        # dense 0-based above, so the in-range guard cannot fire - skip its blocking min/max sync (FIX1).
        # binned_mi_from_codes_gpu's return type widened to Any when it grew an as_device toggle;
        # this caller always uses the default (host-array) path, so narrow the type back explicitly here.
        mi = binned_mi_from_codes_gpu(codes, yg, kx_per_col=[int(n_bins)] * k, ky=int(n_classes), codes_trusted=True)
        return np.asarray(mi)
    except Exception as _exc:
        log_throttle(
            logger,
            "gpu_rank_bin.plugin_mi_failed",
            logging.WARNING,
            "plugin_mi_classif_batch_rank_cuda_resident: GPU path failed (%s: %s); host fallback",
            type(_exc).__name__,
            _exc,
        )
        return None
