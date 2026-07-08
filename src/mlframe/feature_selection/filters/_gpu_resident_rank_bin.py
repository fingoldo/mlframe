"""GPU-resident RANK (argsort equi-frequency) binner + rank-binned plug-in MI for the CONDITIONAL-GATE MI
path under STRICT residency (2026-06-28).

WHY THIS EXISTS
---------------
The STRICT-residency MI core ``_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident`` bins by PERCENTILE
EDGES (the radix-edge fast path -- 71e31818 / 6b7370b4). Edge and the CPU catalog's RANK binning
(``hermite_fe._quantile_bin_njit`` argsort equi-frequency) agree bit-for-bit ONLY on tie-free columns; on
heavily-tied columns they diverge (edge lumps all tied values into one bin; rank splits the ties across bin
boundaries). The conditional-gate operator ``gate_mask`` emits ``1[c>0]*a`` -- ~50% EXACT zeros -- so under
STRICT the gate MI was edge-binned and did NOT byte-match the CPU rank MI the gate scoring uses (commit
89dd47c7 pinned the biz-value test to the rank estimator as a stopgap).

This module provides a cupy ``argsort``-based RANK binner that reproduces ``_quantile_bin_njit``'s recipe
(``base = n // n_bins`` rows per bin, first ``n % n_bins`` bins get one extra), and a rank-binned plug-in MI
that the GATE opts into so its STRICT MI matches the CPU rank MI.

SCOPE -- GATE MI ONLY
---------------------
This is wired ONLY into the gate MI (``_orth_mi_backends._mi_classif_batch(..., rank_binning=True)`` reached
from ``_conditional_gate_fe`` / ``_pairwise_modular_fe._mi``), gated behind the resident opt-in
(``MLFRAME_FE_GPU_STRICT_RESIDENT``). The FE-candidate resident MI stays on the edge path (F2 is already
selection-equivalent with edge, and the radix-edge binning + its iter-1 v2 speedups must NOT be touched).
Default flag-off is byte-for-byte unchanged.

BIT-IDENTITY CONTRACT vs ``_quantile_bin_njit``
-----------------------------------------------
* TIE-FREE columns: codes are BIT-IDENTICAL (maxdiff 0) -- the sort order is unambiguous, so cupy's stable
  ``argsort`` and numba's ``argsort`` assign the SAME rank -> bin to every row.
* TIED columns: numba's ``argsort`` breaks ties with an UNREPRODUCIBLE introsort artifact (it matches neither
  numpy ``kind='stable'`` nor ``kind='quicksort'``), so the per-row CODES cannot be made bit-identical to it
  on a GPU. BUT the GATE MI is bit-identical anyway: the gate's tied groups are exact zeros whose target label
  is (near-)homogeneous, so whichever bin a tied row lands in, its (bin, class) contribution is invariant ->
  the plug-in MI is the SAME (verified diff 0.0 across the biz-value seeds). The rank binner therefore
  delivers the gate MI byte-match the edge path could not, WITHOUT claiming code bit-identity on ties (a
  property the numba reference itself does not pin reproducibly).

CPU FALLBACK / NO-CUPY: every entry point returns ``None`` on any cupy failure so the caller takes the exact
CPU njit rank path -- byte-for-byte the legacy behavior.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _bin_boundaries(n: int, n_bins: int) -> np.ndarray:
    """Cumulative bin-size boundaries matching ``_quantile_bin_njit``: ``base = n // n_bins`` rows per bin,
    the first ``n % n_bins`` bins get one extra row. Returns the (n_bins,) cumulative-count vector; sorted
    position ``p`` belongs to bin ``searchsorted(bnd, p, side='right')`` -- the exact rank->bin assignment of
    the njit loop (which fills bin 0 with the first ``size`` sorted positions, bin 1 with the next, ...)."""
    base = n // n_bins
    rem = n % n_bins
    sizes = np.full(n_bins, base, dtype=np.int64)
    if rem:
        sizes[:rem] += 1
    return np.cumsum(sizes)


def rank_bin_codes_gpu_resident(x_gpu: Any, n_bins: int) -> Any:
    """RANK (argsort equi-frequency) bin codes for a RESIDENT 1-D cupy column. Returns an (n,) cupy int32 code
    vector, or ``None`` on any cupy failure.

    Reproduces ``hermite_fe._quantile_bin_njit``: ``argsort`` the column, then assign the first ``n//n_bins``
    (+1 for the first ``n%n_bins`` bins) sorted positions to bin 0, the next block to bin 1, etc. cupy's
    ``argsort`` is stable -> on TIE-FREE columns the codes are bit-identical to the njit reference (maxdiff 0).
    """
    try:
        import cupy as cp
    except Exception:
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
        si = cp.argsort(xg)  # cupy argsort is stable -> bit-identical rank on tie-free columns
        bnd = cp.asarray(_bin_boundaries(n, nb))
        pos = cp.arange(n, dtype=cp.int64)
        binid = cp.searchsorted(bnd, pos, side="right").astype(cp.int32)
        out = cp.empty(n, dtype=cp.int32)
        out[si] = binid
        return out
    except Exception as _exc:
        logger.debug("rank_bin_codes_gpu_resident: cupy failed (%s)", _exc)
        return None


def rank_bin_codes_batch_gpu_resident(X_gpu: Any, n_bins: int) -> Any:
    """Batched RANK bin codes for a RESIDENT (n, k) cupy matrix -- per-column argsort equi-frequency. Returns
    an (n, k) cupy int32 code matrix, or ``None`` on any cupy failure. Each column is binned INDEPENDENTLY,
    bit-identical (per-column) to :func:`rank_bin_codes_gpu_resident`."""
    try:
        import cupy as cp
    except Exception:
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
        # ARGSORT IS IRREDUCIBLE for the rank byte-match (FIX2 investigation, 2026-06-28). The equi-frequency
        # RANK code needs every row's GLOBAL sorted position (rank) so the n//nb-row block assignment matches
        # _quantile_bin_njit; that is a total order == a full sort. No partition / edge / quantile-cut shortcut
        # reproduces it: an edge/searchsorted cut (the radix-edge resident path) assigns by VALUE not by RANK and
        # is NOT bit-identical on tie-free columns -- it is exactly the divergence this rank binner exists to
        # avoid (89dd47c7). argpartition gives only the nb-1 cut elements, not each row's bin, so it cannot bin
        # the rest without a further sort. The argsort is ~85% of the rank-MI cost (wall_gate_rank_ab.py: rank
        # 5.2x edge) and stays.
        # bench-attempt-rejected (2026-06-28): argsort an f32 VIEW of an f64 operand to halve the sort width.
        # Byte-identical codes ONLY when the input is natively f32 (fix2_f32_byte_match.py maxdiff 0); for general
        # f64 input an f32 downcast can COLLAPSE distinct f64 values, reordering what were tie-free columns ->
        # breaks the bit-identity contract. _mi_classif_batch upcasts the operand to f64 before the binner, so the
        # safe-f32 case is not reachable in-scope. Rejected (correctness).
        # bench-attempt-rejected (2026-06-28): flat-index scatter (si*k+coloff) instead of the 2D fancy scatter.
        # Byte-identical but only ~1% faster (scatter is ~12% of the path; argsort dominates) -- not worth the
        # less-readable flat indexing. Kept the 2D scatter.
        si = cp.argsort(Xg, axis=0)  # (n, k) stable per-column sort indices
        bnd = cp.asarray(_bin_boundaries(n, nb))
        pos = cp.arange(n, dtype=cp.int64)
        binid = cp.searchsorted(bnd, pos, side="right").astype(cp.int32)  # (n,) rank->bin, shared across cols
        out = cp.empty((n, k), dtype=cp.int32)
        # scatter per column: out[si[:, j], j] = binid. Vectorised via take_along_axis-style assignment.
        cols = cp.broadcast_to(cp.arange(k, dtype=cp.int64), (n, k))
        out[si, cols] = binid[:, None]
        return out
    except Exception as _exc:
        logger.debug("rank_bin_codes_batch_gpu_resident: cupy failed (%s)", _exc)
        return None


def plugin_mi_classif_batch_rank_cuda_resident(
    X_gpu: Any, y_gpu: Any, n_bins: int = 20, *, y_min: int | None = None, n_classes: int | None = None,
) -> Optional[np.ndarray]:
    """RANK-binned plug-in MI on ALREADY-RESIDENT cupy arrays -- the rank twin of
    ``_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident`` (which bins by percentile EDGES). Bins each
    column of ``X_gpu`` (n, k) by argsort equi-frequency RANK, then computes plug-in MI vs ``y_gpu`` (n,)
    through the SAME fused MI-from-codes kernel the edge path uses, so the ONLY difference vs the edge path is
    the binning recipe (rank, matching the CPU njit gate scoring). Returns a host (k,) float64 array, or
    ``None`` on any cupy failure (caller takes the CPU njit rank path).

    ``y_min`` / ``n_classes`` may be passed pre-computed (y is a fit-constant) to skip the per-call min/max.
    Only the bounded (k,) MI scalars cross D2H; the operand columns stay resident.
    """
    try:
        import cupy as cp
    except Exception:
        return None
    try:
        Xg = X_gpu if isinstance(X_gpu, cp.ndarray) else cp.asarray(X_gpu)
        if Xg.ndim == 1:
            Xg = Xg[:, None]
        n, k = int(Xg.shape[0]), int(Xg.shape[1])
        if n == 0 or k == 0:
            return np.zeros(k, dtype=np.float64)
        yg = y_gpu if isinstance(y_gpu, cp.ndarray) else cp.asarray(np.ascontiguousarray(y_gpu).astype(np.int64))
        if yg.dtype != cp.int64:
            yg = yg.astype(cp.int64)
        yg = yg.ravel()
        # y is a fit-constant: derive y_min / n_classes once (cheap) when not supplied, then shift so the
        # bincount index never underflows on negative / non-dense labels -- mirrors the edge resident core.
        if y_min is None or n_classes is None:
            _ymm = cp.asnumpy(cp.stack((cp.min(yg), cp.max(yg))))
            y_min = int(_ymm[0])
            n_classes = int(_ymm[1]) - y_min + 1
        if y_min:
            yg = yg - y_min
        codes = rank_bin_codes_batch_gpu_resident(Xg, int(n_bins))
        if codes is None:
            return None
        codes = codes.astype(cp.int64, copy=False)
        from ._fe_batched_mi import binned_mi_from_codes_gpu
        # codes_trusted: rank_bin_codes_batch_gpu_resident emits dense 0..n_bins-1 codes and yg was shifted to
        # dense 0-based above, so the in-range guard cannot fire -- skip its blocking min/max sync (FIX1).
        return binned_mi_from_codes_gpu(codes, yg, kx_per_col=[int(n_bins)] * k, ky=int(n_classes), codes_trusted=True)
    except Exception as _exc:
        logger.debug("plugin_mi_classif_batch_rank_cuda_resident: GPU path failed (%s); host fallback", _exc)
        return None
