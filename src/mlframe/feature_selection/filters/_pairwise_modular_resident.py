"""Resident-operand permutation-null MI for the pairwise-modular FE scan (the SF2 :311 H2D collapse, 2026-06-30).

``_pairwise_modular_fe._perm_null_hi`` is the dominant repetition of the modular scan: for the per-combiner best
modulus it builds the residue ``r = c mod k`` ONCE on the host, then loops ``n_perm`` (=12) times scoring
``_mi(r, y[perm_i])``. Each ``_mi`` reshapes its single residue column to ``(n, 1)`` and -- under
``MLFRAME_FE_GPU_STRICT`` -- routes through the resident plug-in MI, ``cp.asarray``-uploading that ``(n, 1)``
host residue FRESH on every one of the 12 calls at ``_orth_mi_backends.py:311`` (the SF2 share of a 300k STRICT
F2 byte-audit: ~43 MB / 22x single-column residue uploads, the perm-null being the bulk).

BIT-IDENTITY (the code-reorder identity). The host STRICT ``_mi`` rank path bins ``r`` into integer codes ONCE
(``rank_bin_codes_batch_gpu_resident``; ``r`` is the SAME across all 12 perms, so the codes are identical every
perm) and then computes ``MI(codes_r; y[perm_i])`` via ``binned_mi_from_codes_gpu``. MI is invariant under a
JOINT reindex of both variables, so ``MI(codes_r; y[perm]) == MI(codes_r[inv_perm]; y)`` where ``inv_perm`` is
the inverse of ``perm``. Reordering the INTEGER CODES is a pure index gather with NO re-binning, so it has none
of the tie-break ambiguity that reordering the float RESIDUE before binning would (the residue is heavily tied
-- only ``k`` distinct values -- so binning ``r[inv_perm]`` would NOT equal ``codes_r[inv_perm]``). Therefore we
bin ``r`` ONCE on the device, gather ``codes_r[inv_perm_i]`` into a single ``(n, n_perm)`` code matrix, and
score it against the SAME resident y in ONE ``binned_mi_from_codes_gpu`` launch -- byte-identical to the host
per-perm rank loop, with the 12 uploads collapsed to one (small int) code-matrix transient. The permutations are
the SAME seeded host ``rng.permutation`` arrays (passed in), drawn in the SAME loop order.

GATE: ``fe_gpu_device_born_modular_enabled`` (DEFAULT ON under STRICT, opt-out
``MLFRAME_FE_GPU_DEVICE_BORN_MODULAR=0``). Bit-identity holds for the RANK binner (the STRICT ``_mi`` path); the
EDGE binner is not reproduced here (it would require the plug-in's internal percentile-edge codes), so the
non-rank case returns ``None`` and the caller takes the EXACT host loop. On ANY cupy error / no cupy / non-strict
it returns ``None`` and the caller takes the EXACT host per-perm ``_mi`` loop (byte-identical default path
untouched). NEVER ``free_all_blocks``.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["perm_null_residue_mis_resident", "residue_grid_mis_resident", "combiner_mi_resident"]


def combiner_mi_resident(
    c: np.ndarray,
    y,
    *,
    nbins: int,
    rank_binning: bool,
    modulus: int = 0,
) -> Optional[float]:
    """DEVICE-BORN twin of the single-column ``_mi`` calls on an integer combiner: the per-combiner BASELINE
    ``_mi(c_arr; y)`` (``modulus=0``) and the escalate-stage ``_residue_mi`` ``MI(c mod m; y)`` (``modulus=m``).

    Uploads the integer combiner column ONCE via the resident-operand cache, computes the residue ``c % m`` on
    device when ``modulus > 0`` (exact integer cupy ``%``; ``modulus=0`` scores the raw combiner), and scores it
    through the SAME (rank, under STRICT) resident estimator ``_mi`` uses -- so the baseline / residue MI is
    byte-identical to the host ``_mi`` (rank codes from the SAME column). Returns a Python float, OR ``None`` when
    STRICT-residency is off / cupy is unavailable / the binner is not the rank binner / any cupy fault -- in which
    case the caller takes the EXACT host ``_mi`` (byte-identical default path untouched)."""
    try:
        from ._gpu_strict_fe import fe_gpu_device_born_modular_enabled

        if not fe_gpu_device_born_modular_enabled():
            return None
    except Exception:
        return None
    if not rank_binning:
        return None  # only the rank path (the STRICT ``_mi`` binner) is reproduced bit-identically here.
    try:
        import cupy as cp

        from ._fe_resident_operands import resident_operand
        from ._gpu_resident_rank_bin import plugin_mi_classif_batch_rank_cuda_resident

        cf = np.ascontiguousarray(np.asarray(c, dtype=np.int64)).ravel()
        cg = resident_operand(cf, ("modular_combiner", cf.shape[0]), dtype=np.int64)
        col = (cg % int(modulus)) if int(modulus) > 0 else cg
        Xd = col.astype(cp.float64)[:, None]
        _yi = np.ascontiguousarray(np.asarray(y)).astype(np.int64).ravel()
        yd = resident_operand(_yi, "y_mi_classif", dtype=np.int64)
        _ymin = int(_yi.min()) if _yi.size else 0
        _ncls = (int(_yi.max()) - _ymin + 1) if _yi.size else 1
        mi = plugin_mi_classif_batch_rank_cuda_resident(Xd, yd, int(nbins), y_min=_ymin, n_classes=_ncls)
        if mi is None:
            return None
        return float(np.asarray(mi, dtype=np.float64)[0])
    except Exception as _exc:  # noqa: BLE001
        logger.debug("combiner_mi_resident: GPU path failed (%s); host fallback", _exc)
        return None


def residue_grid_mis_resident(
    c: np.ndarray,
    y,
    mods,
    *,
    nbins: int,
) -> Optional[np.ndarray]:
    """DEVICE-BORN twin of ``_residue_grid_mi``: ``MI(c mod k; y)`` for every k in ``mods``, building the
    residues ON the device from a single resident combiner column (collapsing the per-group residue-matrix upload
    at ``_orth_mi_backends.py:139`` -- the dominant modular :311 H2D). The combiner ``c`` uploads ONCE via the
    resident-operand cache; ``c % k`` is exact integer arithmetic on device; each effective-nbins group
    (``max(nbins, k)``) is scored in one resident plug-in call (the SAME percentile-EDGE estimator the host
    ``_residue_grid_mi`` routes to under STRICT -- the grid path leaves ``rank_binning`` False).

    Returns a host ``(len(mods),)`` float64 array in ``mods`` order, OR ``None`` when STRICT-residency is off /
    cupy is unavailable / any cupy fault -- in which case the caller takes the EXACT host ``_residue_grid_mi``
    (byte-identical default path untouched). Selection-equivalent to the host grid: the device residue ``c % k``
    has the SAME integer values as ``np.mod(c, k)``, so the percentile-edge partition -- and the per-column MI --
    matches the host edge path."""
    try:
        from ._gpu_strict_fe import fe_gpu_device_born_modular_enabled

        if not fe_gpu_device_born_modular_enabled():
            return None
    except Exception:
        return None
    mods = [int(k) for k in mods]
    if not mods:
        return np.empty((0,), dtype=np.float64)
    try:
        import cupy as cp

        from ._fe_resident_operands import resident_operand
        from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

        cf = np.ascontiguousarray(np.asarray(c, dtype=np.int64)).ravel()
        cg = resident_operand(cf, ("modular_combiner", cf.shape[0]), dtype=np.int64)
        _yi = np.ascontiguousarray(np.asarray(y)).astype(np.int64).ravel()
        yd = resident_operand(_yi, "y_mi_classif", dtype=np.int64)
        _ymin = int(_yi.min()) if _yi.size else 0
        _ncls = (int(_yi.max()) - _ymin + 1) if _yi.size else 1
        out = np.empty(len(mods), dtype=np.float64)
        # Group moduli by effective nbins exactly as the host does, so each group is one resident plug-in call.
        groups: dict[int, list[int]] = {}
        for idx, k in enumerate(mods):
            groups.setdefault(max(int(nbins), k), []).append(idx)
        for eff_nbins, idxs in groups.items():
            cols = [(cg % int(mods[idx])).astype(cp.float64) for idx in idxs]
            mat = cp.ascontiguousarray(cp.stack(cols, axis=1))
            mis = _plugin_mi_classif_batch_cuda_resident(mat, yd, int(eff_nbins), y_min=_ymin, n_classes=_ncls)
            mis = np.asarray(mis, dtype=np.float64)
            for j, idx in enumerate(idxs):
                out[idx] = float(mis[j])
        return out
    except Exception as _exc:  # noqa: BLE001
        logger.debug("residue_grid_mis_resident: GPU path failed (%s); host fallback", _exc)
        return None


def perm_null_residue_mis_resident(
    r: np.ndarray,
    y,
    perms,
    *,
    eff_nbins: int,
    rank_binning: bool,
) -> Optional[np.ndarray]:
    """Score ``MI(r; y[perm_i])`` for every permutation in ``perms`` in ONE resident plug-in call, via the
    code-reorder identity ``MI(codes_r; y[perm]) == MI(codes_r[inv_perm]; y)``.

    ``r`` is the host residue ``(c mod k)`` (float64), ``perms`` a sequence of index permutations (the SAME
    seeded ``rng.permutation`` arrays the host loop uses, in the SAME order), ``eff_nbins`` the per-residue bin
    count (``max(nbins, k)``), ``rank_binning`` the STRICT gate-MI binner select. Returns a host ``(n_perm,)``
    float64 array of the per-perm residue MIs, OR ``None`` when STRICT-residency is off / cupy is unavailable /
    the binner is not the rank binner / any cupy fault -- in which case the caller takes the EXACT host per-perm
    ``_mi`` loop (byte-identical default path)."""
    try:
        from ._gpu_strict_fe import fe_gpu_device_born_modular_enabled

        if not fe_gpu_device_born_modular_enabled():
            return None
    except Exception:
        return None
    if not rank_binning:
        # Only the RANK code path is reproduced bit-identically here (the STRICT ``_mi`` binner). The EDGE
        # binner's codes live inside the plug-in's percentile path -> defer to the host loop.
        return None
    perms = list(perms)
    if not perms:
        return np.empty((0,), dtype=np.float64)
    try:
        import cupy as cp

        from ._fe_resident_operands import resident_operand
        from ._fe_batched_mi import binned_mi_from_codes_gpu
        from ._gpu_resident_rank_bin import rank_bin_codes_batch_gpu_resident

        rf = np.ascontiguousarray(np.asarray(r, dtype=np.float64)).ravel()
        n = rf.size
        n_perm = len(perms)
        nb = int(eff_nbins)
        # Bin the residue ONCE on device (rank codes), exactly as the host STRICT ``_mi`` rank path does (same r,
        # same binner -> identical codes). The residue rides the resident-operand cache (keyed on its content
        # fingerprint -> re-uploads when r changes; the WIN is the codes-matrix replaces 12 residue uploads).
        rg = resident_operand(rf, ("modular_perm_residue", nb), dtype=cp.float64)
        codes_r = rank_bin_codes_batch_gpu_resident(rg[:, None], nb)
        if codes_r is None:
            return None
        codes_r = codes_r.astype(cp.int64, copy=False).ravel()
        # column i = codes_r[inv_perm_i]; np.argsort(perm) is the inverse permutation. Build the inverse-perm
        # index matrix on the host (cheap int gather) and gather on device -> (n, n_perm) int64 code matrix.
        inv_idx = np.empty((n, n_perm), dtype=np.int64)
        for i, perm in enumerate(perms):
            inv_idx[:, i] = np.argsort(np.asarray(perm))
        inv_g = cp.asarray(np.ascontiguousarray(inv_idx))
        code_mat = cp.ascontiguousarray(codes_r[inv_g])  # (n, n_perm), gathered codes
        # y rides the SAME resident "y_mi_classif" role the host STRICT MI path uses (uploaded once per fit).
        _yi = np.ascontiguousarray(np.asarray(y)).astype(np.int64).ravel()
        yd = resident_operand(_yi, "y_mi_classif", dtype=np.int64)
        _ymin = int(_yi.min()) if _yi.size else 0
        if _ymin:
            yd = yd - _ymin
        _ncls = (int(_yi.max()) - _ymin + 1) if _yi.size else 1
        mis = binned_mi_from_codes_gpu(code_mat, yd, kx_per_col=[nb] * n_perm, ky=int(_ncls))
        return np.asarray(mis, dtype=np.float64)
    except Exception as _exc:  # noqa: BLE001
        logger.debug("perm_null_residue_mis_resident: GPU path failed (%s); host fallback", _exc)
        return None
