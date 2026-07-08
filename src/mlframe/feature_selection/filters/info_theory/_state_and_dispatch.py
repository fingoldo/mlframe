"""Thread-local normalization/aggregator toggles (SU / JMIM / BUR) and the Python-level dispatchers that route raw-MI vs SU based on them.

The ``threading.local()`` singletons live HERE and ONLY here: the setters, getters, and dispatchers that read/write them share this one module instance.
The facade re-exports them; nothing else duplicates them.
"""
from __future__ import annotations

import threading as _threading

import numpy as np

from ._entropy_kernels import (
    conditional_mi,
    conditional_symmetric_uncertainty,
    mi,
    mi_miller_madow,
    symmetric_uncertainty,
)

# 2026-05-28: thread-local SU toggle. Set by MRMR.fit when mi_normalization='su',
# read by evaluation.py / fleuret.py at the scoring sites. Pure functions ``mi``
# and ``conditional_mi`` stay legacy bit-for-bit so cached entropy numbers
# across the rest of the project don't shift.
_SU_STATE = _threading.local()


def use_su_normalization() -> bool:
    return bool(getattr(_SU_STATE, "active", False))


def set_su_normalization(active: bool) -> None:
    _SU_STATE.active = bool(active)


# 2026-05-30 Wave 8 — JMIM / BUR thread-local toggles. Set by MRMR.fit when
# ``redundancy_aggregator='jmim'`` or ``bur_lambda > 0``; read at the
# evaluation hot-path. Same pattern as SU above. Independent of SU so the
# three can compose freely.
_JMIM_STATE = _threading.local()
_BUR_STATE = _threading.local()


def use_jmim_aggregator() -> bool:
    return bool(getattr(_JMIM_STATE, "active", False))


def set_jmim_aggregator(active: bool) -> None:
    _JMIM_STATE.active = bool(active)


def get_bur_lambda() -> float:
    """Returns the current thread-local BUR weight (0.0 = off)."""
    return float(getattr(_BUR_STATE, "weight", 0.0))


def set_bur_lambda(weight: float) -> None:
    _BUR_STATE.weight = float(weight)


# Miller-Madow relevance-MI bias correction toggle. Set by MRMR.fit when ``mi_correction='miller_madow'``; read at the relevance dispatcher below. Independent of
# SU (SU normalises by cardinality; MM subtracts the closed-form small-sample bias), so the two compose but MM is skipped when SU is active (SU's ratio is the
# normalisation and the additive MM bias term has no clean SU analogue without re-correcting both entropies).
_MM_STATE = _threading.local()


def use_mi_miller_madow() -> bool:
    return bool(getattr(_MM_STATE, "active", False))


def set_mi_miller_madow(active: bool) -> None:
    _MM_STATE.active = bool(active)


# Group-aware MI (per-group I(X;Y|G)) state. Set by MRMR.fit when ``group_aware_mi=True`` and groups are supplied,
# republished into joblib workers by ``evaluate_candidates`` (thread-local does not cross workers), and read at the
# relevance scoring site. ``None`` payload = OFF -> the legacy global-MI relevance is byte-identical. Holds the once-
# computed segment sort + offsets so every candidate reuses them.
_GROUP_MI_STATE = _threading.local()


def use_group_mi() -> bool:
    return getattr(_GROUP_MI_STATE, "payload", None) is not None


def get_group_mi():
    """Return the group-MI payload ``(sort_idx, group_offsets, min_rows, size_weighted)`` or ``None`` when off."""
    return getattr(_GROUP_MI_STATE, "payload", None)


def set_group_mi(payload) -> None:
    _GROUP_MI_STATE.payload = payload


# Research-knob thread-locals (RelaxMRMR 3-D redundancy, PID synergy bonus, CMI permutation early-stop). All default OFF so the legacy Fleuret score is byte-identical;
# set by MRMR.fit from the matching constructor knobs, read in evaluation.py at the per-candidate scoring site, forwarded to joblib workers like the SU/JMIM/BUR toggles.
_RELAXMRMR_STATE = _threading.local()
_PID_STATE = _threading.local()
_CMI_PERM_STATE = _threading.local()


def get_relaxmrmr_alpha() -> float:
    """RelaxMRMR 3-D-redundancy weight (0.0 = off -> classic Fleuret score)."""
    return float(getattr(_RELAXMRMR_STATE, "alpha", 0.0))


def set_relaxmrmr_alpha(alpha: float) -> None:
    _RELAXMRMR_STATE.alpha = float(alpha)


def get_pid_synergy_bonus() -> float:
    """PID synergy bonus weight (0.0 = off -> no synergy term added)."""
    return float(getattr(_PID_STATE, "bonus", 0.0))


def set_pid_synergy_bonus(bonus: float) -> None:
    _PID_STATE.bonus = float(bonus)


def get_cmi_perm_stop() -> tuple:
    """CMI permutation early-stop config: ``(active, alpha, n_permutations)``; active=False = off."""
    st = _CMI_PERM_STATE
    return bool(getattr(st, "active", False)), float(getattr(st, "alpha", 0.05)), int(getattr(st, "n_permutations", 100))


def set_cmi_perm_stop(active: bool, alpha: float = 0.05, n_permutations: int = 100) -> None:
    _CMI_PERM_STATE.active = bool(active)
    _CMI_PERM_STATE.alpha = float(alpha)
    _CMI_PERM_STATE.n_permutations = int(n_permutations)


def mi_or_su(factors_data, x, y, factors_nbins, verbose=False, dtype=np.int32) -> float:
    """Dispatch raw MI / SU / Miller-Madow-corrected MI based on the thread-local toggles. Cheap path
    when all are off: a one-call delegation to ``mi`` (which is njit-cached)."""
    if use_su_normalization():
        return float(symmetric_uncertainty(factors_data, x, y, factors_nbins, verbose=verbose, dtype=dtype))
    if use_mi_miller_madow():
        return float(mi_miller_madow(factors_data, x, y, factors_nbins, verbose=verbose, dtype=dtype))
    return float(mi(factors_data, x, y, factors_nbins, verbose=verbose, dtype=dtype))


def cmi_or_csu(factors_data, x, y, z, factors_nbins, dtype=np.int32, **mi_kwargs) -> float:
    """Dispatch conditional MI or CSU based on the thread-local toggle. ``conditional_mi``
    accepts a richer kwarg surface (cache, can_use_x_cache, ...); when SU is on those caches
    are bypassed because the SU denominator is path-dependent on the same entropies, so
    caching the unconditional CMI would silently desync."""
    if use_su_normalization():
        return float(conditional_symmetric_uncertainty(factors_data, x, y, z, factors_nbins, dtype=dtype))
    return float(conditional_mi(factors_data, x, y, z, factors_nbins=factors_nbins, dtype=dtype, **mi_kwargs))
