"""Permutation-null admission verdicts of every candidate pair at once, grouped by anchor.

``_step_pairs_rank._pair_perm_null_admission`` decides, for a pair that fails the prevalence ratio but clears the max-T floor, whether the weaker operand's
``CMI(candidate; y | anchor)`` clears its conditional null. Pair by pair that is one observed CMI plus one null per pair, each a handful of small device
kernels and host reads (3567 pairs at about 0.65 ms in a 100000 x 100 every-family fit). The candidates that share an anchor are scored here by ONE
``batched_cmi_gpu(..., return_cards=True)`` call per block of candidates, and the null of each candidate is the analytic one from the cardinalities that
call returns (``precomp_cards``: the same occupied-cell counts the per-pair null recomputes, so the df is identical). Only the analytic regime is batched
(at least ``_cmi_analytic_null_min_n()`` rows); below it the per-pair permutation null runs as before.

The verdicts are returned in a dict keyed by the pair; ``_pair_perm_null_admission`` takes a verdict from it when present and otherwise scores the pair itself.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.utils.env_flags import env_int

from .._fe_cmi_redundancy_null import _CMI_FLOOR_QUANTILE as _CMI_QUANTILE

logger = logging.getLogger(__name__)

_BLOCK_COLUMNS = 64  # candidates per batched CMI launch


def _block_analytic_nulls(n: int, k_z: Any, k_xz: Any, k_yz: Any, k_xyz: Any, quantile: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """``(floor, null_mean, applies)`` of a block from the cardinalities: ``_analytic_cmi_null``'s conditional branch, vectorised (df =
    k_xyz + k_z - k_xz - k_yz, the expected-cell rule over k_xyz cells, floor = chi2.ppf(quantile, df) / 2n clipped at 0, mean = df / 2n). ``applies`` is
    False where that branch would not return (the caller then runs the per-pair null)."""
    from mlframe.feature_selection.filters._analytic_mi_null import _HAVE_CHI2, _chi2, _min_expected_cell, analytic_null_enabled
    from mlframe.feature_selection.filters._fe_cmi_redundancy_null import _cmi_analytic_null_min_n

    k_xz = np.asarray(k_xz, dtype=np.int64)
    k_xyz = np.asarray(k_xyz, dtype=np.int64)
    df = k_xyz + int(k_z) - k_xz - int(k_yz)
    nf = float(max(1, n))
    cells = np.maximum(1, k_xyz)
    applies = (df > 0) & ((nf / cells.astype(np.float64)) >= _min_expected_cell())
    if not (_HAVE_CHI2 and analytic_null_enabled() and n >= _cmi_analytic_null_min_n()):
        applies[:] = False
    floor = np.zeros(df.shape, dtype=np.float64)  # Intentional float64: null statistic
    mean = np.zeros(df.shape, dtype=np.float64)  # Intentional float64: null statistic
    if applies.any():
        d = df[applies]
        floor[applies] = np.maximum(np.asarray(_chi2.ppf(float(_CMI_QUANTILE), d), dtype=np.float64) / (2.0 * nf), 0.0)
        mean[applies] = d / (2.0 * nf)
    return floor, mean, applies


def _block_verdicts(data_dev: Any, cols: list, y: np.ndarray, z_col: int, seeds: list) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """``(cmi, floor, null_mean)`` of the candidate columns ``cols`` given the support column ``z_col``: one batched CMI on the device-resident codes, the
    analytic null of the whole block, the per-pair null (host column) only where the analytic one does not apply."""
    import cupy as cp

    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu
    from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import _conditional_perm_null

    x = data_dev[:, cols]
    z = cp.ascontiguousarray(data_dev[:, z_col]).astype(cp.int64)
    cmi, k_z, k_xz, k_yz, k_xyz = batched_cmi_gpu(cp.ascontiguousarray(x), y, z, return_cards=True)
    floor, mean, applies = _block_analytic_nulls(int(y.shape[0]), k_z, k_xz, k_yz, k_xyz, _CMI_QUANTILE)
    if not applies.all():
        z_h = np.ascontiguousarray(cp.asnumpy(z))
        for j in np.flatnonzero(~applies):
            x_j = np.ascontiguousarray(cp.asnumpy(x[:, int(j)]), dtype=np.int64)
            floor[j], mean[j] = _conditional_perm_null(x_j, y, z_h, seed=seeds[j], precomp_cards=(int(k_z), int(k_xz[j]), int(k_yz), int(k_xyz[j])))
    return np.asarray(cmi, dtype=np.float64), floor, mean  # Intentional float64: statistic vectors


def _resident_codes(data: np.ndarray, stride: int) -> Any:
    """The (strided) code matrix uploaded once in its own narrow dtype: the blocks gather their columns on the device."""
    import cupy as cp

    return cp.asarray(np.ascontiguousarray(data[::stride]))


def batched_perm_null_admissions(self: Any, gate_cache: dict, cached_MIs: dict, data: np.ndarray, classes_y: np.ndarray) -> dict:
    """``{pair: admitted}`` for every pair of ``gate_cache`` that fails prevalence and clears the max-T floor, in the analytic-null regime; ``{}`` when the
    admission is disabled, the regime does not apply or the device path fails (the per-pair path then decides every pair)."""
    if not bool(getattr(self, "fe_pair_perm_null_admission_enable", False)):
        return {}
    from mlframe.feature_selection.filters._fe_cmi_redundancy_null import _cmi_analytic_null_min_n

    n_rows = int(data.shape[0])
    stride = 1
    cap = env_int("MLFRAME_PAIR_NULL_MAX_ROWS", 250000, minimum=0)
    if cap > 0 and n_rows > cap and n_rows // cap > 1:
        stride = n_rows // cap  # the same strided subsample the per-pair path takes
    if len(range(0, n_rows, stride)) < _cmi_analytic_null_min_n():
        return {}
    groups: dict = {}
    for pair, state in gate_cache.items():
        passes_prevalence, passes_maxt = state[0], state[1]
        if passes_prevalence or not passes_maxt:
            continue
        ia, ib = int(pair[0]), int(pair[1])
        anchor, cand = (ia, ib) if cached_MIs[(ia,)] >= cached_MIs[(ib,)] else (ib, ia)
        groups.setdefault(anchor, []).append((pair, cand))
    if not groups:
        return {}
    try:
        return _score_groups(self, groups, cached_MIs, data, classes_y, stride)
    except Exception as exc:
        logger.debug("batched pair admission failed (%s); the pairs are scored one by one", exc)
        return {}


def _score_groups(self: Any, groups: dict, cached_MIs: dict, data: np.ndarray, classes_y: np.ndarray, stride: int) -> dict:
    """The verdicts of every anchor group, one batched CMI per block of candidates."""
    y = np.ascontiguousarray(np.asarray(classes_y, dtype=np.int64)[::stride])
    excess_frac = float(getattr(self, "fe_pair_perm_null_excess_frac", 0.05))
    seed0 = int(getattr(self, "random_seed", 0) or 0)
    data_dev = _resident_codes(data, stride)
    out: dict = {}
    for anchor, members in groups.items():
        anchor_mi = float(cached_MIs[(anchor,)])
        for b0 in range(0, len(members), _BLOCK_COLUMNS):
            block = members[b0 : b0 + _BLOCK_COLUMNS]
            seeds = [seed0 + 7919 * int(pair[0]) + int(pair[1]) for pair, _ in block]
            cmi, floor, mean = _block_verdicts(data_dev, [c for _, c in block], y, int(anchor), seeds)
            for j, (pair, _cand) in enumerate(block):
                obs = float(cmi[j])
                out[pair] = bool(obs > floor[j] and (obs - mean[j]) >= excess_frac * max(anchor_mi, 1e-9))
    return out


def batched_asymmetric_relax_verdicts(self: Any, pairs: list, synergy_added_idx: Any, data: np.ndarray, classes_y: np.ndarray) -> dict:
    """``{pair: relaxed}`` for the asymmetric synergy pairs of ``pairs`` (exactly one bootstrap-added operand): the relaxation of
    ``_resolve_pair_prevalence_gate`` holds when ``CMI(boot; y | partner)`` clears its conditional null by at least the null floor. Grouped by partner,
    one batched CMI per block; analytic-null regime only, ``{}`` otherwise (the gate then scores each pair itself)."""
    if not synergy_added_idx:
        return {}
    from mlframe.feature_selection.filters._fe_cmi_redundancy_null import _cmi_analytic_null_min_n

    n_rows = int(data.shape[0])
    stride = 1
    cap = env_int("MLFRAME_PAIR_NULL_MAX_ROWS", 250000, minimum=0)
    if cap > 0 and n_rows > cap and n_rows // cap > 1:
        stride = n_rows // cap
    if len(range(0, n_rows, stride)) < _cmi_analytic_null_min_n():
        return {}
    groups: dict = {}
    for pair in pairs:
        i0, i1 = pair[0], pair[1]
        b0, b1 = i0 in synergy_added_idx, i1 in synergy_added_idx
        if b0 == b1:
            continue
        boot, partner = (i0, i1) if b0 else (i1, i0)
        groups.setdefault(int(partner), []).append((pair, int(boot)))
    if not groups:
        return {}
    try:
        y = np.ascontiguousarray(np.asarray(classes_y, dtype=np.int64)[::stride])
        seed0 = int(getattr(self, "random_seed", 0) or 0)
        data_dev = _resident_codes(data, stride)
        out: dict = {}
        for partner, members in groups.items():
            for b0_ in range(0, len(members), _BLOCK_COLUMNS):
                block = members[b0_ : b0_ + _BLOCK_COLUMNS]
                seeds = [seed0 + 6271 * boot + partner for _, boot in block]
                cmi, floor, mean = _block_verdicts(data_dev, [b for _, b in block], y, int(partner), seeds)
                for j, (pair, _boot) in enumerate(block):
                    obs = float(cmi[j])
                    out[pair] = bool(obs > floor[j] and (obs - mean[j]) >= floor[j])
        return out
    except Exception as exc:
        logger.debug("batched asymmetric-synergy relaxation failed (%s); the gate scores the pairs one by one", exc)
        return {}


__all__ = ["batched_asymmetric_relax_verdicts", "batched_perm_null_admissions"]
