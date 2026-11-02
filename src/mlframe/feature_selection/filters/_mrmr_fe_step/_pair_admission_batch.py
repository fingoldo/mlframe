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

logger = logging.getLogger(__name__)

_BLOCK_COLUMNS = 64  # candidates per batched CMI launch


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
    from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu
    from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import _conditional_perm_null

    y = np.ascontiguousarray(np.asarray(classes_y, dtype=np.int64)[::stride])
    excess_frac = float(getattr(self, "fe_pair_perm_null_excess_frac", 0.05))
    seed0 = int(getattr(self, "random_seed", 0) or 0)
    out: dict = {}
    for anchor, members in groups.items():
        z = np.ascontiguousarray(np.asarray(data[::stride, anchor], dtype=np.int64))
        anchor_mi = float(cached_MIs[(anchor,)])
        for b0 in range(0, len(members), _BLOCK_COLUMNS):
            block = members[b0 : b0 + _BLOCK_COLUMNS]
            cand_cols = [c for _, c in block]
            x = np.ascontiguousarray(np.asarray(data[::stride][:, cand_cols], dtype=np.int64))
            cmi, k_z, k_xz, k_yz, k_xyz = batched_cmi_gpu(x, y, z, return_cards=True)
            for j, (pair, _cand) in enumerate(block):
                ia, ib = int(pair[0]), int(pair[1])
                floor, null_mean = _conditional_perm_null(
                    np.ascontiguousarray(x[:, j]), y, z, seed=seed0 + 7919 * ia + ib, precomp_cards=(int(k_z), int(k_xz[j]), int(k_yz), int(k_xyz[j])),
                )
                obs = float(cmi[j])
                out[pair] = bool(obs > floor and (obs - null_mean) >= excess_frac * max(anchor_mi, 1e-9))
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
        from mlframe.feature_selection.filters._fe_batched_mi import batched_cmi_gpu
        from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import _conditional_perm_null

        y = np.ascontiguousarray(np.asarray(classes_y, dtype=np.int64)[::stride])
        seed0 = int(getattr(self, "random_seed", 0) or 0)
        out: dict = {}
        for partner, members in groups.items():
            z = np.ascontiguousarray(np.asarray(data[::stride, partner], dtype=np.int64))
            for b0_ in range(0, len(members), _BLOCK_COLUMNS):
                block = members[b0_ : b0_ + _BLOCK_COLUMNS]
                x = np.ascontiguousarray(np.asarray(data[::stride][:, [b for _, b in block]], dtype=np.int64))
                cmi, k_z, k_xz, k_yz, k_xyz = batched_cmi_gpu(x, y, z, return_cards=True)
                for j, (pair, boot) in enumerate(block):
                    floor, null_mean = _conditional_perm_null(
                        np.ascontiguousarray(x[:, j]), y, z, seed=seed0 + 6271 * boot + partner, precomp_cards=(int(k_z), int(k_xz[j]), int(k_yz), int(k_xyz[j])),
                    )
                    obs = float(cmi[j])
                    out[pair] = bool(obs > floor and (obs - null_mean) >= floor)
        return out
    except Exception as exc:
        logger.debug("batched asymmetric-synergy relaxation failed (%s); the gate scores the pairs one by one", exc)
        return {}


__all__ = ["batched_asymmetric_relax_verdicts", "batched_perm_null_admissions"]
