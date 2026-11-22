"""The prevalence / order-2 maxT gate of every candidate pair in one array pass.

``_resolve_pair_prevalence_gate`` resolves the gate of ONE pair with scalar Python, and the usability pre-pass of ``score_prospective_pairs`` called it once per pair: about 500000 calls at 1000 columns,
the largest single block of that function's host time. Almost all of a pair's gate is arithmetic on its joint MI, the MI of its two operands and a few per-fit constants, so it is evaluated here for
all pairs at once with the same float64 operations; the few pairs that need more (an asymmetric synergy pair with no pre-computed verdict, the family-wise partner test of a pair whose operand clears
the floor alone) go through the scalar code unchanged. ``_resolve_pair_prevalence_gate`` stays as the reference and the fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def resolve_pair_gates_batch(
    pairs: "list[tuple]", pair_mis: "list", ind_sums: "list", singles: "tuple[list, list]", *, scalar_resolve: Any, self: Any, data: Any, classes_y: Any, _pair_resident: Any, _pair_mm_bias: dict,
    _pair_maxt_floor: float, _synergy_added_idx: Any, fe_min_pair_mi_prevalence: float, _synergy_prev_resolved: float, _prevalence_debias_auto: bool, col_codes_cache: dict, conditional_gate: Any,
    relax_verdicts: Any,
) -> "list[tuple]":
    """``[(passes_prevalence, passes_maxt, is_synergy_pair, prev_thresh, floor_cmp)]`` for ``pairs``, the tuple ``_resolve_pair_prevalence_gate`` returns for each.

    ``pair_mis`` / ``ind_sums`` / ``singles`` are the pairs' joint MI, the sum of their operands' MI (as the caller formed it) and the operands' own MI. ``scalar_resolve`` is
    ``_resolve_pair_prevalence_gate``; it handles the pairs this pass does not: a pair with exactly one bootstrap operand and no pre-computed relaxation verdict, and every pair when the MI values are not float64
    (the array arithmetic would then round differently from the scalar code)."""
    n = len(pairs)
    if n == 0:
        return []
    mi = np.asarray(pair_mis)
    ies = np.asarray(ind_sums)
    s0 = np.asarray(singles[0])
    s1 = np.asarray(singles[1])
    common = dict(
        self=self, data=data, classes_y=classes_y, _pair_resident=_pair_resident, _pair_mm_bias=_pair_mm_bias, _pair_maxt_floor=_pair_maxt_floor, _synergy_added_idx=_synergy_added_idx,
        fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence, _synergy_prev_resolved=_synergy_prev_resolved, _prevalence_debias_auto=_prevalence_debias_auto, _col_codes_cache=col_codes_cache,
        conditional_gate=conditional_gate, relax_verdicts=relax_verdicts,
    )

    def _scalar(i: int) -> tuple:
        """The reference result for pair ``i``."""
        result: tuple = scalar_resolve(pairs[i], pair_mis[i], ind_sums[i], single_mis=(singles[0][i], singles[1][i]), **common)
        return result

    if mi.dtype != np.float64 or ies.dtype != np.float64 or s0.dtype != np.float64 or s1.dtype != np.float64:
        return [_scalar(i) for i in range(n)]
    a = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=n)
    b = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=n)
    base = fe_min_pair_mi_prevalence
    prev = np.full(n, base, dtype=np.float64)
    is_syn = np.zeros(n, dtype=bool)
    scalar_idx: "list[int]" = []
    if _synergy_added_idx:
        width = int(max(a.max(), b.max(), max(_synergy_added_idx))) + 1
        syn = np.zeros(width, dtype=bool)
        syn[np.fromiter(_synergy_added_idx, dtype=np.int64)] = True
        b0, b1 = syn[a], syn[b]
        is_syn = b0 | b1
        prev[is_syn] = max(base, _synergy_prev_resolved)
        for i in np.flatnonzero(b0 != b1):
            if relax_verdicts is not None and pairs[i] in relax_verdicts:
                if relax_verdicts[pairs[i]]:
                    prev[i] = base
            else:
                scalar_idx.append(int(i))
    bias = np.zeros(n, dtype=np.float64)
    if _pair_mm_bias:
        for i, p in enumerate(pairs):
            bias[i] = _pair_mm_bias.get(tuple(sorted(p)), 0.0)
    floor_cmp = mi - bias
    obs = floor_cmp if _prevalence_debias_auto else mi
    passes_prev = obs > ies * prev
    passes_maxt = floor_cmp >= _pair_maxt_floor
    if conditional_gate is not None and _pair_maxt_floor > 0.0:
        for i in np.flatnonzero(passes_prev & passes_maxt & (np.maximum(s0, s1) >= _pair_maxt_floor)):
            strong, weak = (0, 1) if s0[i] >= s1[i] else (1, 0)
            try:
                passes_maxt[i] = bool(conditional_gate.clears(pairs[i][weak], pairs[i][strong]))
            except Exception as e:  # nosec B110 - best-effort path, the floor verdict stands (as in the scalar code)
                logger.debug("family-wise partner test failed, keeping the floor verdict: %s", e)
    out = list(zip(passes_prev.tolist(), passes_maxt.tolist(), is_syn.tolist(), prev.tolist(), floor_cmp.tolist()))
    for i in scalar_idx:
        out[i] = _scalar(i)
    return out
