"""Greedy diverse-top-M selection for the Hermite pair-coefficient search.

Carved out of ``_hermite_fe_optimise`` for file size; pure numpy, no dependency on the search code. Re-imported there, so import sites are unchanged.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._safe_scale import unit_vector


def _select_diverse_topm(history: list, top_m: int, min_l2_distance: float = 0.3) -> list:
    """Greedy diverse-top-M selection from (score, raw_mi, bf_idx, coef_a, coef_b) tuples; keeps entries whose joint (L2-normalized) coef vector is >= min_l2_distance from prior kept.

    Module-private; coefficient vectors of differing lengths are zero-padded to a common axis for cross-degree comparison.
    """
    if not history:
        return []
    # Secondary key on bf_idx (r[2]) so tied top-MI
    # Hermite history doesn't shift `kept[0]` across iteration orders.
    sorted_h = sorted(history, key=lambda r: (-r[0], r[2]))
    # Pad lengths to the max coef vector for cross-degree comparison.
    max_a = max(e[3].shape[0] for e in sorted_h)
    max_b = max(e[4].shape[0] for e in sorted_h)

    def _padded_vec(coef_a, coef_b):
        """Concatenate + zero-pad two coefficient vectors to a fixed ``max_a + max_b`` length for cross-degree comparison."""
        v = np.zeros(max_a + max_b, dtype=np.float64)
        v[: coef_a.shape[0]] = coef_a
        v[max_a : max_a + coef_b.shape[0]] = coef_b
        return v

    kept = [sorted_h[0]]

    kept_dirs = [unit_vector(_padded_vec(sorted_h[0][3], sorted_h[0][4]))]
    for entry in sorted_h[1:]:
        if len(kept) >= top_m:
            break
        cand_vec = _padded_vec(entry[3], entry[4])
        cand_dir = unit_vector(cand_vec)
        is_diverse = True
        for k_dir in kept_dirs:
            cos_sim = float(abs(np.dot(cand_dir, k_dir)))
            cos_sim = min(cos_sim, 1.0)  # numerical safety
            l2_dist = np.sqrt(max(2 * (1 - cos_sim), 0.0))
            if l2_dist < min_l2_distance:
                is_diverse = False
                break
        if is_diverse:
            kept.append(entry)
            kept_dirs.append(cand_dir)
    return kept
