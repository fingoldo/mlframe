"""Operand-floor check of the pair-FE joint-prevalence gate, split out of ``_pairs_score`` to keep that module under the size budget."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mutual_info_score

__all__ = ["_beats_the_larger_operand", "operand_uplift_null_bar"]


def operand_uplift_null_bar(codes_a, codes_b, codes_y, floor: float, n_perm: int = 20, quantile: float = 0.95, seed: int = 0) -> float:
    """Data-driven uplift bar over the larger operand: the null quantile of joint-MI uplift when the partner is noise.

    The partner's discretised codes are permuted, which keeps its marginal and destroys its relation to y. The joint MI
    of that noise pair, divided by *floor*, is the uplift a plug-in estimate reaches by chance; the bar is its *quantile*
    and never below 1.0. Returns 1.0 when the operand codes are unavailable or the floor is not positive.
    """
    if codes_a is None or codes_b is None or floor <= 0.0:
        return 1.0
    codes_a = np.asarray(codes_a, dtype=np.int64)
    codes_b = np.asarray(codes_b, dtype=np.int64)
    n_b = int(codes_b.max()) + 1 if codes_b.size else 1
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)  # Intentional float64: statistic vector (MI / null max / p-value inputs), exact permutation ties
    for i in range(n_perm):
        joint = codes_a * n_b + codes_b[rng.permutation(codes_b.shape[0])]
        null[i] = mutual_info_score(joint, codes_y) / floor
    return max(1.0, float(np.quantile(null, quantile)))


def _beats_the_larger_operand(best_mi: float, operand_marginal_mi, raw_vars_pair, messages, pair_mi: float = 0.0, uplift_bar: float = 1.0) -> bool:
    """Whether the engineered MI clears *uplift_bar* over the larger operand's own MI.

    With one strong operand and one noise operand the pair joint MI is about the strong operand's own MI, so the joint
    gate alone passed noise-degraded copies of it (uplift 1.011-1.032); genuine pairs measured 1.36-2.32. Appends the
    reason to *messages* when rejecting and *messages* is not None. The floor only guards a pair whose partner adds nothing:
    when the raw pair's joint MI itself clears the same uplift over the larger operand, the partner carries real joint
    information and the engineered column is not a degraded copy of one operand.
    """
    floor = max(operand_marginal_mi(raw_vars_pair[0]), operand_marginal_mi(raw_vars_pair[1]))
    bar = floor * uplift_bar
    if floor <= 0.0 or best_mi > bar or pair_mi > bar:
        return True
    if messages is not None:
        messages.append(
            f"joint gate operand floor: best engineered MI={best_mi:.4f} does not beat the larger operand marginal "
            f"MI={floor:.4f}; a degraded copy of one operand, not a pair feature."
        )
    return False
