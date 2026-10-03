"""Operand-floor check of the pair-FE joint-prevalence gate, split out of ``_pairs_score`` to keep that module under the size budget."""

from __future__ import annotations

from ._pairs_gates import _FE_JOINT_GATE_MIN_OPERAND_UPLIFT

__all__ = ["_beats_the_larger_operand"]


def _beats_the_larger_operand(best_mi: float, operand_marginal_mi, raw_vars_pair, messages, pair_mi: float = 0.0) -> bool:
    """Whether the engineered MI clears ``_FE_JOINT_GATE_MIN_OPERAND_UPLIFT`` over the larger operand's own MI.

    With one strong operand and one noise operand the pair joint MI is about the strong operand's own MI, so the joint
    gate alone passed noise-degraded copies of it (uplift 1.011-1.032); genuine pairs measured 1.36-2.32. Appends the
    reason to *messages* when rejecting and *messages* is not None. The floor only guards a pair whose partner adds nothing:
    when the raw pair's joint MI itself clears the same uplift over the larger operand, the partner carries real joint
    information and the engineered column is not a degraded copy of one operand.
    """
    floor = max(operand_marginal_mi(raw_vars_pair[0]), operand_marginal_mi(raw_vars_pair[1]))
    bar = floor * _FE_JOINT_GATE_MIN_OPERAND_UPLIFT
    if floor <= 0.0 or best_mi > bar or pair_mi > bar:
        return True
    if messages is not None:
        messages.append(
            f"joint gate operand floor: best engineered MI={best_mi:.4f} does not beat the larger operand marginal "
            f"MI={floor:.4f}; a degraded copy of one operand, not a pair feature."
        )
    return False
