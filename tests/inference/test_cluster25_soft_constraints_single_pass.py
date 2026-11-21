"""Audit 2026-11-20 finding 25-F27: soft mode is one ordered pass, so a chain can stay violated, while hard mode repeats.

The docstring said both repeat to a fixpoint. The example from the audit shows the difference: one soft pass leaves the rule
(2, 1) violated, and hard mode, which repeats, resolves it.
"""

import numpy as np

from mlframe.inference.logical_constraints import apply_logical_constraints

RULES = [(2, 1, 1.0), (1, 0, 1.0)]
ROW = np.array([[0.2, 0.5, 0.9]])


def _violates(out, child, parent):
    """True where the child column exceeds the parent column."""
    return bool(out[0, child] > out[0, parent])


def test_soft_single_pass_leaves_the_chain_violated():
    """One soft pass returns [0.9, 0.2, 0.5]: rule (2, 1) is still violated."""
    out = apply_logical_constraints(ROW, RULES, mode="soft")
    assert _violates(out, 2, 1)


def test_hard_mode_repeats_and_resolves_the_chain():
    """Hard mode repeats its pass until stable, so no rule is violated."""
    out = apply_logical_constraints(ROW, [(2, 1), (1, 0)], mode="hard")
    assert not _violates(out, 2, 1)
    assert not _violates(out, 1, 0)
