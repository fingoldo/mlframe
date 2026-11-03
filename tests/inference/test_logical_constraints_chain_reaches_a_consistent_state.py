"""Hard-mode logical constraints repeat to a fixpoint, so a rule chain ends with every implication satisfied."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.inference.logical_constraints import apply_logical_constraints


@pytest.mark.parametrize("rules", [[(2, 1), (1, 0)], [(1, 0), (2, 1)]])
def test_chain_is_consistent_in_either_declaration_order(rules):
    """tier3 <= tier2 <= tier1: one pass left [0.9, 0.1, 0.5] (tier3 above tier2); the fixpoint gives [0.9, 0.5, 0.1]."""
    preds = np.array([[0.1, 0.5, 0.9]])
    out = apply_logical_constraints(preds, rules, mode="hard")
    assert out[0, 2] <= out[0, 1] <= out[0, 0]
    assert sorted(out[0].tolist()) == [0.1, 0.5, 0.9]  # still a permutation: marginals preserved


def test_already_consistent_input_is_unchanged():
    """No violation, no change (negative control)."""
    preds = np.array([[0.9, 0.5, 0.1]])
    np.testing.assert_array_equal(apply_logical_constraints(preds, [(2, 1), (1, 0)], mode="hard"), preds)
