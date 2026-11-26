"""Cluster 22 regression: a floored row sums to 1 even when the input row did not (audit 2026-11-20, finding 22-F14).

The rest-of-row mass was ``1 - active_prob``, which assumes the row already sums to 1. A row of mass 0.3 floored to 0.5 came out summing to 0.611.
"""

import numpy as np

from mlframe.calibration.sticky_state_persistence_floor import apply_sticky_state_persistence_floor


def test_row_that_does_not_sum_to_one_is_renormalised_to_one():
    """[0.1, 0.2] with the active class 0 and floor 0.5 becomes [0.5, 0.5], summing to 1 (22-F14)."""
    out = apply_sticky_state_persistence_floor(np.array([[0.1, 0.2]]), np.array([0]), 0.5)
    np.testing.assert_allclose(out, [[0.5, 0.5]])
    assert np.isclose(out.sum(), 1.0)


def test_row_that_already_sums_to_one_keeps_the_old_result():
    """A properly normalised row is floored exactly as before: the active class gets the floor and the rest is scaled (22-F14)."""
    out = apply_sticky_state_persistence_floor(np.array([[0.1, 0.3, 0.6]]), np.array([0]), 0.5)
    np.testing.assert_allclose(out, [[0.5, 0.5 * 0.3 / 0.9, 0.5 * 0.6 / 0.9]])
