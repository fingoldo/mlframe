"""Audit 2026-11-20 finding 27-F10: a NaN in any seed curve is an error, not a stability verdict.

The mean curve's argmax used to land on the NaN position, and the agreement check could then pass. The check now raises and names the
seed whose curve is non-finite.
"""

import numpy as np
import pytest

from mlframe.training._overlapping_walk_forward_cv import cv_stability_check


def test_nan_in_one_seed_curve_is_refused_and_named():
    """Seed 1 has a NaN fold: the call raises and names seed 1."""
    curves = np.array([[0.5, 0.6, 0.7, 0.6], [0.5, np.nan, 0.7, 0.6], [0.5, 0.6, 0.7, 0.6]])
    with pytest.raises(ValueError, match=r"\[1\]"):
        cv_stability_check(curves)


def test_finite_curves_still_produce_a_verdict():
    """Clean curves return the usual keys, with stable as a bool."""
    curves = np.array([[0.5, 0.6, 0.7, 0.6], [0.5, 0.61, 0.71, 0.6], [0.52, 0.6, 0.7, 0.59]])
    out = cv_stability_check(curves)
    assert isinstance(out["stable"], (bool, np.bool_))
