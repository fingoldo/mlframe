"""Tweedie deviance rejects zero targets for power > 2 (no negative deviance), and ordinal cutpoints reject labels outside [0, n_classes)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.classification._ordinal_cutpoints import optimal_ordinal_cutpoints
from mlframe.metrics.regression._regression_deviance import fast_tweedie_deviance


def test_power_three_with_a_zero_target_is_not_a_negative_deviance():
    """fast_tweedie_deviance([0.0, 1.0], [1.0, 1.0], power=3): the y=0 row is outside the support; the result must not be negative."""
    got = fast_tweedie_deviance(np.array([0.0, 1.0]), np.array([1.0, 1.0]), power=3.0)
    assert np.isnan(got) or got >= 0.0


def test_poisson_range_power_keeps_zero_targets():
    """For 1 < power < 2 a zero target is valid (negative control)."""
    got = fast_tweedie_deviance(np.array([0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.5]), power=1.5)
    assert np.isfinite(got) and got >= 0.0


def test_ordinal_label_outside_the_class_range_raises():
    """n_classes=3 but a label 3 grew the bincount and the warm start; now a clear ValueError."""
    with pytest.raises(ValueError, match="labels must lie in"):
        optimal_ordinal_cutpoints(np.array([0, 1, 2, 3]), np.array([0.1, 0.5, 1.5, 2.5]), n_classes=3)
