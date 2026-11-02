"""Harmonic blends: a zero-weight member never affects the result, and the streaming path agrees with the materialised one on a zero prediction."""

from __future__ import annotations

import numpy as np

from mlframe.models.ensembling.base import _harmonic_mean_stacked
from mlframe.models.ensembling.predict import _finalize_mean_of_transformed, _push_transformed


def test_zero_weight_member_predicting_zero_does_not_zero_the_blend():
    """weights [1, 0]: member 1 predicts exactly 0 but carries no weight, so the blend is member 0's 0.8."""
    stacked = np.array([[0.8, 0.6], [0.0, 0.4]])
    out = _harmonic_mean_stacked(stacked, np.array([1.0, 0.0]))
    np.testing.assert_allclose(out, [0.8, 0.6])


def test_positive_weight_member_predicting_zero_still_gives_zero():
    """A member with weight that predicts 0 keeps the harmonic mean at 0 (negative control)."""
    out = _harmonic_mean_stacked(np.array([[0.8], [0.0]]), np.array([0.5, 0.5]))
    assert out[0] == 0.0


class _Acc:
    """Minimal mean accumulator standing in for the Welford one."""

    def __init__(self):
        self.vals = []

    def push(self, x):
        """Store a pushed array."""
        self.vals.append(np.asarray(x, dtype=np.float64))


def test_streaming_harmonic_mean_with_a_zero_prediction_is_near_zero_like_the_materialised_path():
    """Members [0.0, 0.5]: the materialised harmonic mean is 0; the streaming one used to give 1.0."""
    acc = _Acc()
    for member in (np.array([0.0]), np.array([0.5])):
        _push_transformed(acc, member, "harm")
    streamed = _finalize_mean_of_transformed(np.mean(acc.vals, axis=0), "harm")
    assert streamed[0] < 1e-9
