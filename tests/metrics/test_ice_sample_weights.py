"""``ICE.evaluate`` scores a weighted fit with the fit's own per-row weights.

The metric took CatBoost's ``weight`` argument and dropped it, so a fit with ``sample_weight`` early-stopped on a number
describing a different objective than the one it minimised. The weights now reach the metric; equal weights change
nothing and are not passed; a custom metric that takes no ``sample_weight`` is scored unweighted and says so once per fit.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.metrics._ice_metric import ICE


def _approxes(n: int):
    """CatBoost-style approxes and a binary target of ``n`` rows."""
    rng = np.random.default_rng(0)
    return [rng.normal(size=n)], (rng.random(n) < 0.3).astype(np.int8)


class _Recorder:
    """A metric that records the weights it received."""

    def __init__(self):
        self.seen = []

    def __call__(self, y_true, y_score, sample_weight=None):
        self.seen.append(sample_weight)
        return 0.5


def test_non_uniform_weights_reach_the_metric():
    """Weights that differ between rows are passed through to the wrapped metric unchanged."""
    rec = _Recorder()
    approxes, target = _approxes(200)
    w = np.ones(200)
    w[7] = 9.0
    value, out_w = ICE(metric=rec, higher_is_better=False).evaluate(approxes, target, w)
    assert value == pytest.approx(0.5) and out_w == 1
    np.testing.assert_array_equal(rec.seen[-1], w)


@pytest.mark.parametrize("weight", [None, np.full(200, 2.5)])
def test_absent_or_equal_weights_are_not_passed(weight):
    """No weights, or all-equal ones, change nothing, so the metric is called without them."""
    rec = _Recorder()
    approxes, target = _approxes(200)
    ICE(metric=rec, higher_is_better=False).evaluate(approxes, target, weight)
    assert rec.seen == [None]


def test_a_metric_without_sample_weight_is_scored_unweighted_and_says_so_once(caplog):
    """A metric that cannot take weights is scored unweighted, with one warning rather than one per call."""
    def plain(y_true, y_score):
        """A metric with no sample_weight parameter."""
        return 0.25

    approxes, target = _approxes(200)
    w = np.ones(200)
    w[3] = 4.0
    ice = ICE(metric=plain, higher_is_better=False)
    with caplog.at_level(logging.WARNING):
        assert ice.evaluate(approxes, target, w)[0] == pytest.approx(0.25)
    assert "takes no sample_weight" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        ice.evaluate(approxes, target, w)
    assert "takes no sample_weight" not in caplog.text, "once per fit, not once per iteration"
