"""``ICE.evaluate`` must not report an unweighted score for a weighted fit.

The metric used to take CatBoost's ``weight`` argument and drop it (``output_weight = 1  # weight is not used``), so a
fit with ``sample_weight`` early-stopped on a number describing a different objective than the one it minimised. The
kernels have no per-row weighting, so the metric refuses instead of guessing.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._ice_metric import ICE


def _metric(y_true, y_score):
    return 0.5


def _ice() -> ICE:
    return ICE(metric=_metric, higher_is_better=False)


def _approxes(n: int):
    rng = np.random.default_rng(0)
    return [rng.normal(size=n)], (rng.random(n) < 0.3).astype(np.int8)


def test_uniform_weights_are_accepted():
    approxes, target = _approxes(200)
    value, _ = _ice().evaluate(approxes, target, np.full(200, 2.5))
    assert value == pytest.approx(0.5)


def test_absent_weights_are_accepted():
    approxes, target = _approxes(200)
    value, _ = _ice().evaluate(approxes, target, None)
    assert value == pytest.approx(0.5)


def test_non_uniform_weights_are_refused():
    approxes, target = _approxes(200)
    w = np.full(200, 1.0)
    w[7] = 9.0
    with pytest.raises(ValueError, match="does not support per-row sample weights"):
        _ice().evaluate(approxes, target, w)


def test_the_weight_vector_is_inspected_only_once_per_fit():
    """The check runs on the first call; a fit cannot change its weights midway, and re-checking every iteration
    would put an O(n) pass on the metric hot path."""
    metric = _ice()
    approxes, target = _approxes(200)
    metric.evaluate(approxes, target, np.ones(200))
    w = np.ones(200)
    w[3] = 4.0
    metric.evaluate(approxes, target, w)  # no raise: already checked
