"""``ICE.evaluate`` must say so when it scores a weighted fit with an unweighted metric.

The metric took CatBoost's ``weight`` argument and dropped it (``output_weight = 1  # weight is not used``), so a fit
with ``sample_weight`` early-stopped on a number describing a different objective than the one it minimised, silently.
No ICE kernel takes per-row weights yet - making them weight-aware is the real fix - so the divergence is announced
once per fit. Refusing the fit instead was tried and reverted: weighted fits (fairness / inverse-frequency weighting)
are a supported path, and killing them is worse than scoring them with a named caveat.
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


def test_non_uniform_weights_are_announced_once(caplog):
    import logging

    approxes, target = _approxes(200)
    w = np.full(200, 1.0)
    w[7] = 9.0
    metric = _ice()
    with caplog.at_level(logging.WARNING):
        value, _ = metric.evaluate(approxes, target, w)
    assert value == pytest.approx(0.5), "the fit must proceed: a weighted fit is a supported path"
    assert "UNWEIGHTED" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        metric.evaluate(approxes, target, w)
    assert "UNWEIGHTED" not in caplog.text, "once per fit, not once per iteration"


def test_the_weight_vector_is_inspected_only_once_per_fit():
    """The check runs on the first call; a fit cannot change its weights midway, and re-checking every iteration
    would put an O(n) pass on the metric hot path."""
    metric = _ice()
    approxes, target = _approxes(200)
    metric.evaluate(approxes, target, np.ones(200))
    w = np.ones(200)
    w[3] = 4.0
    metric.evaluate(approxes, target, w)  # already checked: no second pass over the vector
