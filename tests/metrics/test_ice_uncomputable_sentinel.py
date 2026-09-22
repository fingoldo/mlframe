"""A term ICE cannot compute must score the WORST value, never the best.

ICE is lower-is-better. A NaN loss term (one iteration emitting a probability slightly above 1.0 makes Brier NaN) used
to be substituted with 0.0, which is the best possible contribution, so the broken iteration beat its healthy
neighbours and won early stopping. The skip sentinel of ``ICE.evaluate`` had the same shape: a skipped EVAL set scored
0 on every iteration and froze selection at iteration 1. CatBoost rejects a non-finite custom metric value ("JSON
writer: invalid float value"), so the worst value has to be finite.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._ice_metric import ICE
from mlframe.metrics.calibration import ICE_UNCOMPUTABLE, integral_calibration_error_from_metrics

_HEALTHY = dict(calibration_mae=0.05, calibration_std=0.02, calibration_coverage=1.0, brier_loss=0.1, roc_auc=0.8, pr_auc=0.4)


def test_sentinel_is_finite():
    assert np.isfinite(ICE_UNCOMPUTABLE)


@pytest.mark.parametrize("term", ["brier_loss", "calibration_mae", "calibration_std", "calibration_coverage"])
def test_a_nan_loss_term_scores_worst(term):
    healthy = float(integral_calibration_error_from_metrics(**_HEALTHY))
    broken = float(integral_calibration_error_from_metrics(**{**_HEALTHY, term: float("nan")}))
    assert broken == ICE_UNCOMPUTABLE
    assert broken > healthy, "a term that cannot be computed must never beat a healthy iteration"


@pytest.mark.parametrize("term", ["roc_auc", "pr_auc"])
def test_a_nan_reward_term_only_forfeits_its_reward(term):
    healthy = float(integral_calibration_error_from_metrics(**_HEALTHY))
    no_reward = float(integral_calibration_error_from_metrics(**{**_HEALTHY, term: float("nan")}))
    assert np.isfinite(no_reward) and no_reward > healthy


def _evaluate(metric, n, seed=0):
    rng = np.random.default_rng(seed)
    return metric.evaluate([rng.normal(size=n)], (rng.random(n) < 0.3).astype(np.int8), None)[0]


def test_skip_sentinel_is_the_worst_value_not_zero():
    metric = ICE(metric=lambda y_true, y_score: 0.25, higher_is_better=False, max_arr_size=100)
    assert _evaluate(metric, 50) == 0.25
    assert _evaluate(metric, 500) == ICE_UNCOMPUTABLE


def test_skip_sentinel_follows_the_metric_direction():
    metric = ICE(metric=lambda y_true, y_score: 0.25, higher_is_better=True, max_arr_size=100)
    assert _evaluate(metric, 500) == -ICE_UNCOMPUTABLE
