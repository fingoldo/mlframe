"""RMSLE must mean one thing in this repo, and must be reproducible.

`metrics.scoring.rmsle_loss` clipped negatives to 0 and kept them in the average, while `fast_rmsle` (the registry
metric) skipped them and averaged over the rest: the same predictions scored differently under the scorer and the
metric, and the skipping variant rewarded a model for pushing more rows below zero. The parallel kernel also summed
per-thread partials, so its value moved in the last ulps between identical calls.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.metrics.regression._regression_extras import fast_rmsle
from mlframe.metrics.scoring import rmsle_loss


@pytest.mark.parametrize("n", [500, 300_000])
def test_scorer_and_metric_agree_with_negative_rows(n):
    rng = np.random.default_rng(0)
    y_true = rng.gamma(2.0, 2.0, n)
    y_pred = y_true + rng.normal(0, 2.0, n)  # some predictions go negative
    assert (y_pred < 0).any()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert fast_rmsle(y_true, y_pred) == pytest.approx(rmsle_loss(y_true, y_pred), rel=1e-12)


def test_pushing_rows_negative_is_not_rewarded():
    """Under the skip rule a model that moved a badly-predicted row below zero removed it from the average."""
    y_true = np.array([1.0, 2.0, 100.0])
    good_rows_bad_tail = np.array([1.0, 2.0, 1.0])
    same_but_tail_negative = np.array([1.0, 2.0, -1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert fast_rmsle(y_true, same_but_tail_negative) >= fast_rmsle(y_true, good_rows_bad_tail)


def test_the_parallel_path_is_reproducible():
    rng = np.random.default_rng(1)
    n = 400_000
    y_true = rng.gamma(2.0, 2.0, n)
    y_pred = y_true * rng.lognormal(0, 0.2, n)
    assert len({repr(fast_rmsle(y_true, y_pred)) for _ in range(5)}) == 1


def test_negative_rows_still_warn():
    with pytest.warns(RuntimeWarning, match="clipped to 0"):
        fast_rmsle(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), np.array([1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.5]))
