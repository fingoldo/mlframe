"""Cross-split regression verdict must not reward a split for simply having a smaller target.

A production chart said GENERALIZES on a raw val->test RMSE ratio of 0.30x while R^2 fell from 0.02 to -0.90: test's
target mean was 0.26 against val's 1.26, so its raw RMSE shrank with it.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.split_comparison import overfit_verdict


def test_smaller_target_with_collapsed_fit_is_not_green():
    rng = np.random.default_rng(0)
    y_val = rng.lognormal(1.0, 1.0, 5000)
    p_val = y_val * 0.2 + y_val.mean() * 0.8  # weak but centred
    y_test = rng.lognormal(-1.0, 1.0, 5000)
    p_test = np.full_like(y_test, y_val.mean())  # biased far above the smaller test target
    v = overfit_verdict(task="regression", per_split={"val": {"y_true": y_val, "y_pred": p_val}, "test": {"y_true": y_test, "y_pred": p_test}})
    assert v.color != "green"
    assert "std(y)" in v.reason


def test_training_curve_sampled_train_series_spans_the_run():
    """CatBoost logs learn every k iterations; a 52-point series for 259 iterations at k=5 lacks the final point and used to stop at ~n/k."""
    from mlframe.reporting.charts.training_curve import _sampled_positions

    pos = _sampled_positions(52, 259, 5)  # the production shape: grid without the final iteration
    assert pos is not None and pos[0] == 0 and pos[-1] == 255
    assert _sampled_positions(53, 259, 5)[-1] == 258
    assert _sampled_positions(52, 259, None) is None  # unknown period: never stretched
