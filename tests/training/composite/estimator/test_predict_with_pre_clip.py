"""``predict_with_pre_clip`` gives the clipped and the pre-clip prediction from one inner predict, identical to the two calls.

The wrap pass scored ``predict`` and ``predict_pre_clip`` separately: two full inner predicts and inverses per (entry, split)
for one clip apart. The combined call must return the same arrays and leave the same clip counters in ``runtime_stats_``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator


def _fitted():
    """A wrapper fitted on a narrow y range, and a frame whose feature pushes predictions beyond it (so the clip bites)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"b": rng.uniform(1.0, 5.0, 300), "x": rng.normal(size=300)})
    y = X["b"].to_numpy() + X["x"].to_numpy()
    est = CompositeTargetEstimator(base_estimator=LinearRegression(), transform_name="diff", base_column="b").fit(X, y)
    est.fitted_params_["y_clip_low"], est.fitted_params_["y_clip_high"] = 0.0, 5.0  # a narrow envelope, so the clip bites
    X_new = pd.DataFrame({"b": np.linspace(1.0, 5.0, 50), "x": np.linspace(-9.0, 9.0, 50)})  # bases in range, x far out
    return est, X_new


def test_one_call_equals_predict_and_predict_pre_clip():
    """Same clipped and pre-clip arrays; the clip really moved some rows here."""
    est, X = _fitted()
    clipped, pre = est.predict_with_pre_clip(X)
    np.testing.assert_array_equal(clipped, est.predict(X))
    np.testing.assert_array_equal(pre, est.predict_pre_clip(X))
    assert np.any(clipped != pre), "the fixture must make the clip bite"


def test_the_runtime_counters_match_a_plain_predict():
    """The combined call records exactly what predict records."""
    a, X = _fitted()
    b, _ = _fitted()
    a.predict(X)
    b.predict_with_pre_clip(X)
    assert dict(a.runtime_stats_) == dict(b.runtime_stats_)


def test_the_combined_call_runs_the_inner_once():
    """One inner predict, where predict + predict_pre_clip ran two."""
    est, X = _fitted()
    calls = []
    real = est.estimator_.predict
    est.estimator_.predict = lambda Z: calls.append(1) or real(Z)
    est.predict_with_pre_clip(X)
    assert len(calls) == 1
