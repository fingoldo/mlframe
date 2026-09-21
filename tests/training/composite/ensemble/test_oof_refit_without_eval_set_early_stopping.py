"""An early-stopping booster refit for the ensemble OOF without an eval set must train, not raise and drop out.

The OOF carve skips its eval slice below 1000 rows (and when a group carve is impossible), but the cloned booster kept
``early_stopping_rounds``, so LightGBM raised "For early stopping, at least one dataset and eval metric is required" and
every LightGBM component was excluded from the cross-target ensemble on such folds. With no eval slice the clone now
trains the deployed model's best iteration count with early stopping off.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from mlframe.training.composite.ensemble import _maybe_pass_sample_weight
from mlframe.training.composite.ensemble._oof_split import _best_iteration_of, _disable_early_stopping

lgb = pytest.importorskip("lightgbm")


def _early_stopped_model(seed: int = 0):
    """A LightGBM regressor fitted with early stopping on an eval set, plus data for refits."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(600, 5))
    y = 2.0 * x[:, 0] + rng.normal(size=600)
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.1, early_stopping_rounds=20, verbose=-1)
    model.fit(x[:400], y[:400], eval_set=[(x[400:], y[400:])])
    return model, x, y


def test_refitting_a_clone_without_an_eval_set_trains():
    """The OOF refit path with no eval slice: no exception, and the clone learns the target."""
    model, x, y = _early_stopped_model()
    refit = clone(model)
    _maybe_pass_sample_weight(refit, x[:300], y[:300], None, eval_set=None, fitted_source=model)
    preds = refit.predict(x[300:])
    assert np.corrcoef(preds, y[300:])[0, 1] > 0.8


def test_the_clone_trains_the_deployed_round_count():
    """With no eval data, the honest stand-in for early stopping is the round the deployed model stopped at."""
    model, _, _ = _early_stopped_model()
    best = _best_iteration_of(model)
    assert best is not None and 0 < best < 500
    refit = clone(model)
    _disable_early_stopping(refit, model)
    params = refit.get_params()
    assert params["early_stopping_rounds"] is None
    assert params["n_estimators"] == best


def test_a_model_without_early_stopping_is_left_alone():
    """Nothing to disable: the clone's parameters stay exactly as they were."""
    plain = lgb.LGBMRegressor(n_estimators=77, verbose=-1)
    before = plain.get_params()
    _disable_early_stopping(plain, None)
    assert plain.get_params() == before


def test_an_eval_set_keeps_early_stopping_on():
    """When the carve does provide eval data, early stopping stays as configured."""
    model, x, y = _early_stopped_model()
    refit = clone(model)
    _maybe_pass_sample_weight(refit, x[:300], y[:300], None, eval_set=(x[300:400], y[300:400]), fitted_source=model)
    assert refit.get_params()["early_stopping_rounds"] == 20
