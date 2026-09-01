"""The `staged` early-stopping backend was a complete no-op.

It "truncated" its snapshot with `set_params(n_estimators=best_stage)`. sklearn's gradient-boosting `predict`
walks the fitted `estimators_` array and never consults `n_estimators` after the fit, so the snapshot went on
predicting with every stage it had grown -- the wrapper returned the fully-overfit model while `best_score_` and
`n_iterations_` reported a plausible early stop.

The old code passes any test that only inspects `get_params()`. These assert on the fitted ensemble and on the
predictions, which is where the difference actually lives.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from mlframe.estimators.early_stopping import EarlyStoppingWrapper

MAX_ITER = 60


def _easy_binary(n: int = 600, seed: int = 0):
    """A problem an early stop reaches quickly, so `best_stage` is far below the budget."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 6))
    return X, (X[:, 0] + 0.3 * rng.normal(0, 1, n) > 0).astype(int)


class TestTheFittedEnsembleIsCut:
    """`n_estimators` is a hyperparameter; `estimators_` is what predicts."""

    def test_the_stage_array_is_shorter_than_the_budget(self):
        """The assertion the previous implementation could not satisfy."""
        X, y = _easy_binary()
        w = EarlyStoppingWrapper(base_model=GradientBoostingClassifier(random_state=0), max_iter=MAX_ITER, patience=3)
        w.fit(X, y)
        assert len(w.best_model_.estimators_) < MAX_ITER, "early stopping left the full ensemble in place"

    def test_the_declared_budget_matches_the_fitted_one(self):
        """A snapshot whose two halves disagree misleads any later warm-start refit."""
        X, y = _easy_binary()
        w = EarlyStoppingWrapper(base_model=GradientBoostingClassifier(random_state=0), max_iter=MAX_ITER, patience=3)
        w.fit(X, y)
        assert w.best_model_.get_params()["n_estimators"] == len(w.best_model_.estimators_)

    def test_the_predictions_differ_from_the_untruncated_model(self):
        """The end-to-end statement: an early stop must change what the model predicts."""
        X, y = _easy_binary()
        w = EarlyStoppingWrapper(base_model=GradientBoostingClassifier(random_state=0), max_iter=MAX_ITER, patience=3)
        w.fit(X, y)
        full = GradientBoostingClassifier(random_state=0, n_estimators=MAX_ITER).fit(X, y)
        assert not np.allclose(w.best_model_.predict_proba(X), full.predict_proba(X)), "the snapshot still predicts with every stage"

    def test_the_regressor_path_truncates_too(self):
        """`staged_predict` rather than `staged_predict_proba` -- the other branch of the same method."""
        rng = np.random.default_rng(1)
        n = 500
        X = rng.normal(0, 1, (n, 5))
        y = X[:, 0] * 2.0 + rng.normal(0, 0.3, n)
        w = EarlyStoppingWrapper(base_model=GradientBoostingRegressor(random_state=0), max_iter=MAX_ITER, patience=3)
        w.fit(X, y)
        assert len(w.best_model_.estimators_) < MAX_ITER

    def test_the_wrapper_still_predicts(self):
        """Slicing a fitted attribute must leave a usable estimator, not a half-built one."""
        X, y = _easy_binary()
        w = EarlyStoppingWrapper(base_model=GradientBoostingClassifier(random_state=0), max_iter=MAX_ITER, patience=3)
        w.fit(X, y)
        proba = w.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.all(np.isfinite(proba))
        assert set(np.unique(w.predict(X))) <= {0, 1}


class TestNoStopLeavesTheModelWhole:
    """The guard must not cut an ensemble that never triggered a stop."""

    def test_a_patience_larger_than_the_budget_keeps_every_stage(self):
        """With `patience > max_iter` the loop cannot stop early, so nothing should be removed."""
        X, y = _easy_binary(seed=2)
        w = EarlyStoppingWrapper(base_model=GradientBoostingClassifier(random_state=0), max_iter=10, patience=100)
        w.fit(X, y)
        assert len(w.best_model_.estimators_) == pytest.approx(w.best_model_.get_params()["n_estimators"])
        assert len(w.best_model_.estimators_) >= 1
