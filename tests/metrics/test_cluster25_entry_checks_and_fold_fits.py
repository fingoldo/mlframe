"""Entry checks, fold-local fits and degenerate-input handling from audit 2026-11-20 cluster 25 (findings 25-F12..F19)."""

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin

from mlframe.feature_selection.stochastic_bandit_selection import stochastic_bandit_selection
from mlframe.feature_selection.zero_importance_pruning import iterative_zero_importance_pruning
from mlframe.inference.explainability import _fit_fold_model
from mlframe.metrics._core_precision_mape import maximum_absolute_percentage_error
from mlframe.metrics._drift import wasserstein_1d


def test_wasserstein_drop_of_non_finite_values_is_logged(caplog):
    """Non-finite values are still dropped, but the drop is reported with counts (25-F12)."""
    ref = np.array([0.0, 1.0, 2.0, 3.0])
    tgt = np.array([0.0, np.nan, 2.0, 3.0])
    with caplog.at_level(logging.WARNING, logger="mlframe.metrics._drift"):
        value = wasserstein_1d(ref, tgt)
    assert np.isfinite(value)
    assert any("non-finite" in rec.getMessage() for rec in caplog.records)


def test_mape_refuses_length_mismatch():
    """The parallel and serial kernels index y_pred unchecked, so a mismatch must raise (25-F13)."""
    with pytest.raises(ValueError):
        maximum_absolute_percentage_error(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_xgboost_contrib_kwargs_carry_the_model_missing_sentinel():
    """The interaction path must use the model's missing sentinel and categorical flag, as predict does (25-F14)."""
    xgb = pytest.importorskip("xgboost")
    from mlframe.inference.native_gpu_shap import native_contrib_kwargs

    X = np.array([[1.0, 2.0], [-999.0, 3.0], [4.0, -999.0], [5.0, 6.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    model = xgb.XGBRegressor(n_estimators=3, missing=-999.0).fit(X, y)
    matrix_kwargs, _predict_kwargs = native_contrib_kwargs(model)
    assert matrix_kwargs.get("missing") == -999.0


class _RecordingEstimator(BaseEstimator, ClassifierMixin):
    """Remembers how many rows each fit received."""

    def fit(self, X, y, **kwargs):
        """Record the training row count and classes."""
        self.n_fit_rows_ = len(X)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict class zero for every row."""
        return np.zeros(len(X), dtype=int)


def test_eval_fraction_fit_excludes_the_test_fold():
    """The fold's held-out rows must not reach the fit when eval_fraction is set (25-F15)."""
    X_train = pd.DataFrame({"a": np.arange(80.0)})
    y_train = pd.Series(np.zeros(80, dtype=int))
    X_test = pd.DataFrame({"a": np.arange(80.0, 100.0)})
    y_test = pd.Series(np.zeros(20, dtype=int))
    pipe = Pipeline([("est", _RecordingEstimator())])
    _fit_fold_model(pipe, Pipeline, X_train, X_test, y_train, y_test, {"eval_fraction": 0.2}, plot=False)
    assert pipe.named_steps["est"].n_fit_rows_ == 80


def test_optimizer_etr_does_not_receive_the_catboost_iterations_key():
    """ExtraTreesRegressor would receive ``iterations=150`` as a kwarg and fail; ETR defaults to no extra params (25-F16)."""
    from mlframe.models._optimization_search import MBHOptimizer

    opt = MBHOptimizer(search_space=np.arange(30), model_name="ETR")
    assert "iterations" not in (opt.model_params or {})


def test_preserve_rank_order_is_monotone_within_each_group_on_many_groups():
    """With many small groups the per-group isotonic refit keeps each group in its own prediction order (25-F17)."""
    from mlframe.inference.group_zero_sum_constraint import apply_group_zero_sum_constraint

    rng = np.random.default_rng(1)
    n = 400
    preds = rng.standard_normal(n)
    groups = rng.integers(0, 150, size=n)
    out = apply_group_zero_sum_constraint(preds, groups, target_sum=0.0, preserve_rank_order=True)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        order = idx[np.argsort(preds[idx], kind="stable")]
        assert np.all(np.diff(out[order]) >= -1e-9)


def test_zero_importance_candidates_are_judged_on_training_rows_only():
    """Each importance is computed on a training fold, never on all rows (25-F18)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((100, 4)), columns=list("abcd"))
    y = (X["a"] > 0).astype(int).to_numpy()
    seen_rows = []

    def importance_fn(fitted, X_round, y_round):
        """Record the row count of each importance call and return uniform importances."""
        seen_rows.append(len(X_round))
        return np.ones(X_round.shape[1])

    iterative_zero_importance_pruning(
        RandomForestClassifier(n_estimators=5, random_state=0), X, y, accuracy_score, cv=KFold(n_splits=4, shuffle=True, random_state=0),
        importance_fn=importance_fn, max_rounds=1,
    )
    assert seen_rows and set(seen_rows) == {75}


def test_bandit_warns_when_no_epoch_scores_finite():
    """A NaN CV score is skipped, and if none is finite the caller is told the subset is arbitrary (25-F19)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((60, 5)), columns=list("abcde"))
    y = (X["a"] > 0).astype(int).to_numpy()

    def nan_scoring(y_true, y_pred):
        """Every score is NaN."""
        return float("nan")

    with pytest.warns(RuntimeWarning, match="finite"):
        stochastic_bandit_selection(RandomForestClassifier(n_estimators=5, random_state=0), X, y, nan_scoring, subset_size=2, n_epochs=3)
