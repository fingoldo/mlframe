"""OOF fold policy: stratified folds for rare classes, loud skips, and fold models capped at the deployed early-stopping round."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from mlframe.training._oof_round_budget import apply_round_budget, deployed_round_budget
from mlframe.training.trainer import _compute_oof_preds


def _rare_frame(seed: int, n: int, n_pos: int):
    """Features, rare-class labels and the seed that makes a shuffled KFold strand every positive in one test fold."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_pos, replace=False)] = 1
    X = pd.DataFrame({"a": rng.normal(size=n) + 2.0 * y, "b": rng.normal(size=n)})
    return X, y


def test_rare_class_oof_uses_stratified_folds_and_succeeds():
    """3 positives in 60 rows and 3 folds: shuffled KFold at this seed strands all positives in one test fold; stratified keeps every fold fit-able."""
    X, y = _rare_frame(15, 60, 3)
    diag: dict = {}
    _, probs = _compute_oof_preds(
        model=LogisticRegression(), train_df=X, train_target=y, is_classifier_model=True, n_splits=3, random_seed=15, diagnostics=diag,
    )
    assert probs is not None and np.isfinite(probs).all()
    assert diag["fold_scheme"] == "stratified_kfold"
    assert "skipped_reason" not in diag


def test_classifier_oof_skip_is_a_warning_with_a_recorded_reason(caplog):
    """When a class is too rare to stratify and a training fold loses it, the skip is logged at WARNING and flagged in ``diagnostics``."""
    X, y = _rare_frame(11, 100, 2)
    diag: dict = {}
    with caplog.at_level(logging.INFO, logger="mlframe.training.trainer"):
        oof_preds, oof_probs = _compute_oof_preds(
            model=LogisticRegression(), train_df=X, train_target=y, is_classifier_model=True, n_splits=5, random_seed=11, diagnostics=diag,
        )
    assert oof_preds is None and oof_probs is None
    assert diag["skipped_reason"]
    assert any(r.levelno == logging.WARNING and "OOF prediction skipped" in r.getMessage() for r in caplog.records)


class _EsStub(BaseEstimator, ClassifierMixin):
    """Boosting-like classifier with an early-stopping knob that records the round budget of every fit."""

    fit_budgets: list = []

    def __init__(self, n_estimators: int = 700, early_stopping_rounds=100):
        """Store the two hyperparameters."""
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, **kw):
        """Record the budget and learn the class set."""
        type(self).fit_budgets.append((self.n_estimators, self.early_stopping_rounds))
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        """Constant two-column probabilities."""
        return np.full((len(X), 2), 0.5)

    def predict(self, X):
        """Constant class prediction."""
        return np.zeros(len(X), dtype=int)


def test_oof_fold_models_stop_at_the_deployed_best_round():
    """Fold clones run ``best_iteration_`` rounds with early stopping off, not the full configured 700."""
    X, y = _rare_frame(3, 200, 60)
    deployed = _EsStub()
    deployed.best_iteration_ = 37
    _EsStub.fit_budgets = []
    diag: dict = {}
    _, probs = _compute_oof_preds(
        model=deployed, train_df=X, train_target=y, is_classifier_model=True, n_splits=4, random_seed=0, diagnostics=diag,
    )
    assert probs is not None
    assert set(_EsStub.fit_budgets) == {(37, None)}
    assert diag["round_budget"] == 37


def test_round_budget_readers_cover_lightgbm_catboost_and_unknown():
    """``deployed_round_budget`` reads LightGBM-style 1-based and CatBoost-style 0-based best rounds; unknown models give None."""

    class _Lgb:
        """LightGBM-like: 1-based best_iteration_."""

        best_iteration_ = 12

    class _Cb:
        """CatBoost-like: 0-based get_best_iteration()."""

        @staticmethod
        def get_best_iteration():
            """Return the 0-based best round."""
            return 11

    assert deployed_round_budget(_Lgb()) == 12
    assert deployed_round_budget(_Cb()) == 12
    assert deployed_round_budget(object()) is None
    est = _EsStub()
    assert apply_round_budget(est, 5) == "n_estimators" and est.n_estimators == 5
    assert apply_round_budget(est, None) is None
