"""Cluster 27 regression: the dichotomic ES path refits on the full train and keeps float history (audit 2026-11-20, finding 27-F13).

The docstrings promise a final model that saw every train row. With an internal val split the code kept the candidate fitted on the train
part only, and ``history`` held ``(budget, (score, estimator))`` tuples although it is declared a list of floats.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training._partial_fit_es_wrapper import PartialFitESWrapper


class _Budgeted(RegressorMixin, BaseEstimator):
    """A regressor without partial_fit whose budget is ``n_iter``; it records how many rows it was fitted on."""

    def __init__(self, n_iter=1):
        self.n_iter = n_iter

    def fit(self, X, y):
        """Remember the fit size and the mean target."""
        self.n_fit_ = len(X)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        """Predict the fitted mean, shrunk by a budget-dependent amount so budgets score differently."""
        return np.full(len(X), self.mean_ * (1 - 1.0 / (self.n_iter + 1)))


def _data(n=200):
    """Rows with a positive target."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 3)), rng.uniform(1, 2, n)


def test_internal_split_ends_with_a_model_fitted_on_all_rows_and_float_history():
    """The kept estimator saw all 200 rows, and history is a flat list of floats with matching budgets."""
    X, y = _data()
    w = PartialFitESWrapper(_Budgeted(), is_classification=False, budget_param="n_iter", budget_min=1, budget_max=16, val_size=0.2, random_state=0)
    w.fit(X, y)
    assert w.estimator.n_fit_ == 200
    assert all(isinstance(v, float) for v in w.history)
    assert len(w.history) == len(w.history_budgets) and w.best_iter in w.history_budgets
    assert w.estimator.n_iter == w.best_iter


def test_external_val_keeps_the_candidate_fitted_on_the_full_train_set():
    """With an external val set X is already the whole train set, so no extra refit is needed and the fit size is len(X)."""
    X, y = _data(150)
    Xv, yv = _data(50)
    w = PartialFitESWrapper(_Budgeted(), is_classification=False, budget_param="n_iter", budget_min=1, budget_max=8, external_X_val=Xv, external_y_val=yv)
    w.fit(X, y)
    assert w.estimator.n_fit_ == 150
