"""Aims at giving overfitting detection capability to models that do not support it natively."""

from __future__ import annotations


# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

import copy

from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

from pyutilz.pythonlib import store_params_in_object, get_parent_func_args


class EarlyStoppingWrapper(BaseEstimator):
    def __init__(
        self,
        base_model: object,
        start_iter: int = 100,
        max_iter: int = None,
        # stopping conditions
        max_runtime_mins: float = None,
        patience: int = 5,
        tolerance: float = 0.0,
        # CV
        validation_fraction: float = 0.1,
        scoring: Callable = accuracy_score,
    ):
        store_params_in_object(obj=self, params=get_parent_func_args())

    def fit(self, X, y):
        self.best_score_ = -np.inf
        self.best_model_ = None
        self.no_improvement_count_ = 0

        # Support both classifiers and regressors. ``partial_fit`` takes a ``classes=`` kwarg ONLY on
        # classifiers (it errors on regressors like SGDRegressor / PassiveAggressiveRegressor), and the
        # default ``accuracy_score`` is meaningless for regression -- swap to R^2 (still greater-is-better,
        # so the ``score > best_score_`` improvement logic is unchanged) unless the caller passed an
        # explicit scorer.
        from sklearn.base import is_regressor
        self._is_regressor = is_regressor(self.base_model)
        scoring = self.scoring
        if self._is_regressor and scoring is accuracy_score:
            from sklearn.metrics import r2_score
            scoring = r2_score

        # Wave 24 P0 (2026-05-20): pre-fix int(len(X) * frac) could round
        # down to 0 on small X (e.g. len=9, frac=0.1 -> int(0.9)=0). Then
        # ``X[:-0]`` is Python's "slice from start to before index 0",
        # i.e. an EMPTY array -- training silently collapsed with no
        # exception, no warning. Same for X_val (``X[-0:]`` returns the
        # WHOLE array as val, but training was already dead).
        # Guard explicitly: clamp to >=1 and require at least one
        # training row (len(X) - n_val_samples >= 1).
        n_val_samples = max(1, int(len(X) * self.validation_fraction))
        if n_val_samples >= len(X):
            raise ValueError(
                f"early-stopping: validation_fraction={self.validation_fraction} "
                f"with len(X)={len(X)} leaves zero training rows "
                f"(n_val_samples={n_val_samples}). Use a smaller "
                f"validation_fraction or more samples."
            )
        X_train, X_val = X[:-n_val_samples], X[-n_val_samples:]
        y_train, y_val = y[:-n_val_samples], y[-n_val_samples:]

        pf_kwargs = {} if self._is_regressor else {"classes": np.unique(y)}
        for i in range(1, self.max_iter + 1):
            self.base_model.partial_fit(X_train, y_train, **pf_kwargs)
            y_pred = self.base_model.predict(X_val)
            score = scoring(y_val, y_pred)

            if score > self.best_score_ + self.tolerance:
                self.best_score_ = score
                # Snapshot the model AT the best iteration. Storing the live
                # base_model reference is wrong: partial_fit keeps mutating it in
                # later iterations, so best_model_ would hold the FINAL (often
                # degraded) weights, not the best ones -- silently defeating ES.
                self.best_model_ = copy.deepcopy(self.base_model)
                self.no_improvement_count_ = 0
            else:
                self.no_improvement_count_ += 1

            if self.no_improvement_count_ >= self.patience:
                print(f"Early stopping at iteration {i}")
                break

        return self

    def predict(self, X):
        return self.best_model_.predict(X)

    def predict_proba(self, X):
        return self.best_model_.predict_proba(X)


# Demo / smoke test kept for reference. Guarded behind __main__ so it no longer
# trains a model + prints to stdout at import time.
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    base_model = SGDClassifier(max_iter=1, tol=None)  # max_iter=1 so partial_fit drives iteration
    early_stopping_model = EarlyStoppingWrapper(base_model, patience=5, max_iter=100)

    early_stopping_model.fit(X_train, y_train)
    y_pred = early_stopping_model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
