"""Aims at giving overfitting detection capability to models that do not support it natively."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

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
        n_val_samples = int(len(X) * self.validation_fraction)
        X_train, X_val = X[:-n_val_samples], X[-n_val_samples:]
        y_train, y_val = y[:-n_val_samples], y[-n_val_samples:]

        for i in range(1, self.max_iter + 1):
            self.base_model.partial_fit(X_train, y_train, classes=np.unique(y))
            y_pred = self.base_model.predict(X_val)
            score = self.scoring(y_val, y_pred)

            if score > self.best_score_ + self.tolerance:
                self.best_score_ = score
                self.best_model_ = self.base_model
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


# ������ �������������
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

base_model = SGDClassifier(max_iter=1, tol=None)  # max_iter=1, ����� ������������ partial_fit
early_stopping_model = EarlyStoppingWrapper(base_model, patience=5, max_iter=100)

early_stopping_model.fit(X_train, y_train)
y_pred = early_stopping_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
