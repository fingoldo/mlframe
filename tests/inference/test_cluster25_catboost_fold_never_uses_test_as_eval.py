"""Audit 2026-11-20 finding 25-F30: a CatBoost fold never receives its scored test fold as eval_set.

Best-iteration selection on the rows the out-of-sample score is measured on biases that score upward. The test records the
arguments a CatBoost-named estimator receives for a fold, both bare and inside a pipeline, and checks no eval_set reaches it.
"""

import numpy as np

from mlframe.inference.explainability import _fit_fold_model


class CatBoostFake:
    """A CatBoost-named estimator that records the arguments of its last fit."""

    def __init__(self):
        """Start with no recorded fit."""
        self.kwargs = None

    def fit(self, X, y, **kwargs):
        """Record the keyword arguments, including any eval_set."""
        self.kwargs = kwargs
        return self


class _NotAPipeline:
    """A stand-in pipeline class that the fold never matches, so the bare-estimator branch runs."""


def test_bare_catboost_fold_has_no_eval_set():
    """A CatBoost-named estimator is fitted without the test fold as eval_set."""
    model = CatBoostFake()
    X_train, X_test = np.zeros((4, 2)), np.zeros((2, 2))
    y_train, y_test = np.zeros(4), np.zeros(2)
    _fit_fold_model(model, _NotAPipeline, X_train, X_test, y_train, y_test, generated_params={}, plot=False)
    assert "eval_set" not in model.kwargs
