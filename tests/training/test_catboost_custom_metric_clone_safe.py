"""Regression tests for the CatBoost ``custom_metric`` / ``sklearn.clone()`` dead-wiring fix.

Bug: ``get_training_configs``'s ``CB_CLASSIF``/``CB_REGR`` dicts had their ``custom_metric`` assignment
commented out, because passing ``custom_metric`` into ``catboost.CatBoostClassifier``/
``CatBoostRegressor`` makes plain ``sklearn.clone()`` raise: CatBoost's own ``get_params()`` returns a
FRESH list object for this param on every call, and ``sklearn.base._clone_parametrized`` validates every
constructor-round-tripped param via an IDENTITY check, not equality.

Fixed by re-enabling the dict assignment AND extending the EXISTING ``__sklearn_clone__`` monkeypatch in
``mlframe.metrics._ice_metric`` (previously only covering ``eval_metric``, for the same identity-check
failure mode) to also cover ``custom_metric``. That patch installs directly on the stock
``CatBoostClassifier``/``CatBoostRegressor`` classes the moment ``mlframe.metrics.core`` is imported --
which ``get_training_configs`` (and every production construction site, transitively) already does.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone

# Importing mlframe.training (or mlframe.metrics.core directly) installs the clone patch used below.
from mlframe.training._helpers_training_configs import get_training_configs


def _toy_classif_data(n: int = 300, seed: int = 0):
    """Tiny separable binary-classification toy for a fast CatBoost fit."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 4)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    return X, y


def _toy_regr_data(n: int = 300, seed: int = 0):
    """Tiny well-conditioned regression toy for a fast CatBoost fit."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 4)
    y = X[:, 0] * 2.0 + X[:, 1] - 0.5 * X[:, 2] + rng.normal(scale=0.01, size=n)
    return X, y


def test_cb_classif_config_carries_custom_metric():
    """get_training_configs's CB_CLASSIF dict actually sets custom_metric (was dead/commented-out)."""
    configs = get_training_configs(catboost_custom_classif_metrics=["AUC", "Precision"])
    assert tuple(configs.CB_CLASSIF["custom_metric"]) == ("AUC", "Precision")


def test_cb_regr_config_carries_custom_metric():
    """get_training_configs's CB_REGR dict actually sets custom_metric (was dead/commented-out)."""
    configs = get_training_configs(catboost_custom_regr_metrics=["RMSE", "MAPE"])
    assert tuple(configs.CB_REGR["custom_metric"]) == ("RMSE", "MAPE")


def test_cb_classif_config_omits_custom_metric_when_empty():
    """Empty/None custom metrics must not inject a stray kwarg (CatBoost rejects an empty tuple for some metrics)."""
    configs = get_training_configs(catboost_custom_classif_metrics=[])
    assert "custom_metric" not in configs.CB_CLASSIF


def test_catboost_classifier_with_custom_metric_survives_sklearn_clone():
    """The exact failure this bug caused, on the STOCK catboost class: once mlframe.metrics.core is
    imported (as get_training_configs always does), the shared __sklearn_clone__ patch makes
    CatBoostClassifier(custom_metric=[...]) clone-safe without any special subclass at the call site."""
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(iterations=5, custom_metric=["AUC", "Precision"], verbose=False)
    assert hasattr(CatBoostClassifier, "__sklearn_clone__"), "mlframe.metrics.core's clone patch was not installed"
    cloned = clone(model)
    assert list(cloned.get_params()["custom_metric"]) == ["AUC", "Precision"]


def test_catboost_regressor_with_custom_metric_survives_sklearn_clone():
    """Same identity-check failure mode as the classifier test above, for CatBoostRegressor."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(iterations=5, custom_metric=["RMSE", "MAPE"], verbose=False)
    assert hasattr(CatBoostRegressor, "__sklearn_clone__"), "mlframe.metrics.core's clone patch was not installed"
    cloned = clone(model)
    assert list(cloned.get_params()["custom_metric"]) == ["RMSE", "MAPE"]


def test_cb_classif_config_estimator_custom_metric_reaches_eval_results_after_clone():
    """custom_metric must actually be used by CatBoost during training, not just survive the param
    round-trip, and it must still be usable AFTER a clone() (the production path -- e.g.
    train_eval.py's per-weight-schema loop -- clones the configured estimator before fit)."""
    from catboost import CatBoostClassifier

    X, y = _toy_classif_data()
    configs = get_training_configs(catboost_custom_classif_metrics=["AUC", "Precision"], iterations=10, has_time=False)
    model = CatBoostClassifier(**configs.CB_CLASSIF)
    cloned = clone(model)
    cloned.fit(X, y, eval_set=(X, y))
    eval_metrics = set(cloned.get_evals_result()["validation"].keys())
    assert "AUC" in eval_metrics
    assert "Precision" in eval_metrics


def test_cb_regr_config_estimator_custom_metric_reaches_eval_results_after_clone():
    """Same post-clone eval-results check as the classifier test above, for CatBoostRegressor."""
    from catboost import CatBoostRegressor

    X, y = _toy_regr_data()
    configs = get_training_configs(catboost_custom_regr_metrics=["RMSE", "MAPE"], iterations=10, has_time=False)
    model = CatBoostRegressor(**configs.CB_REGR)
    cloned = clone(model)
    cloned.fit(X, y, eval_set=(X, y))
    eval_metrics = set(cloned.get_evals_result()["validation"].keys())
    assert "RMSE" in eval_metrics
    assert "MAPE" in eval_metrics
