"""Cluster 41 regression: the model-kind predicates answer quietly for unknown estimators (audit 2026-11-20, finding 41-F20)."""

import logging

from sklearn.linear_model import Ridge

from mlframe.training.strategies import is_catboost_model, is_neural_model


def test_predicates_do_not_warn_for_an_unknown_estimator(caplog):
    """A plain Ridge is neither CatBoost nor neural, and asking must not log a dispatch WARNING (41-F20)."""
    with caplog.at_level(logging.WARNING):
        assert is_catboost_model(Ridge()) is False
        assert is_neural_model(Ridge()) is False
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "No registered strategy" in r.getMessage()]
