"""OOF diagnostics say when a supervised pre_pipeline step ran outside the folds (tracker 2026-11-02 wave 15, LEAK15-3)."""

import logging

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.training._trainer_train_and_evaluate_parts import _stamp_oof_honesty


def test_a_selector_in_the_pre_pipeline_is_flagged_and_logged(caplog):
    """SelectKBest was fitted with every train label before the OOF loop; its OOF scores are optimistic."""
    diag = {}
    with caplog.at_level(logging.WARNING, logger="mlframe.training.trainer"):
        _stamp_oof_honesty(diag, Pipeline([("scale", StandardScaler()), ("select", SelectKBest(k=2))]))
    assert diag["supervised_preprocessing_outside_folds"] is True
    assert any("outside the folds" in r.getMessage() for r in caplog.records)


def test_unsupervised_or_missing_pre_pipeline_is_not_flagged():
    """A scaler uses no labels; no pre_pipeline means nothing ran outside the folds."""
    for pp in (None, Pipeline([("scale", StandardScaler())])):
        diag = {}
        _stamp_oof_honesty(diag, pp)
        assert diag["supervised_preprocessing_outside_folds"] is False
