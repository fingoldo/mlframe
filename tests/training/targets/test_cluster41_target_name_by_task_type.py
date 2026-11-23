"""Cluster 41 regression: model names and degenerate-target warnings follow the task type (audit 2026-11-20, finding 41-F7)."""

import logging

import numpy as np

from mlframe.training._configs_base import TargetTypes
from mlframe.training.targets._train_eval_select_target import _select_regression_target


def test_quantile_regression_is_not_tagged_as_binary_rate(caplog):
    """Quantile regression gets the regression tag, not BTTR, and no single-class warning (41-F7)."""
    y = np.linspace(0.0, 10.0, 50)
    with caplog.at_level(logging.WARNING):
        name = _select_regression_target(TargetTypes.QUANTILE_REGRESSION, "t", "m", [], y, y)
    assert "BTTR" not in name
    assert "MTTR" in name
    assert not any("degenerate" in r.getMessage() for r in caplog.records)


def test_multiclass_without_label_one_is_not_reported_single_class(caplog):
    """A multiclass target with no label 1 must not trigger the binary single-class warning (41-F7)."""
    y = np.array([0, 2, 2, 0, 3, 3])
    with caplog.at_level(logging.WARNING):
        _select_regression_target(TargetTypes.MULTICLASS_CLASSIFICATION, "t", "m", [], y, y)
    assert not any("degenerate" in r.getMessage() for r in caplog.records)


def test_binary_single_class_still_warns(caplog):
    """Control: a real binary target with one class still gets the degenerate warning (41-F7)."""
    y = np.zeros(20, dtype=int)
    with caplog.at_level(logging.WARNING):
        _select_regression_target(TargetTypes.BINARY_CLASSIFICATION, "t", "m", [], y, y)
    assert any("degenerate" in r.getMessage() for r in caplog.records)
