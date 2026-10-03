"""Enum-like string fields of the training configs reject a value the code does not know, at construction."""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from mlframe.metrics.classification._threshold_optimization import THRESHOLD_METRICS
from mlframe.training.configs import MultilabelDispatchConfig, PreprocessingBackendConfig, TrainingBehaviorConfig


def test_decision_threshold_metric_literal_matches_the_optimizers_metrics():
    """The config's accepted values are exactly the metrics the threshold optimizer implements."""
    annotation = TrainingBehaviorConfig.model_fields["tune_decision_threshold_metric"].annotation
    assert set(typing.get_args(annotation)) == set(THRESHOLD_METRICS)


def test_unknown_decision_threshold_metric_raises():
    """A typo in the metric name used to surface only when the threshold was tuned."""
    with pytest.raises(ValidationError):
        TrainingBehaviorConfig(tune_decision_threshold_metric="f_1")


def test_unknown_multilabel_strategy_and_chain_order_raise():
    """Strategy and chain order are closed sets."""
    with pytest.raises(ValidationError):
        MultilabelDispatchConfig(strategy="wrappr")
    with pytest.raises(ValidationError):
        MultilabelDispatchConfig(chain_order_strategy="randomly")
    with pytest.raises(ValidationError, match="chain_order_user"):
        MultilabelDispatchConfig(chain_order_strategy="user")


@pytest.mark.parametrize("value", ["ordinal", "onehot", "none", None])
def test_categorical_encoding_accepts_the_implemented_values(value):
    """The implemented encodings plus ``none`` / None (no encoding step) validate."""
    assert PreprocessingBackendConfig(categorical_encoding=value).categorical_encoding == value


def test_unknown_categorical_encoding_raises():
    """``target`` was documented but never implemented, so it is rejected rather than silently skipped."""
    with pytest.raises(ValidationError):
        PreprocessingBackendConfig(categorical_encoding="target")
