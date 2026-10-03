"""``threshold_optimizer_kwargs`` / ``isotonic_risk_kwargs`` / ``confidence_shrinkage_kwargs`` are validated at config creation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlframe.training.calibration_params.configs import ConfidenceShrinkageConfig, IsotonicRiskConfig, ThresholdOptimizerConfig
from mlframe.training.configs import RegressionCalibrationConfig, TrainingBehaviorConfig
from mlframe.training.core._phase_finalize_calibration import _param_kwargs


def test_a_dict_is_coerced_into_the_strict_model():
    """A dict of the callee's arguments still works and becomes the typed model."""
    cfg = TrainingBehaviorConfig(threshold_optimizer_kwargs={"n_thresholds": 50, "cv": 3}, isotonic_risk_kwargs={"remediate": True})
    assert isinstance(cfg.threshold_optimizer_kwargs, ThresholdOptimizerConfig)
    assert isinstance(cfg.isotonic_risk_kwargs, IsotonicRiskConfig)
    assert _param_kwargs(cfg.threshold_optimizer_kwargs) == {"n_thresholds": 50, "cv": 3}
    assert _param_kwargs(cfg.isotonic_risk_kwargs) == {"remediate": True}


def test_a_misspelled_argument_raises_at_creation():
    """These used to be forwarded as ``**kwargs`` and fail only inside finalize, where the step is logged and skipped."""
    with pytest.raises(ValidationError, match="n_threshold"):
        TrainingBehaviorConfig(threshold_optimizer_kwargs={"n_threshold": 50})
    with pytest.raises(ValidationError, match="remedate"):
        TrainingBehaviorConfig(isotonic_risk_kwargs={"remedate": True})
    with pytest.raises(ValidationError, match="neutral_val"):
        RegressionCalibrationConfig(confidence_shrinkage_kwargs={"neutral_val": 0.5})


def test_out_of_range_values_raise_at_creation():
    """Range guards: a threshold grid needs two points, the isotonic segment ratio is a fraction, ``cv`` needs two folds."""
    with pytest.raises(ValidationError):
        ThresholdOptimizerConfig(n_thresholds=1)
    with pytest.raises(ValidationError):
        ThresholdOptimizerConfig(cv=1)
    with pytest.raises(ValidationError):
        IsotonicRiskConfig(segment_ratio_threshold=1.5)


def test_none_and_plain_mappings_are_accepted_by_the_finalize_helper():
    """Callers that hand the finalize phase a plain mapping (older code, tests) keep working."""
    assert _param_kwargs(None) == {}
    assert _param_kwargs({"neutral_value": 0.4}) == {"neutral_value": 0.4}
    assert _param_kwargs(ConfidenceShrinkageConfig(neutral_value=0.4, segment_ids=[1, 2])) == {"neutral_value": 0.4, "segment_ids": [1, 2]}


def test_defaults_forward_nothing():
    """An unconfigured step is called with the callee's own defaults."""
    assert TrainingBehaviorConfig().threshold_optimizer_kwargs is None
    assert RegressionCalibrationConfig().confidence_shrinkage_kwargs is None
