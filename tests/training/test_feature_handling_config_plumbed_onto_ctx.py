"""Verify setup_configuration plumbs the caller's feature_handling_config onto ctx.

Pre-fix behaviour: setup_configuration validated the kwarg but never stamped it onto the returned TrainingContext,
so the Wave-3 _maybe_run_feature_handling_apply consumer in _phase_train_one_target read None and the FH plan was
silently skipped on every suite invocation.

Post-fix: the supplied config lands on ctx.artifacts["feature_handling_config"] (TrainingContext is slots=True with
no dedicated slot, so artifacts is the only in-scope storage path; the consumer needs a separate follow-up to read
from artifacts).
"""
from __future__ import annotations

from mlframe.training.core._phase_config_setup import setup_configuration
from mlframe.training.feature_handling.config import FeatureHandlingConfig


def test_feature_handling_config_lands_on_ctx_artifacts():
    fhc = FeatureHandlingConfig()
    ctx = setup_configuration(
        preprocessing_config=None,
        pipeline_config=None,
        feature_types_config=None,
        split_config=None,
        hyperparams_config=None,
        behavior_config=None,
        reporting_config=None,
        output_config=None,
        outlier_detection_config=None,
        feature_selection_config=None,
        confidence_analysis_config=None,
        baseline_diagnostics_config=None,
        dummy_baselines_config=None,
        quantile_regression_config=None,
        composite_target_discovery_config=None,
        feature_handling_config=fhc,
        model_name="m",
        target_name="t",
        mlframe_models=None,
        verbose=0,
    )
    stashed = (
        getattr(ctx, "feature_handling_config", None)
        or ctx.artifacts.get("feature_handling_config")
    )
    assert stashed is fhc, (
        "setup_configuration must plumb the caller's FeatureHandlingConfig onto ctx "
        "(slot OR artifacts) so the Wave-3 _maybe_run_feature_handling_apply consumer can see it."
    )


def test_feature_handling_config_none_does_not_pollute_artifacts():
    """When the caller passes None we must not stash a None entry into artifacts (downstream `.get(..., None)` already returns None)."""
    ctx = setup_configuration(
        preprocessing_config=None,
        pipeline_config=None,
        feature_types_config=None,
        split_config=None,
        hyperparams_config=None,
        behavior_config=None,
        reporting_config=None,
        output_config=None,
        outlier_detection_config=None,
        feature_selection_config=None,
        confidence_analysis_config=None,
        baseline_diagnostics_config=None,
        dummy_baselines_config=None,
        quantile_regression_config=None,
        composite_target_discovery_config=None,
        feature_handling_config=None,
        model_name="m",
        target_name="t",
        mlframe_models=None,
        verbose=0,
    )
    assert "feature_handling_config" not in ctx.artifacts
