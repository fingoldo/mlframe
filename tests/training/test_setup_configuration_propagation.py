"""Meta-test: every public kwarg passed to ``train_mlframe_models_suite`` that
should land on ``TrainingContext`` actually does.

This catches the bug class fixed at commit 7479b54 (verbose=0 was silently
treated as verbose=1 because setup_configuration ran ``log_phase`` checks against
the local kwarg but never assigned ``ctx.verbose = verbose``).

The test calls ``setup_configuration`` with deliberately non-default values for
every kwarg known to flow into the context, then asserts each one is reflected on
the returned ctx. New ctx slots that are forwarded from setup_configuration must
be added here; the test will surface a default-vs-passed mismatch otherwise.

Bug surface this guards:
  - Caller passes verbose=0; ctx.verbose stays at class-default 1; every
    ``if ctx.verbose:`` block in downstream phases fires telemetry,
    pulls heavy renderers.plotly import, etc.
  - Caller passes mlframe_models=['lgb']; ctx.mlframe_models stays at
    default []; the per-model dispatch falls back to "all models".
  - Similar latent bugs for any new kwarg added to setup_configuration
    without a corresponding ctx assignment.
"""

from __future__ import annotations

import pytest

from mlframe.training.core._phase_config_setup import setup_configuration

# Sentinel marker objects for the config-typed kwargs. These don't need to be
# real Pydantic dataclasses; they just need to be distinguishable from None so
# the round-trip assertion can confirm propagation. linear_model_config and
# multilabel_dispatch_config slots on TrainingContext are typed Any so a plain
# object works.
_LINEAR_SENTINEL = object()
_MULTILABEL_SENTINEL = object()

# Each entry: (kwarg_name_to_setup_configuration, distinctive_value, ctx_slot_name).
# `ctx_slot_name` is the attribute on TrainingContext to verify against.
# When None it means the kwarg is accepted by setup_configuration but is NOT
# stored on ctx (e.g. one-shot local-use only). Listing it explicitly forces
# this test to be updated when behaviour changes.
_PROPAGATION_TABLE: list[tuple[str, object, str | None]] = [
    ("model_name", "diag_meta_model_name", "model_name"),
    ("target_name", "diag_meta_target_name", "target_name"),
    ("mlframe_models", ["lgb"], "mlframe_models"),
    ("use_mlframe_ensembles", False, "use_mlframe_ensembles"),
    ("use_ordinary_models", False, "use_ordinary_models"),
    ("verbose", 0, "verbose"),
    # Same bug class as verbose: both were public-API kwargs silently dropped on
    # the path to TrainingContext until commit after 7479b54. Without these two
    # entries the meta-test would not have caught the second wave of bugs that
    # the audit agent surfaced.
    ("linear_model_config", _LINEAR_SENTINEL, "linear_model_config"),
    ("multilabel_dispatch_config", _MULTILABEL_SENTINEL, "multilabel_dispatch_config"),
]


@pytest.mark.parametrize("kwarg,value,ctx_attr", _PROPAGATION_TABLE)
def test_setup_configuration_propagates_kwarg_to_ctx(kwarg, value, ctx_attr):
    """Each listed kwarg must round-trip through setup_configuration onto ctx.

    Builds a minimal arg map (all configs None so _ensure_config materialises
    defaults; only the kwarg under test gets the distinctive value). Asserts
    ctx.<ctx_attr> equals the distinctive value.
    """
    base_kwargs = dict(
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
        linear_model_config=None,
        multilabel_dispatch_config=None,
        model_name="default_model_name",
        target_name="default_target_name",
        mlframe_models=None,
        verbose=1,
    )
    base_kwargs[kwarg] = value
    ctx = setup_configuration(**base_kwargs)

    if ctx_attr is None:
        return  # locally consumed; nothing to verify on ctx

    actual = getattr(ctx, ctx_attr)
    assert actual == value, (
        f"setup_configuration kwarg {kwarg!r}={value!r} did not propagate to "
        f"ctx.{ctx_attr} (got {actual!r}). This is the bug class fixed at "
        f"commit 7479b54 for verbose; add the missing forward in "
        f"_phase_config_setup.py's TrainingContext(...) constructor call."
    )


def test_verbose_zero_round_trips_as_int_zero():
    """Specific regression for the original bug: verbose=0 must produce ctx.verbose==0.

    Prior to commit 7479b54 the TrainingContext constructor call omitted
    verbose=verbose, so ctx.verbose silently stayed at the dataclass default of 1.
    Every ``if ctx.verbose:`` block downstream then fired even when the caller
    explicitly asked for silent operation, including a 25ms cold-start import of
    mlframe.reporting.renderers.plotly for kaleido telemetry.
    """
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
        linear_model_config=None,
        multilabel_dispatch_config=None,
        model_name="vtest",
        target_name="vtest",
        mlframe_models=None,
        verbose=0,
    )
    assert ctx.verbose == 0, f"ctx.verbose expected 0, got {ctx.verbose!r}. Caller's verbose=0 was silently overridden by TrainingContext default."
    # Also: verbose=None must not crash (None-safe int coercion in the fix)
    ctx_none = setup_configuration(
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
        linear_model_config=None,
        multilabel_dispatch_config=None,
        model_name="vtest",
        target_name="vtest",
        mlframe_models=None,
        verbose=None,
    )
    assert ctx_none.verbose == 1, "verbose=None should fall back to the class default of 1"
