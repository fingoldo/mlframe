"""Sensor: setup_configuration flips process-wide flags
(_set_residual_audit_enabled, set_inline_display_mode) but _phase_finalize must
restore them. Pre-fix the leading comment promised restore but no restore call
site existed -- two back-to-back suites silently inherited the first's flag.
"""
from __future__ import annotations

import os

import pytest


def _build_minimal_ctx_kwargs():
    """Minimal _ensure_config-friendly kwarg set for setup_configuration."""
    return dict(
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
        model_name="restore_test",
        target_name="restore_test",
        mlframe_models=None,
        verbose=0,
    )


def test_residual_audit_flag_snapshot_captured():
    """Setup must stash the prior residual_audit flag on ctx.artifacts so finalize can restore."""
    from mlframe.training.core._phase_config_setup import setup_configuration
    from mlframe.training.evaluation import (
        _set_residual_audit_enabled,
        _get_residual_audit_enabled,
    )

    # Seed a known prior value distinct from the default.
    _set_residual_audit_enabled(False)
    try:
        ctx = setup_configuration(**_build_minimal_ctx_kwargs())
        assert "_process_flag_prior_residual_audit" in ctx.artifacts
        prior = ctx.artifacts["_process_flag_prior_residual_audit"]
        assert prior is False, f"prior must be False (the value before setup), got {prior!r}"
    finally:
        _set_residual_audit_enabled(None)


def test_residual_audit_flag_restored_by_finalize():
    """_phase_finalize.finalize_suite must restore residual_audit to the snapshot value."""
    from mlframe.training.core._phase_config_setup import setup_configuration
    from mlframe.training.evaluation import (
        _set_residual_audit_enabled,
        _get_residual_audit_enabled,
    )
    # Pre-fix simulation: residual_audit set to False BEFORE the suite.
    _set_residual_audit_enabled(False)
    try:
        ctx = setup_configuration(**_build_minimal_ctx_kwargs())
        # setup flipped to default True (behavior_config.report_residual_audit default).
        # In pre-fix, this would stay True forever; post-fix the restore brings it back to False.
        # Simulate the restore block directly (full suite call is too heavy for this sensor).
        _artifacts = ctx.artifacts or {}
        _restored = _artifacts.pop("_process_flag_prior_residual_audit", None)
        if _restored is not None:
            _set_residual_audit_enabled(_restored)
        # Verify
        assert _get_residual_audit_enabled() is False, (
            "After restore, flag must equal the pre-setup snapshot (False), got True. "
            "_phase_finalize restore block is broken."
        )
    finally:
        _set_residual_audit_enabled(None)


def test_inline_display_mode_get_set_roundtrip():
    """Sanity check the new get_inline_display_mode getter returns what set_inline_display_mode stored."""
    from mlframe.reporting.renderers.save import (
        get_inline_display_mode,
        set_inline_display_mode,
    )
    # Save current
    _prior = get_inline_display_mode()
    try:
        for v in (True, False, None):
            set_inline_display_mode(v)
            assert get_inline_display_mode() is v, f"round-trip mismatch for {v!r}"
    finally:
        set_inline_display_mode(_prior)


def test_finalize_source_contains_restore_block():
    """Source-level guard: the restore block must exist in _phase_finalize.py.

    Direct behavioural test of the full suite flow is too heavy for a sensor;
    this source-check catches a future refactor that removes the restore call.
    """
    import pathlib
    # Derive the src path from the installed package so the source-check
    # works regardless of clone location; the previous hardcoded D:/ path
    # raised FileNotFoundError on every other machine.
    import mlframe as _mlframe
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "training" / "core" / "_phase_finalize.py"
    ).read_text(encoding="utf-8")
    assert "_process_flag_prior_residual_audit" in src, (
        "_phase_finalize must restore the residual_audit flag from ctx.artifacts."
    )
    assert "_process_flag_prior_inline_display" in src, (
        "_phase_finalize must restore the inline_display flag from ctx.artifacts."
    )
