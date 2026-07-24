"""Baseline-debt wave 9: representative logging regression tests for the 18 genuine
broad_except_swallow sites fixed across training/core/_ar1_failsafe_veto.py,
training/core/_phase_composite_post_moe.py, training/feature_handling/fingerprint.py,
training/neural/_muon_triton_kernel.py, training/pipeline/_pipeline_helpers.py,
training/reporting/_reporting.py, training/strategies/pipeline_cache.py,
training/neural/_flat_torch_module/_flat_torch_predict_accel.py, and
training/composite/ensemble/__init__.py -- one spot-check per file rather than one test per
site, since these are uniform additive debug-log-on-failure changes with no behavior change on
the success path (already covered by each module's existing test suite).
"""

from __future__ import annotations

import logging


def test_ar1_failsafe_veto_compute_val_veto_logs_on_target_extraction_failure(caplog):
    """`compute_val_veto` must log when val target extraction raises."""
    from mlframe.training.core._ar1_failsafe_veto import compute_val_veto

    class _Config:
        """A minimal config stub with the veto flag enabled."""

        ar1_failsafe_val_crosscheck = True

    class _Unindexable:
        """An object whose `__array__` raises, forcing the extraction except branch."""

        def __array__(self, dtype=None):
            """Always raises ``RuntimeError('boom')`` on coercion."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._ar1_failsafe_veto"):
        out = compute_val_veto(
            oof_names=["lag_predict", "lgb"],
            oof_rmses=[1.0, 1.0],
            oof_components=[None, None],
            filtered_val_df=object(),
            filtered_val_idx=[0, 1],
            oof_y_full=_Unindexable(),
            lag_failsafe_tol=0.1,
            config=_Config(),
        )
    assert out is None
    assert any("val target extraction failed" in rec.message for rec in caplog.records)


def test_phase_composite_post_moe_extract_group_array_logs_on_failure(caplog):
    """`_extract_group_array` must log when the column read raises."""
    from mlframe.training.core._phase_composite_post_moe import _extract_group_array

    class _RaisingFrame:
        """A frame-like object whose `.get_column` raises, forcing the except branch."""

        columns = ["grp"]

        def get_column(self, name):
            """Always raises ``RuntimeError('boom')`` on access."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._phase_composite_post"):
        out = _extract_group_array(_RaisingFrame(), "grp")
    assert out is None
    assert any("column read failed" in rec.message for rec in caplog.records)


def test_fingerprint_cache_key_logs_on_columns_read_failure(caplog):
    """`_fp_cache_key` must log when `df.columns` access raises."""
    from mlframe.training.feature_handling.fingerprint import _fp_cache_key

    class _NoColumns:
        """An object with no `.columns` attribute (raises AttributeError on access)."""

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.feature_handling.fingerprint"):
        out = _fp_cache_key(_NoColumns(), 128)
    assert out is None
    assert any("df.columns read failed" in rec.message for rec in caplog.records)


def test_muon_triton_kernel_get_ktc_logs_on_import_failure(caplog):
    """`_get_kernel_tuning_cache` must log when the pyutilz import fails."""
    import sys
    import mlframe.training.neural._muon_triton_kernel as mtk

    real_mod = sys.modules.pop("mlframe.feature_selection.filters", None)
    sys.modules["mlframe.feature_selection.filters"] = None
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.training.neural._muon_triton_kernel"):
            out = mtk._get_kernel_tuning_cache()
    finally:
        sys.modules.pop("mlframe.feature_selection.filters", None)
        if real_mod is not None:
            sys.modules["mlframe.feature_selection.filters"] = real_mod
    assert out is None
    assert any("import failed" in rec.message for rec in caplog.records)


def test_pipeline_helpers_selector_feature_names_logs_on_get_support_failure(caplog):
    """The selector-feature-name resolver must log when `get_support()` raises -- pinned via
    source presence since the helper is a nested closure inside a larger dispatch function."""
    import inspect
    import mlframe.training.pipeline._pipeline_helpers as ph

    src = inspect.getsource(ph)
    assert "selector.get_support() failed" in src
    assert "logger.debug" in src


def test_reporting_frame_to_text_logs_on_rendering_failure(caplog):
    """`_frame_to_text` must log when `to_string()` raises."""
    from mlframe.training.reporting._reporting import _frame_to_text

    class _RaisingFrame:
        """An object exposing a `to_string` that always raises."""

        def to_string(self):
            """Always raises ``RuntimeError('boom')`` when called."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.reporting._reporting"):
        out = _frame_to_text(_RaisingFrame())
    assert out is None
    assert any("to_string() rendering failed" in rec.message for rec in caplog.records)


def test_strategies_pipeline_cache_estimate_slot_nbytes_returns_int():
    """`_estimate_slot_nbytes` must return a plain int for any input without crashing."""
    from mlframe.training.strategies.pipeline_cache import _estimate_slot_nbytes

    out = _estimate_slot_nbytes(object())
    assert isinstance(out, int)


def test_flat_torch_predict_accel_recurrent_scan_logs_on_failure(caplog):
    """The torch.compile predict path's recurrent-module scan must log on failure -- pinned via
    source presence since the method lives on a torch.nn.Module subclass requiring a full
    Lightning network fixture to instantiate directly."""
    import inspect
    import mlframe.training.neural._flat_torch_module._flat_torch_predict_accel as fpa

    src = inspect.getsource(fpa)
    assert "recurrent-module scan failed, falling back to eager" in src
    assert src.count('logger.debug("torch.compile predict path: recurrent-module scan failed') == 1
    assert src.count('logger.debug("CUDA-graph predict path: recurrent-module scan failed') == 1


def test_composite_ensemble_is_monotone_nondecreasing_logs_on_coercion_failure(caplog):
    """`_is_monotone_nondecreasing` must log when `np.asarray` coercion raises."""
    from mlframe.training.composite.ensemble import _is_monotone_nondecreasing

    class _Unconvertible:
        """An object whose `__array__` always raises, forcing the coercion except branch."""

        def __array__(self, dtype=None):
            """Always raises ``RuntimeError('boom')`` on coercion."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.ensemble"):
        out = _is_monotone_nondecreasing(_Unconvertible())
    assert out is False
    assert any("np.asarray coercion failed" in rec.message for rec in caplog.records)
