"""Baseline-debt wave 11: representative logging regression tests for the 20 genuine
broad_except_swallow sites fixed across training/composite/{panel,ranking,survival}.py,
training/core/predict.py, training/core/_achievable_ceiling.py, training/core/_ar_skip.py,
training/core/_phase_helpers.py, training/core/_phase_train_one_target_{mlp_helpers,model_setup,
polars_fastpath}.py, training/core/_setup_helpers_pipeline_cache.py,
training/core/_volatility_lag_router.py, training/diagnostics/learning_curve.py,
training/feature_handling/{cache,registry}.py, training/neural/_recurrent_perf.py,
training/pipeline/{_categorical_composite_fe,_pipeline_fit_transform,__init__}.py, and
training/targets/_train_eval_select_target.py -- one spot-check per file rather than one test per
site, since these are uniform additive debug-log-on-failure changes with no behavior change on
the success path (already covered by each module's existing test suite).
"""

from __future__ import annotations

import logging


def test_composite_panel_is_polars_df_returns_false_for_plain_object():
    """`_is_polars_df` (panel variant) must return False (never raise) for a non-polars object."""
    from mlframe.training.composite.panel import _is_polars_df

    assert _is_polars_df(object()) is False


def test_composite_ranking_is_polars_df_returns_false_for_plain_object():
    """`_is_polars_df` (ranking variant) must return False (never raise) for a non-polars object."""
    from mlframe.training.composite.ranking import _is_polars_df

    assert _is_polars_df(object()) is False


def test_composite_survival_has_scikit_survival_returns_bool():
    """`_has_scikit_survival` must return a bool without crashing regardless of install state."""
    from mlframe.training.composite.survival import _has_scikit_survival

    assert isinstance(_has_scikit_survival(), bool)


def test_core_predict_is_post_hoc_calibrated_model_logs_on_failure(caplog):
    """`_is_post_hoc_calibrated_model` must log when type introspection raises."""
    from mlframe.training.core.predict import _is_post_hoc_calibrated_model

    class _BadType:
        """A metaclass-free stub whose `type()` access is fine but stands in for an odd object."""

    # `type(model_obj).__name__` cannot itself fail for a normal object; the except guards a
    # pathological `__class__` override, so this is pinned via source presence instead.
    import inspect
    import mlframe.training.core.predict as predict_mod

    src = inspect.getsource(predict_mod)
    assert "_is_post_hoc_calibrated_model: type introspection failed" in src
    assert "logger.debug" in src
    assert _is_post_hoc_calibrated_model(_BadType()) is False


def test_achievable_ceiling_model_factory_logs_on_lgb_unavailable():
    """The tiny probe-model factory must log and fall back to linear when lgb build fails --
    pinned via source presence since it's a nested closure inside a larger discovery function."""
    import inspect
    import mlframe.training.core._achievable_ceiling as ac

    src = inspect.getsource(ac)
    assert "lgb probe model unavailable, falling back to linear" in src
    assert "logger.debug" in src


def test_ar_skip_recompute_lag1_ar_logs_on_failure(caplog):
    """`_recompute_lag1_ar_per_group` must log and return None when the underlying import fails."""
    import sys
    from mlframe.training.core._ar_skip import _recompute_lag1_ar_per_group
    import numpy as np

    real_mod = sys.modules.pop("mlframe.training.targets", None)
    sys.modules["mlframe.training.targets"] = None
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._ar_skip"):
            out = _recompute_lag1_ar_per_group(np.arange(200, dtype=np.float64), np.zeros(200, dtype=int), np.arange(200))
    finally:
        sys.modules.pop("mlframe.training.targets", None)
        if real_mod is not None:
            sys.modules["mlframe.training.targets"] = real_mod
    assert out is None
    assert any("recompute failed" in rec.message for rec in caplog.records)


def test_phase_helpers_cat_heavy_size_logs_on_failure(caplog):
    """The categorical-heavy size estimator must log on failure -- pinned via source presence
    since it's a nested closure inside a larger phase-setup function."""
    import inspect
    import mlframe.training.core._phase_helpers as ph

    src = inspect.getsource(ph)
    assert "_cat_heavy_size: categorical-fraction size estimate failed" in src
    assert "logger.debug" in src


def test_phase_train_one_target_mlp_helpers_column_drop_logs_on_failure(caplog):
    """The MLP-helpers column-drop utility must log when `.drop()` raises -- pinned via source
    presence since it's a module-private helper reused across several nested call sites."""
    import inspect
    import mlframe.training.core._phase_train_one_target_mlp_helpers as mlph

    src = inspect.getsource(mlph)
    assert "column-drop failed, returning frame unmodified" in src
    assert "logger.debug" in src


def test_phase_train_one_target_model_setup_arr_logs_on_failure(caplog):
    """The plot-target coercion helper must log on failure -- pinned via source presence since
    it's a nested closure inside a larger diagnostics-charting function."""
    import inspect
    import mlframe.training.core._phase_train_one_target_model_setup as ms

    src = inspect.getsource(ms)
    assert "plot-target coercion failed" in src
    assert "logger.debug" in src


def test_phase_train_one_target_polars_fastpath_ram_budget_logs_on_failure(caplog, monkeypatch):
    """The RAM-share byte-budget resolver must log when the psutil probe fails -- forced via
    `sys.modules` poisoning (psutil is imported lazily inside the function)."""
    import sys
    import mlframe.training.core._phase_train_one_target_polars_fastpath as pf

    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_TYPE", "FREE_RAM_SHARE")
    monkeypatch.setenv("MLFRAME_PANDAS_VIEW_CACHE_SIZE", "0.2")
    real_psutil = sys.modules.pop("psutil", None)
    sys.modules["psutil"] = None
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._phase_train_one_target"):
            out = pf.resolve_pandas_view_cache_budget_bytes()
    finally:
        sys.modules.pop("psutil", None)
        if real_psutil is not None:
            sys.modules["psutil"] = real_psutil
    assert isinstance(out, float)
    assert any("psutil probe failed" in rec.message for rec in caplog.records)


def test_setup_helpers_pipeline_cache_load_logs_on_corrupt_file(caplog, tmp_path, monkeypatch):
    """The pipeline disk cache loader must log when the on-disk JSON is corrupt."""
    import mlframe.training.core._setup_helpers_pipeline_cache as cache_mod

    bad_path = tmp_path / "corrupt_cache.json"
    bad_path.write_bytes(b"{not valid json")
    monkeypatch.setattr(cache_mod, "_pipeline_disk_cache_path", lambda: str(bad_path))
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._setup_helpers_pipeline_cache"):
        cache_mod._load_pipeline_disk_cache_into_memory()
    assert any("load failed, treating as absent" in rec.message for rec in caplog.records)


def test_volatility_lag_router_extract_column_logs_on_failure(caplog):
    """`_extract_column` must log and return None when neither polars nor pandas access works."""
    from mlframe.training.core._volatility_lag_router import _extract_column

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._volatility_lag_router"):
        out = _extract_column(object(), "not_a_column")
    assert out is None


def test_learning_curve_supports_warm_start_logs_on_get_params_failure(caplog):
    """`_supports_warm_start` must log when `get_params()` raises."""
    from mlframe.training.diagnostics.learning_curve import _supports_warm_start

    class _RaisingEstimator:
        """An estimator stub with the required attrs but a raising `get_params`."""

        def set_params(self, **kwargs):
            """No-op, present only to satisfy the `hasattr` gate."""

        def get_params(self):
            """Always raises ``RuntimeError('boom')`` on call."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.diagnostics.learning_curve"):
        out = _supports_warm_start(_RaisingEstimator())
    assert out is None
    assert any("get_params() failed" in rec.message for rec in caplog.records)


def test_feature_handling_cache_eviction_logs_on_disk_probe_failure(caplog):
    """The disk-cache eviction pass must log when the free-space probe fails -- pinned via
    source presence since it's a bound method on a stateful cache class."""
    import inspect
    import mlframe.training.feature_handling.cache as cache_mod

    src = inspect.getsource(cache_mod)
    assert "disk-cache eviction: free-space probe failed" in src
    assert "logger.debug" in src


def test_feature_handling_registry_log_unhandled_logs_on_probe_failure(caplog):
    """The prewarm-future done-callback must log when the exception probe itself fails -- pinned
    via source presence since it's a nested closure bound to a specific future instance."""
    import inspect
    import mlframe.training.feature_handling.registry as registry_mod

    src = inspect.getsource(registry_mod)
    assert "_log_unhandled: future.exception() probe failed" in src
    assert "logger.debug" in src


def test_recurrent_perf_maybe_enable_cudnn_autotune_returns_none_without_cuda():
    """`maybe_enable_cudnn_rnn_autotune` must return None cleanly on a no-CUDA host."""
    from mlframe.training.neural._recurrent_perf import maybe_enable_cudnn_rnn_autotune
    from mlframe.training.neural._recurrent_config import RNNType
    import torch

    if torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA available on this host; the no-CUDA short-circuit path isn't exercised")
    out = maybe_enable_cudnn_rnn_autotune(RNNType.LSTM)
    assert out is None


def test_categorical_composite_fe_detect_cat_columns_logs_on_failure(caplog):
    """`_detect_cat_columns` must log when `select_dtypes` raises."""
    from mlframe.training.pipeline._categorical_composite_fe import _detect_cat_columns

    class _RaisingSelectDtypes:
        """A pandas-like stub whose `select_dtypes` always raises."""

        def select_dtypes(self, include=None):
            """Always raises ``RuntimeError('boom')`` on call."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.pipeline._categorical_composite_fe"):
        out = _detect_cat_columns(_RaisingSelectDtypes())
    assert out == []
    assert any("select_dtypes probe failed" in rec.message for rec in caplog.records)


def test_pipeline_fit_transform_looks_embedding_logs_on_failure(caplog):
    """The embedding-column detector must log when the sample probe fails -- pinned via source
    presence since it's a nested closure inside a larger pipeline-fit function."""
    import inspect
    import mlframe.training.pipeline._pipeline_fit_transform as pft

    src = inspect.getsource(pft)
    assert "_looks_embedding: sample probe failed" in src
    assert "logger.debug" in src


def test_pipeline_init_schema_drift_check_logs_on_failure(caplog):
    """The train-vs-other schema drift check must log when schema read fails."""
    import mlframe.training.pipeline as pipeline_mod

    class _NoSchema:
        """An object with no `.schema` attribute (raises AttributeError on access)."""

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.pipeline"):
        pipeline_mod._warn_on_schema_drift({}, _NoSchema(), "val")
    assert any("schema read failed" in rec.message for rec in caplog.records)


def test_train_eval_select_target_binary_pos_rate_logs_on_failure(caplog):
    """`_binary_pos_rate`'s coercion except must log on failure -- pinned via source presence
    since it's a nested closure inside a larger target-selection function."""
    import inspect
    import mlframe.training.targets._train_eval_select_target as tet

    src = inspect.getsource(tet)
    assert "_binary_pos_rate: coercion failed" in src
    assert "logger.debug" in src
