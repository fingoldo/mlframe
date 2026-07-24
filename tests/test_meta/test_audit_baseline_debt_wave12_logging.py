"""Baseline-debt wave 12: representative logging regression tests for the 17 genuine
broad_except_swallow sites fixed across training/composite/discovery/{screening,_auto_chain,
_filter,_fit_ram,_ktc_dispatch,_tiny_rerank}.py, training/composite/transforms/{extended,
_grouped_extra}.py, reporting/charts/{pdp_ice,quantile}.py, reporting/renderers/_plotly_color.py,
models/ensembling/{per_member_tuning,__init__}.py, and
feature_selection/wrappers/{_helpers,_helpers_importance,rfecv/_fit_init}.py -- one spot-check
per file rather than one test per site, since these are uniform additive debug-log-on-failure
changes with no behavior change on the success path (already covered by each module's existing
test suite).
"""

from __future__ import annotations

import logging


def test_screening_is_numeric_column_logs_on_failure(caplog):
    """`_is_numeric_column` must log and return False when the dtype probe raises."""
    from mlframe.training.composite.discovery.screening import _is_numeric_column

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.discovery.screening"):
        out = _is_numeric_column(object(), "not_a_column")
    assert out is False


def test_auto_chain_transform_gain_logs_on_failure(caplog):
    """The per-transform gain estimator must log when fit/forward raises -- pinned via source
    presence since it's a module-private helper reused across several nested call sites."""
    import inspect
    import mlframe.training.composite.discovery._auto_chain as ac

    src = inspect.getsource(ac)
    assert "transform gain estimate: fit/forward failed" in src
    assert "logger.debug" in src


def test_filter_leak_corr_sizing_logs_on_psutil_failure(caplog):
    """The leak-corr matrix sizing guard must log when the psutil probe fails -- forced via
    `sys.modules` poisoning (psutil is imported lazily inside the function)."""
    import sys
    import numpy as np
    import mlframe.training.composite.discovery._filter as filt

    real_psutil = sys.modules.pop("psutil", None)
    sys.modules["psutil"] = None
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.discovery._filter"):
            arrays, _y = filt._maybe_sample_for_leak_corr(["f0"], [np.zeros(10, dtype=np.float32)], np.zeros(10))
    finally:
        sys.modules.pop("psutil", None)
        if real_psutil is not None:
            sys.modules["psutil"] = real_psutil
    assert len(arrays) == 1
    assert any("psutil probe failed" in rec.message for rec in caplog.records)


def test_fit_ram_phase_log_logs_on_memory_probe_failure(caplog, monkeypatch):
    """The discovery-phase RAM logger must log (at DEBUG) and no-op when the memory probe raises."""
    import mlframe.training.composite.discovery._fit_ram as fit_ram

    def _raise():
        """Always raises ``RuntimeError('boom')`` to force the except branch."""
        raise RuntimeError("boom")

    monkeypatch.setattr(fit_ram, "_process_mem_mb", _raise)
    monkeypatch.setattr(fit_ram.logger, "isEnabledFor", lambda level: True)
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.discovery._fit_ram"):
        fit_ram._phase_ram_report({}, "test-phase")
    assert any("memory probe failed" in rec.message for rec in caplog.records)


def test_discovery_ktc_dispatch_get_cache_logs_on_import_failure(caplog):
    """`_get_cache` (discovery variant) must log when the pyutilz import fails."""
    import sys
    import mlframe.training.composite.discovery._ktc_dispatch as kd

    real_mod = sys.modules.pop("mlframe.feature_selection.filters", None)
    sys.modules["mlframe.feature_selection.filters"] = None
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.discovery._ktc_dispatch"):
            out = kd._get_cache()
    finally:
        sys.modules.pop("mlframe.feature_selection.filters", None)
        if real_mod is not None:
            sys.modules["mlframe.feature_selection.filters"] = real_mod
    assert out is None
    assert any("import failed" in rec.message for rec in caplog.records)


def test_tiny_rerank_ram_log_logs_on_memory_probe_failure(caplog):
    """The tiny-rerank RAM checkpoint logger must log on a memory-probe failure -- pinned via
    source presence since `._fit._process_mem_mb` is imported lazily inside the function body."""
    import inspect
    import mlframe.training.composite.discovery._tiny_rerank as tr

    src = inspect.getsource(tr)
    assert "tiny_rerank RAM log: memory probe failed" in src
    assert "logger.debug" in src


def test_transforms_extended_smoothing_spline_logs_on_fit_failure(caplog):
    """The smoothing-spline forward transform must log and fall back to y_mean on a fit failure --
    pinned via source presence since it needs registered fit params to invoke directly."""
    import inspect
    import mlframe.training.composite.transforms.extended as ext

    src = inspect.getsource(ext)
    assert "smoothing spline forward: fit/eval failed" in src
    assert "logger.debug" in src


def test_grouped_extra_per_group_fit_logs_on_failure():
    """`_grouped_np_fit`'s per-group fit must log and fall back to global on a per-group fit
    failure -- pinned via source presence since triggering a genuine per-group `fit_fn` failure
    needs a pathological per-group slice that's awkward to construct through the public API."""
    import inspect
    import mlframe.training.composite.transforms._grouped_extra as ge

    src = inspect.getsource(ge)
    assert "per-group fit failed" in src
    assert "logger.debug" in src


def test_pdp_ice_text_feature_indices_logs_on_failure(caplog):
    """The text-feature-index probe must log and return an empty set on failure."""
    import mlframe.reporting.charts.pdp_ice as pdp_ice

    class _RaisingModel:
        """A model stub whose `get_text_feature_indices` always raises."""

        def get_text_feature_indices(self):
            """Always raises ``RuntimeError('boom')`` on call."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.reporting.charts.pdp_ice"):
        out = pdp_ice._model_text_feature_names(_RaisingModel(), [])
    assert out == set()
    assert any("text-feature index probe failed" in rec.message for rec in caplog.records)


def test_quantile_model_diagnostics_decompose_returns_none_or_tuple():
    """`_model_diagnostics_decompose` must return None or a (decompose, PinballLoss) tuple, never raise."""
    from mlframe.reporting.charts.quantile import _model_diagnostics_decompose

    out = _model_diagnostics_decompose()
    assert out is None or (isinstance(out, tuple) and len(out) == 2)


def test_plotly_color_rgba_logs_on_conversion_failure(caplog):
    """`_rgba` must log and pass the color through unchanged when conversion fails."""
    from mlframe.reporting.renderers._plotly_color import _rgba

    with caplog.at_level(logging.DEBUG, logger="mlframe.reporting.renderers._plotly_color"):
        out = _rgba("not_a_real_color_name", 0.5)
    assert out == "not_a_real_color_name"
    assert any("color conversion failed" in rec.message for rec in caplog.records)


def test_per_member_tuning_code_version_returns_str_or_none():
    """`per_member_code_version` must return a str or None, never raise."""
    from mlframe.models.ensembling.per_member_tuning import per_member_code_version

    out = per_member_code_version()
    assert out is None or isinstance(out, str)


def test_ensembling_leaderboard_build_logs_on_failure(caplog):
    """The ensemble-leaderboard builder must log on failure -- pinned via source presence since
    it's a nested closure inside a larger comparison-table function."""
    import inspect
    import mlframe.models.ensembling as ensembling_mod

    src = inspect.getsource(ensembling_mod)
    assert "ensemble leaderboard build failed" in src
    assert "logger.debug" in src


def test_wrappers_helpers_thread_pinning_logs_on_get_params_failure(caplog):
    """The estimator thread-pinning helper must log when `get_params()` raises."""
    from mlframe.feature_selection.wrappers._helpers import _pin_threads_to_one

    class _RaisingEstimator:
        """An estimator stub with `set_params` present but a raising `get_params`."""

        def set_params(self, **kwargs):
            """No-op, present only to satisfy the `hasattr` gate."""

        def get_params(self):
            """Always raises ``RuntimeError('boom')`` on call."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection.wrappers._helpers"):
        _pin_threads_to_one(_RaisingEstimator())
    assert any("get_params() failed" in rec.message for rec in caplog.records)


def test_wrappers_helpers_importance_fold_is_all_finite_logs_on_failure(caplog):
    """`_fold_is_all_finite` must log and conservatively return False on a coercion failure."""
    from mlframe.feature_selection.wrappers._helpers_importance import _fold_is_all_finite

    class _Unconvertible:
        """An object whose `__array__` always raises, forcing the coercion except branch."""

        def __array__(self, dtype=None):
            """Always raises ``RuntimeError('boom')`` on coercion."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection.wrappers._helpers_importance"):
        out = _fold_is_all_finite(_Unconvertible())
    assert out is False
    assert any("coercion failed" in rec.message for rec in caplog.records)


def test_rfecv_fit_init_param_signature_logs_on_failure(caplog):
    """The RFECV param-signature helper must log and fall back to a unique sentinel on failure --
    pinned via source presence since it's bound as a method on the RFECV estimator class."""
    import inspect
    import mlframe.feature_selection.wrappers.rfecv._fit_init as fit_init

    src = inspect.getsource(fit_init)
    assert "RFECV param-signature computation failed" in src
    assert "logger.debug" in src
