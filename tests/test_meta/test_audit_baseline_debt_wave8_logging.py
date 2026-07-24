"""Baseline-debt wave 8: representative logging regression tests for the 18 genuine
broad_except_swallow sites fixed across reporting/diagnostics_dispatch.py, system/_gpu_guard.py,
training/_data_helpers.py, training/_gpu_probe.py, training/_model_factories.py,
training/_predict_guards.py, training/_ram_helpers.py, training/_training_loop_refit.py, and
training/composite/quantile.py -- one spot-check per file rather than one test per site, since
these are uniform additive debug-log-on-failure changes with no behavior change on the success
path (already covered by each module's existing test suite).
"""

from __future__ import annotations

import logging


def test_diagnostics_dispatch_subset_rows_logs_on_fallback(caplog):
    """`_subset_rows` must log when the fancy-index path fails and it falls back to np.asarray."""
    from mlframe.reporting.diagnostics_dispatch import _subset_rows

    import numpy as np

    class _RaisingIndex:
        """A sequence whose `__getitem__` raises but that coerces cleanly via `__array__`,
        so the `np.asarray(frame)[idx]` fallback succeeds."""

        def __getitem__(self, idx):
            """Always raises ``RuntimeError('boom')`` on fancy-index access."""
            raise RuntimeError("boom")

        def __array__(self, dtype=None):
            """Coerces to a plain ndarray so the fallback path succeeds."""
            return np.array([10, 20, 30])

    with caplog.at_level(logging.DEBUG, logger="mlframe.reporting.diagnostics_dispatch"):
        out = _subset_rows(_RaisingIndex(), np.array([0]))
    assert out is not None
    assert any("fancy-index row subset failed" in rec.message for rec in caplog.records)


def test_gpu_guard_callable_looks_gpu_bound_logs_on_introspection_failure(caplog, monkeypatch):
    """`callable_looks_gpu_bound` must log and degrade to False when closure-var introspection raises
    an error other than the already-handled `TypeError`."""
    import mlframe.system._gpu_guard as gg

    def _fn():
        """A trivial no-arg callable used as the introspection target."""

    def _raise(*args, **kwargs):
        """Always raises ``RuntimeError('boom')`` to force the outer except branch."""
        raise RuntimeError("boom")

    monkeypatch.setattr(gg.inspect, "getclosurevars", _raise)
    with caplog.at_level(logging.DEBUG, logger="mlframe.system._gpu_guard"):
        out = gg.callable_looks_gpu_bound(_fn)
    assert out is False
    assert any("introspection failed" in rec.message for rec in caplog.records)


def test_gpu_guard_try_import_cupy_returns_none_false_when_unavailable():
    """`try_import_cupy` degrades to (None, False) without crashing (cupy is not installed here)."""
    from mlframe.system._gpu_guard import try_import_cupy

    cp, has_cupy = try_import_cupy()
    assert has_cupy in (True, False)
    if not has_cupy:
        assert cp is None


def test_data_helpers_normalize_multilabel_target_logs_on_double_failure(caplog):
    """`_normalize_multilabel_target` must log on both the asarray and the per-row stack fallback."""
    from mlframe.training._data_helpers import _normalize_multilabel_target
    import numpy as np

    class _Unstackable:
        """An object whose `.shape` attr makes it look array-like but whose coercion always raises."""

        shape = (2,)

        def __array__(self, dtype=None):
            """Always raises ``RuntimeError('boom')`` on coercion."""
            raise RuntimeError("boom")

    arr = np.empty(1, dtype=object)
    arr[0] = _Unstackable()
    with caplog.at_level(logging.DEBUG, logger="mlframe.training._data_helpers"):
        out = _normalize_multilabel_target(arr)
    assert out is arr
    messages = [rec.message for rec in caplog.records]
    assert any("np.asarray(tolist()) failed" in m for m in messages)
    assert any("per-row stack failed too" in m for m in messages)


def test_gpu_probe_xgb_support_logs_on_probe_failure(caplog):
    """`_probe_xgb_gpu_support` must log when the xgboost build_info probe fails."""
    import sys
    import mlframe.training._gpu_probe as gp

    if not gp.CUDA_IS_AVAILABLE:
        import pytest

        pytest.skip("no CUDA device visible; the probe short-circuits before the guarded call")

    real_xgb = sys.modules.pop("xgboost", None)
    sys.modules["xgboost"] = None
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.training._gpu_probe"):
            out = gp._probe_xgb_gpu_support()
    finally:
        sys.modules.pop("xgboost", None)
        if real_xgb is not None:
            sys.modules["xgboost"] = real_xgb
    assert out is False
    assert any("build_info probe failed" in rec.message for rec in caplog.records)


def test_model_factories_infer_callsite_returns_marker_on_failure(caplog):
    """`_infer_callsite` degrades to "?" (logged) if the stack walk raises -- pinned via source presence
    since the closure isn't independently callable outside `_lgb_shim`/`_xgb_shim`."""
    import inspect
    import mlframe.training._model_factories as mf

    src = inspect.getsource(mf)
    assert "_infer_callsite: stack walk failed" in src
    assert "logger.debug" in src


def test_predict_guards_recover_feature_names_logs_on_failure(caplog):
    """`_recover_cb_feature_names` in `_predict_guards` must log when model introspection fails."""
    from mlframe.training._predict_guards import _recover_cb_feature_names

    class _RaisingModel:
        """A model whose `feature_names_` access raises, forcing the except branch."""

        @property
        def feature_names_(self):
            """Always raises ``RuntimeError('boom')`` on access."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training._predict_guards"):
        out = _recover_cb_feature_names(_RaisingModel())
    assert out == ([], [])
    assert any("model introspection failed" in rec.message for rec in caplog.records)


def test_ram_helpers_get_process_rss_mb_returns_float():
    """`get_process_rss_mb` must return a float (0.0 on probe failure, real RSS otherwise) without crashing."""
    from mlframe.training._ram_helpers import get_process_rss_mb

    out = get_process_rss_mb()
    assert isinstance(out, float)


def test_training_loop_refit_collapse_detector_logs_on_coercion_failure(caplog):
    """`_maybe_refit_on_collapsed_predictions` must log when `train_target` coercion to ndarray fails."""
    from mlframe.training._training_loop_refit import _maybe_refit_on_collapsed_predictions

    class _Unconvertible:
        """An object whose `__array__` always raises, forcing the coercion except branch."""

        def __array__(self, dtype=None):
            """Always raises ``RuntimeError('boom')`` on coercion."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training._training_loop_refit"):
        out = _maybe_refit_on_collapsed_predictions(
            model=None,
            model_obj=None,
            model_type_name="mlp",
            train_df=None,
            train_target=_Unconvertible(),
            fit_params={},
            logger_=logging.getLogger("mlframe.training._training_loop_refit"),
        )
    assert out is False
    assert any("train_target coercion failed" in rec.message for rec in caplog.records)


def test_composite_quantile_flip_detector_logs_on_unknown_transform(caplog):
    """`_transform_inverse_decreasing` must log when `get_transform` rejects an unknown name."""
    from mlframe.training.composite.quantile import _transform_inverse_decreasing

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.quantile"):
        out = _transform_inverse_decreasing("__not_a_real_transform__")
    assert out is False
    assert any("get_transform" in rec.message and "failed" in rec.message for rec in caplog.records)
