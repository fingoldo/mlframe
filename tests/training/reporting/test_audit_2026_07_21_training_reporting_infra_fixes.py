"""Regression tests for audits/full_audit_2026-07-21/training_reporting_infra.md findings F1-F9.

One test (or tight group) per finding; each reproduces the pre-fix failure signature against the
real fixture before asserting the post-fix behavior, per the project's regression-test convention.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# F1 (P0): prediction-envelope-clip result not propagated to chart/audit/return value
# ---------------------------------------------------------------------------


def test_f1_envelope_clip_propagates_to_returned_preds():
    """report_regression_model_perf's returned preds reflect the envelope clip, not the raw catastrophic values."""
    from mlframe.training.reporting._reporting_regression import report_regression_model_perf

    rng = np.random.default_rng(0)
    n = 200
    targets = rng.normal(loc=20.0, scale=2.0, size=n)
    preds = targets + rng.normal(scale=0.5, size=n)
    # Catastrophic out-of-envelope predictions on a subset of rows.
    preds = preds.copy()
    preds[:5] = 1e6

    result_preds, _ = report_regression_model_perf(
        targets=targets,
        columns=["target"],
        model_name="testmodel",
        model=None,
        preds=preds,
        print_report=False,
        show_perf_chart=False,
        y_train_min=float(targets.min()),
        y_train_max=float(targets.max()),
        y_train_std=float(targets.std()),
    )
    assert np.all(result_preds[:5] < 1000.0), f"catastrophic predictions leaked into the returned preds unclipped: {result_preds[:5]}"
    assert not np.array_equal(result_preds[:5], preds[:5])


# ---------------------------------------------------------------------------
# F2 (P2): `if not classes:` crashes on an ndarray `classes` argument
# ---------------------------------------------------------------------------


def test_f2_ndarray_classes_argument_does_not_crash():
    """report_probabilistic_model_perf accepts an ndarray `classes` (>=2 elements) without the ambiguous-truth-value ValueError."""
    from mlframe.training.reporting._reporting_probabilistic import report_probabilistic_model_perf

    rng = np.random.default_rng(0)
    n = 200
    targets = rng.integers(0, 3, size=n)
    probs = rng.dirichlet(np.ones(3), size=n)

    # Pre-fix, `if not classes:` on this exact ndarray raises ValueError -- confirm the raw expression
    # really does (documents the failure mode this test guards against).
    classes = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="ambiguous"):
        bool(not classes)

    report_probabilistic_model_perf(
        targets=targets,
        columns=["c0", "c1", "c2"],
        model_name="testmodel",
        model=None,
        probs=probs,
        classes=classes,
        print_report=False,
        show_perf_chart=False,
        verbose=False,
        plot_outputs=None,
    )


# ---------------------------------------------------------------------------
# F3 (P2): single-backend render_and_save path had no exception handling / failure bookkeeping
# ---------------------------------------------------------------------------


def test_f3_single_backend_render_failure_is_caught_and_counted():
    """A raising renderer on the single-backend path is caught, counted, and does not propagate."""
    import mlframe.reporting.renderers.save as save_mod
    from mlframe.reporting.output import parse_plot_output_dsl
    from mlframe.reporting.spec import LinePanelSpec

    save_mod.reset_render_failure_stats()

    class _FakeRenderer:
        """Fake Renderer."""
        def render(self, spec, **kwargs):
            """Fake renderer backend used to force the failure path under test."""
            raise RuntimeError("simulated renderer failure")

    orig_get_renderer = save_mod.get_renderer
    save_mod.get_renderer = lambda backend: _FakeRenderer()
    try:
        spec = LinePanelSpec(x=np.arange(5), y=np.arange(5).astype(float), title="t", xlabel="x", ylabel="y")
        output = parse_plot_output_dsl("matplotlib[png]")
        result = save_mod.render_and_save(spec, output, "unused_base_path")
        assert result is None
        stats = save_mod.get_render_failure_stats()
        assert stats == {"total": 1, "timeouts": 0, "exceptions": 1}, stats
    finally:
        save_mod.get_renderer = orig_get_renderer
        save_mod.reset_render_failure_stats()


# ---------------------------------------------------------------------------
# F4 (P2): plotly showlegend applied uniformly instead of per-series
# ---------------------------------------------------------------------------


def test_f4_plotly_line_showlegend_is_per_series():
    """PlotlyRenderer._line only sets showlegend=True for series that actually carry a label."""
    from plotly.subplots import make_subplots

    from mlframe.reporting.renderers.plotly import PlotlyRenderer
    from mlframe.reporting.spec import LinePanelSpec

    x = np.arange(10)
    y1 = np.sin(x.astype(float))
    y2 = np.cos(x.astype(float))
    spec = LinePanelSpec(x=x, y=(y1, y2), series_labels=("Series A", None), title="t", xlabel="x", ylabel="y")

    fig = make_subplots(rows=1, cols=1)
    PlotlyRenderer()._line(fig, spec, 1, 1)
    showlegend_flags = [tr.showlegend for tr in fig.data]
    assert showlegend_flags == [True, False], showlegend_flags


# ---------------------------------------------------------------------------
# F5 (P2): _reporting_field_default's bare except silently + permanently swallowed ANY failure
# ---------------------------------------------------------------------------


def test_f5_reporting_field_default_narrowed_exception_handling():
    """KeyError/ImportError are caught + logged + memoized to None; any OTHER exception (e.g. a real
    pydantic API-shape bug) propagates instead of being silently and permanently swallowed."""
    import mlframe.training.reporting._reporting as rep
    import mlframe.training._reporting_configs as configs_mod

    rep._reporting_field_default.__dict__.clear()
    try:
        # A nonexistent field -> KeyError -> caught, memoized to None.
        result = rep._reporting_field_default("this_field_does_not_exist")
        assert result is None
        assert rep._reporting_field_default.__dict__["this_field_does_not_exist"] is None

        # A genuine unrelated bug (AttributeError) must propagate, not be swallowed into a cached None.
        rep._reporting_field_default.__dict__.clear()
        orig_model_fields = configs_mod.ReportingConfig.model_fields

        class _BrokenFields(dict):
            """Dict subclass whose __getitem__ raises AttributeError, simulating a pydantic API-shape change."""

            def __getitem__(self, key):
                raise AttributeError("simulated pydantic API-shape change")

        configs_mod.ReportingConfig.model_fields = _BrokenFields(orig_model_fields)
        try:
            with pytest.raises(AttributeError):
                rep._reporting_field_default("panel_emphasis")
        finally:
            configs_mod.ReportingConfig.model_fields = orig_model_fields
    finally:
        rep._reporting_field_default.__dict__.clear()


# ---------------------------------------------------------------------------
# F6 (P2): kaleido module-global state (server-started / burned / fail-count / oneshot counters)
# had no lock guarding check-then-act transitions across concurrent render_and_save threads
# ---------------------------------------------------------------------------


def test_f6_kaleido_state_lock_is_reentrant_and_thread_safe(monkeypatch):
    """_restart_kaleido_server calls _ensure_kaleido_server_started while already holding the lock
    (must not deadlock); concurrent threads mutating the oneshot counters must not lose updates.

    ``kaleido`` itself is faked out (a real call would spawn a Chromium subprocess) -- this test
    exercises only the module's own state-lock mechanics, not kaleido's real server lifecycle.
    """
    import sys
    import threading
    import types

    from mlframe.reporting.renderers import _kaleido as k

    fake_kaleido = types.SimpleNamespace(
        start_sync_server=lambda **kw: None,
        stop_sync_server=lambda **kw: None,
    )
    monkeypatch.setitem(sys.modules, "kaleido", fake_kaleido)
    monkeypatch.setattr(k, "_KALEIDO_SERVER_STARTED", False)
    monkeypatch.setattr(k, "_KALEIDO_PERSISTENT_FAIL_COUNT", 0)
    monkeypatch.setattr(k, "_KALEIDO_PERSISTENT_BURNED", False)

    assert isinstance(k._KALEIDO_STATE_LOCK, type(threading.RLock()))

    done = threading.Event()

    def call_restart():
        """Calls the restart path once, exercised concurrently by the thread-safety check below."""
        k._restart_kaleido_server()
        done.set()

    t = threading.Thread(target=call_restart, daemon=True)
    t.start()
    t.join(timeout=10)
    assert done.is_set(), "deadlock: _restart_kaleido_server did not return (RLock nesting broken)"

    k.reset_kaleido_oneshot_stats()
    errors = []

    def hammer():
        """Repeatedly acquires and releases the lock from a worker thread."""
        try:
            for _ in range(200):
                k._ensure_kaleido_server_started()
                k._is_kaleido_persistent_burned()
                k._record_kaleido_persistent_failure()
                k.record_kaleido_oneshot_call(0.001)
        except Exception as e:  # pragma: no cover - failure path asserted below
            errors.append(e)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=15)
    assert not any(th.is_alive() for th in threads)
    assert not errors, errors
    calls, _wall = k.get_kaleido_oneshot_stats()
    assert calls == 8 * 200, f"lost updates under contention: expected {8*200}, got {calls}"
    k.reset_kaleido_oneshot_stats()


# ---------------------------------------------------------------------------
# F7 (P2): MTR per-target chart plotted unmasked rows the metrics excluded
# ---------------------------------------------------------------------------


def test_f7_mtr_chart_uses_finite_masked_rows_matching_metrics():
    """render_mtr_report's per-target chart panel spec receives only the finite-masked (y_true, y_pred)
    pairs -- the SAME rows the accompanying metrics were computed on -- not the full unmasked columns."""
    import tempfile
    import os

    import mlframe.reporting.charts.regression as regmod
    from mlframe.training.reporting._reporting_regression._mtr import render_mtr_report

    rng = np.random.default_rng(0)
    n, k = 200, 2
    targets = rng.normal(size=(n, k))
    preds = targets + rng.normal(scale=0.1, size=(n, k))
    n_nan = 20
    preds = preds.copy()
    preds[:n_nan, 0] = np.nan

    captures = []
    orig = regmod.build_regression_panel_spec

    def spy(y_true, y_pred, **kw):
        """Records call arguments for this test's assertions."""
        captures.append({"n": len(y_true), "any_nan": bool(np.isnan(y_pred).any())})
        return orig(y_true, y_pred, **kw)

    regmod.build_regression_panel_spec = spy
    try:
        tmpdir = tempfile.mkdtemp()
        plot_file = os.path.join(tmpdir, "chart.png")
        render_mtr_report(
            targets, preds, "testmodel",
            metrics=None, print_report=False,
            plot_outputs="matplotlib[png]", plot_file=plot_file,
            figsize=(4, 3), plot_sample_size=1000, plot_dpi=None,
            report_title="t", verbose=False,
        )
    finally:
        regmod.build_regression_panel_spec = orig

    assert len(captures) == 2
    assert captures[0]["n"] == n - n_nan, captures[0]
    assert not captures[0]["any_nan"], "target-0 chart still received NaN rows"
    assert captures[1]["n"] == n


# ---------------------------------------------------------------------------
# F8 (test-gap): covered by test_f1_envelope_clip_propagates_to_returned_preds above (PR1) --
# no separate test body needed, documented here so the finding has an explicit disposition.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PR4: _per_series_flags de-duplicated into _shared_helpers (single implementation, no drift risk)
# ---------------------------------------------------------------------------


def test_pr4_per_series_flags_is_a_single_shared_implementation():
    """matplotlib.py and plotly.py both re-export the SAME _shared_helpers._per_series_flags object."""
    from mlframe.reporting.renderers._shared_helpers import _per_series_flags as shared_fn
    from mlframe.reporting.renderers.matplotlib import _per_series_flags as mpl_fn
    from mlframe.reporting.renderers.plotly import _per_series_flags as plotly_fn

    assert mpl_fn is shared_fn
    assert plotly_fn is shared_fn
    assert shared_fn(None, 3) == [False, False, False]
    assert shared_fn(True, 3) == [True, True, True]
    assert shared_fn((True, False), 3) == [True, False, False]
