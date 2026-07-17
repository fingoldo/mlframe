"""Integration tests for the _reporting.py wiring tail (INV-37 / INV-52 / INV-56).

INV-37: multi-target panel-grid suptitle carries the [NF / M rows] shape annotation.
INV-52: _maybe_display logs a plain-text frame rendering when no IPython kernel is present.
INV-56: saved chart paths are stamped into metrics["charts"]["paths"]; keep_figure_handles retains the pure-data spec.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pytest

from mlframe.training.reporting import _reporting as R


# ---------------------------------------------------------------------------
# INV-52: _maybe_display logs a text frame when no IPython kernel is present
# ---------------------------------------------------------------------------


def test_maybe_display_logs_frame_when_no_ipython(monkeypatch, caplog):
    monkeypatch.setattr(R, "_ipython_display", None)
    df = pd.DataFrame({"segment": ["a", "b"], "acc": [0.9, 0.6]})
    with caplog.at_level(logging.INFO, logger=R.logger.name):
        R._maybe_display(df)
    joined = "\n".join(rec.message for rec in caplog.records)
    assert "segment" in joined and "acc" in joined


def test_maybe_display_logs_styler_underlying_frame(monkeypatch, caplog):
    monkeypatch.setattr(R, "_ipython_display", None)
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    styler = R._style_with_caption(df, "caption")  # a Styler when jinja2 present, else the bare frame
    with caplog.at_level(logging.INFO, logger=R.logger.name):
        R._maybe_display(styler)
    joined = "\n".join(rec.message for rec in caplog.records)
    assert "x" in joined and "y" in joined


def test_maybe_display_ignores_non_frame(monkeypatch, caplog):
    monkeypatch.setattr(R, "_ipython_display", None)
    with caplog.at_level(logging.INFO, logger=R.logger.name):
        R._maybe_display("not a frame")  # str has no .to_string returning a frame layout
    # No crash; a plain string has no DataFrame.to_string, so nothing meaningful is logged.
    assert R._frame_to_text("not a frame") is None


def test_frame_to_text_handles_dataframe():
    df = pd.DataFrame({"a": [1]})
    out = R._frame_to_text(df)
    assert out is not None and "a" in out


# ---------------------------------------------------------------------------
# INV-56: keep_figure_handles + chart path accounting via the training-curve path
# ---------------------------------------------------------------------------


class _FakeLGB:
    def __init__(self, evals, best_iter):
        self.evals_result_ = evals
        self.best_iteration_ = best_iter


def _synthetic_evals(n=60, es=40):
    it = np.arange(n)
    train = 1.0 / (1.0 + it)
    val = 1.0 / (1.0 + it) + np.maximum(0.0, (it - es)) * 0.002
    return {"valid_0": {"l2": train.tolist()}, "valid_1": {"l2": val.tolist()}}, es


def test_training_curve_stamps_path_and_keeps_spec(tmp_path):
    from mlframe.reporting.spec import FigureSpec

    evals, es = _synthetic_evals()
    base = os.path.join(str(tmp_path), "m")

    class _Cfg:
        training_curves = True
        keep_figure_handles = True

    metrics: dict = {}
    R._render_training_curves(
        _FakeLGB(evals, es),
        model_name="LGB",
        plot_file=base,
        plot_outputs="matplotlib[png]",
        plot_dpi=None,
        metrics=metrics,
        reporting_config=_Cfg(),
    )
    assert "paths" in metrics["charts"]
    assert any("training_curve" in p for p in metrics["charts"]["paths"])
    assert "training_curve" in metrics.get("figure_specs", {})
    assert isinstance(metrics["figure_specs"]["training_curve"], FigureSpec)


def test_keep_figure_handles_off_by_default(tmp_path):
    evals, es = _synthetic_evals()

    class _Cfg:
        training_curves = True
        # no keep_figure_handles attr -> getattr default False

    metrics: dict = {}
    R._render_training_curves(
        _FakeLGB(evals, es),
        model_name="LGB",
        plot_file=os.path.join(str(tmp_path), "m"),
        plot_outputs="matplotlib[png]",
        plot_dpi=None,
        metrics=metrics,
        reporting_config=_Cfg(),
    )
    assert "figure_specs" not in metrics


def test_kept_figure_spec_is_picklable(tmp_path):
    import pickle

    evals, es = _synthetic_evals()

    class _Cfg:
        training_curves = True
        keep_figure_handles = True

    metrics: dict = {}
    R._render_training_curves(
        _FakeLGB(evals, es),
        model_name="LGB",
        plot_file=os.path.join(str(tmp_path), "m"),
        plot_outputs="matplotlib[png]",
        plot_dpi=None,
        metrics=metrics,
        reporting_config=_Cfg(),
    )
    # Pure-data spec, no live figure handle -> round-trips through pickle.
    blob = pickle.dumps(metrics["figure_specs"]["training_curve"])
    assert pickle.loads(blob) is not None


# ---------------------------------------------------------------------------
# INV-37: panel-grid suptitle carries the [NF / M rows] annotation
# ---------------------------------------------------------------------------


def test_panel_grid_suptitle_carries_shape_annotation(tmp_path, monkeypatch):
    captured = {}

    def _fake_dispatch(*args, **kwargs):
        captured["suptitle"] = kwargs.get("suptitle")
        return "multiclass"

    import mlframe.reporting.auto_dispatch as ad

    monkeypatch.setattr(ad, "render_multi_target_panels", _fake_dispatch)

    n, K = 300, 3
    rng = np.random.default_rng(0)
    targets = rng.integers(0, K, size=n)
    probs = rng.dirichlet(np.ones(K), size=n)
    metrics: dict = {}
    R.report_model_perf(
        targets=targets,
        columns=[f"f{i}" for i in range(7)],
        model_name="LGB",
        model=None,
        classes=list(range(K)),
        probs=probs,
        preds=None,
        plot_file=os.path.join(str(tmp_path), "m"),
        plot_outputs="matplotlib[png]",
        multiclass_panels="CONFUSION",
        target_type="multiclass_classification",
        n_features=7,
        metrics=metrics,
        show_fi=False,
        print_report=False,
    )
    assert captured.get("suptitle") is not None
    # Shape annotation appended: "<n>F/<rows> rows".
    assert "7F/" in captured["suptitle"]
    assert "rows]" in captured["suptitle"]


# ---------------------------------------------------------------------------
# INV-14: log_chart_summary fires at suite finalize
# ---------------------------------------------------------------------------


def test_log_chart_summary_reports_saved_count(caplog):
    from mlframe.training.core._setup_helpers import log_chart_summary

    metadata = {"charts": {"saved": ["multiclass_panels", "training_curve"], "failed": []}}
    with caplog.at_level(logging.INFO):
        msg = log_chart_summary(metadata, save_charts=True, data_dir="/tmp/run")
    assert "2 chart" in msg
    assert "/tmp/run" in msg


def test_log_chart_summary_hints_when_nothing_saved(caplog):
    from mlframe.training.core._setup_helpers import log_chart_summary

    with caplog.at_level(logging.INFO):
        msg = log_chart_summary({}, save_charts=False, data_dir=None)
    assert "0 charts saved" in msg
    assert "data_dir" in msg
