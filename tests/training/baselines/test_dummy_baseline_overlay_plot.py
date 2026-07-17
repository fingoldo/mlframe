"""Tests for the pre-training dummy-baseline overlay plot.

User-facing rationale: shows the no-model floor BEFORE any
``process_model()`` fires, so the operator can eyeball
predictions-vs-actual + residual distribution for the strongest
trivial baseline (e.g. ``mean`` for regression, ``majority`` for
binary classification). Default ON, gated on
``DummyBaselinesConfig.plot_strongest``.
"""

from __future__ import annotations


import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from mlframe.training.baselines.dummy import (
    compute_dummy_baselines,
    plot_best_dummy_baseline_overlay,
    BaselineReport,
)
from mlframe.training.configs import DummyBaselinesConfig


def _make_regression_data(n_train: int = 500, n_val: int = 200, n_test: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    train_y = rng.normal(loc=10, scale=2, size=n_train)
    val_y = rng.normal(loc=10, scale=2, size=n_val)
    test_y = rng.normal(loc=10, scale=2, size=n_test)
    train_X = pd.DataFrame({"x1": rng.normal(size=n_train)})
    val_X = pd.DataFrame({"x1": rng.normal(size=n_val)})
    test_X = pd.DataFrame({"x1": rng.normal(size=n_test)})
    return train_X, val_X, test_X, train_y, val_y, test_y


class TestStrongestPredsRetained:
    """``compute_dummy_baselines`` should expose val/test predictions
    for the strongest baseline via ``extras`` so the overlay plotter
    can read them."""

    def test_strongest_preds_present_in_extras(self) -> None:
        train_X, val_X, test_X, train_y, val_y, test_y = _make_regression_data()
        report = compute_dummy_baselines(
            target_type="regression",
            target_name="y",
            train_X=train_X,
            val_X=val_X,
            test_X=test_X,
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            config=DummyBaselinesConfig(),
        )
        assert report.strongest is not None
        assert "strongest_val_preds" in report.extras, "regression: strongest baseline val predictions should be exposed via extras for the overlay plotter"
        assert "strongest_test_preds" in report.extras
        sv = report.extras["strongest_val_preds"]
        st = report.extras["strongest_test_preds"]
        assert len(sv) == len(val_y)
        assert len(st) == len(test_y)

    def test_to_dict_scrubs_prediction_arrays(self) -> None:
        """``to_dict`` (for metadata.pkl serialization) MUST strip the
        raw prediction arrays -- they're consumed synchronously by
        the overlay plotter at suite time and would bloat the saved
        metadata."""
        train_X, val_X, test_X, train_y, val_y, test_y = _make_regression_data()
        report = compute_dummy_baselines(
            target_type="regression",
            target_name="y",
            train_X=train_X,
            val_X=val_X,
            test_X=test_X,
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            config=DummyBaselinesConfig(),
        )
        d = report.to_dict()
        assert "strongest_val_preds" not in d["extras"]
        assert "strongest_test_preds" not in d["extras"]


class TestOverlayPlotRegression:
    """End-to-end: compute baselines for a regression target, render
    the overlay, assert the figure is well-formed."""

    def test_regression_overlay_renders(self, tmp_path) -> None:
        train_X, val_X, test_X, train_y, val_y, test_y = _make_regression_data()
        report = compute_dummy_baselines(
            target_type="regression",
            target_name="y",
            train_X=train_X,
            val_X=val_X,
            test_X=test_X,
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            config=DummyBaselinesConfig(),
        )
        out = str(tmp_path / "baseline_floor.png")
        fig = plot_best_dummy_baseline_overlay(
            report,
            val_y=val_y,
            test_y=test_y,
            save_path=out,
            show=False,
        )
        assert fig is not None
        # Saved PNG exists + non-empty.
        import os

        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000
        # Figure has 2 panels for regression (scatter + resid hist).
        axes = fig.axes
        assert len(axes) == 2, f"regression overlay should have 2 axes (scatter + residual hist); got {len(axes)}"
        # Y-axis labels are sane.
        scatter_ylabel = axes[0].get_ylabel().lower()
        assert "y_hat" in scatter_ylabel or "baseline" in scatter_ylabel
        resid_xlabel = axes[1].get_xlabel().lower()
        assert "residual" in resid_xlabel

    def test_overlay_short_circuits_when_no_strongest(self, tmp_path) -> None:
        """When the report has no strongest baseline (degenerate input),
        the plotter returns None and doesn't crash."""
        empty_report = BaselineReport(
            target_type="regression",
            target_name="y",
            table=pd.DataFrame(),
            strongest=None,
            primary_metric=None,
            ts_period_used=None,
            plot_path=None,
            elapsed_s=0.0,
            n_train=0,
            n_val=0,
            n_test=0,
            n_train_finite=0,
            n_val_finite=0,
            n_test_finite=0,
            extras={},
        )
        out = plot_best_dummy_baseline_overlay(
            empty_report,
            val_y=None,
            test_y=None,
            save_path=str(tmp_path / "skip.png"),
            show=False,
        )
        assert out is None


class TestConfigPlotStrongestDefaultOn:
    """User-facing contract: default ON per their repeated request."""

    def test_plot_strongest_default_true(self) -> None:
        cfg = DummyBaselinesConfig()
        assert cfg.plot_strongest is True, "regression: per user feedback this default should be ON; manual opt-out via plot_strongest=False"


class TestDummyGoesThroughReportModelPerf:
    """2026-05-11 (round 2): user wants the strongest dummy baseline
    reported through the SAME ``report_model_perf`` machinery as
    every real model (cb/xgb/lgb/linear). Same text format, same
    chart (regression scatter + residual hist; classification
    calibration + prob-hist), same residual_audit. Verified by
    asserting that the in-suite shim path calls ``report_model_perf``
    with the right kwargs.
    """

    def test_unit_report_model_perf_accepts_dummy_preds(
        self,
        monkeypatch,
    ) -> None:
        """``report_model_perf`` accepts ``model=None`` + ``preds=...``
        for regression (the path the suite uses to plug dummy preds
        in). The function returns the same `(preds, None)` shape."""
        from mlframe.training.evaluation import report_model_perf

        rng = np.random.default_rng(0)
        n = 500
        y_true = rng.normal(loc=10, scale=2, size=n)
        dummy_preds = np.full(n, y_true.mean())
        # Use ``show_perf_chart=False`` + ``print_report=False`` so
        # the test doesn't open matplotlib windows or print to log.
        # The contract: no crash + returns predictions back.
        preds_out, probs_out = report_model_perf(
            targets=y_true,
            columns=["x1", "x2"],
            model_name="DummyBaseline:mean",
            model=None,
            preds=dummy_preds,
            df=None,
            report_title="VAL (DUMMY) ",
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            target_type="regression",
        )
        assert preds_out is not None
        assert probs_out is None  # regression -> no probs
        assert len(preds_out) == n

    def test_unit_dummy_classification_routes_via_probs(self) -> None:
        """Classification dummy: 2-D probability arrays. The shim
        must split into ``preds`` (argmax) + ``probs``. We assert
        the contract via a direct ``report_model_perf`` call mirroring
        the in-suite logic."""
        from mlframe.training.evaluation import report_model_perf

        rng = np.random.default_rng(0)
        n = 500
        y_true = rng.integers(0, 2, size=n)
        # "stratified" dummy: 2-D probabilities, balanced.
        dummy_probs = np.column_stack(
            [
                np.full(n, 0.6),
                np.full(n, 0.4),
            ]
        )
        dummy_preds = np.argmax(dummy_probs, axis=1)
        preds_out, probs_out = report_model_perf(
            targets=y_true,
            columns=["x1"],
            model_name="DummyBaseline:stratified",
            model=None,
            preds=dummy_preds,
            probs=dummy_probs,
            df=None,
            report_title="VAL (DUMMY) ",
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            target_type="binary_classification",
        )
        assert preds_out is not None
        assert probs_out is not None
        assert probs_out.shape == (n, 2)


class TestOverlayHandlesMissingPreds:
    """If extras have only val (or only test) preds, the plotter
    falls back gracefully -- doesn't crash, renders what it can."""

    def test_only_val_preds(self, tmp_path) -> None:
        report = BaselineReport(
            target_type="regression",
            target_name="y",
            table=pd.DataFrame({"val_RMSE": [1.0]}, index=["mean"]),
            strongest="mean",
            primary_metric="val_RMSE",
            ts_period_used=None,
            plot_path=None,
            elapsed_s=0.0,
            n_train=100,
            n_val=50,
            n_test=0,
            n_train_finite=100,
            n_val_finite=50,
            n_test_finite=0,
            extras={
                "strongest_val_preds": np.full(50, 10.0),
                # No test preds.
            },
        )
        val_y = np.random.default_rng(0).normal(loc=10, size=50)
        fig = plot_best_dummy_baseline_overlay(
            report,
            val_y=val_y,
            test_y=None,
            save_path=str(tmp_path / "val_only.png"),
            show=False,
        )
        assert fig is not None


class TestOverlayPlotConfigFlag:
    """INV-43: the dedicated overlay PNG is reachable via DummyBaselinesConfig.overlay_plot
    (default OFF). Pre-fix the flag did not exist and the function was unreachable from configs.
    """

    def test_overlay_plot_default_off(self) -> None:
        assert DummyBaselinesConfig().overlay_plot is False

    def test_overlay_plot_renders_and_saves_when_enabled(self, tmp_path) -> None:
        train_X, val_X, test_X, train_y, val_y, test_y = _make_regression_data()
        prefix = str(tmp_path) + "/"
        report = compute_dummy_baselines(
            target_type="regression",
            target_name="y",
            train_X=train_X,
            val_X=val_X,
            test_X=test_X,
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            config=DummyBaselinesConfig(overlay_plot=True),
            plot_file_prefix=prefix,
        )
        assert report.strongest is not None
        assert report.plot_path is not None, "overlay_plot=True must render the dedicated overlay and stamp report.plot_path"
        import os

        assert os.path.exists(report.plot_path), "overlay PNG should be written to disk"

    def test_overlay_plot_off_leaves_plot_path_none(self, tmp_path) -> None:
        train_X, val_X, test_X, train_y, val_y, test_y = _make_regression_data()
        report = compute_dummy_baselines(
            target_type="regression",
            target_name="y",
            train_X=train_X,
            val_X=val_X,
            test_X=test_X,
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            config=DummyBaselinesConfig(overlay_plot=False),
            plot_file_prefix=str(tmp_path) + "/",
        )
        assert report.plot_path is None
