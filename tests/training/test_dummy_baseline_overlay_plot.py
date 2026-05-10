"""Tests for the pre-training dummy-baseline overlay plot.

User-facing rationale: shows the no-model floor BEFORE any
``process_model()`` fires, so the operator can eyeball
predictions-vs-actual + residual distribution for the strongest
trivial baseline (e.g. ``mean`` for regression, ``majority`` for
binary classification). Default ON, gated on
``DummyBaselinesConfig.plot_strongest``.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mlframe.training.dummy_baselines import (
    compute_dummy_baselines,
    plot_best_dummy_baseline_overlay,
    BaselineReport,
)
from mlframe.training.configs import DummyBaselinesConfig


def _make_regression_data(n_train: int = 500, n_val: int = 200,
                            n_test: int = 200, seed: int = 0):
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
            train_X=train_X, val_X=val_X, test_X=test_X,
            train_y=train_y, val_y=val_y, test_y=test_y,
            config=DummyBaselinesConfig(),
        )
        assert report.strongest is not None
        assert "strongest_val_preds" in report.extras, (
            "regression: strongest baseline val predictions should "
            "be exposed via extras for the overlay plotter")
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
            train_X=train_X, val_X=val_X, test_X=test_X,
            train_y=train_y, val_y=val_y, test_y=test_y,
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
            train_X=train_X, val_X=val_X, test_X=test_X,
            train_y=train_y, val_y=val_y, test_y=test_y,
            config=DummyBaselinesConfig(),
        )
        out = str(tmp_path / "baseline_floor.png")
        fig = plot_best_dummy_baseline_overlay(
            report, val_y=val_y, test_y=test_y, save_path=out, show=False,
        )
        assert fig is not None
        # Saved PNG exists + non-empty.
        import os
        assert os.path.exists(out)
        assert os.path.getsize(out) > 1000
        # Figure has 2 panels for regression (scatter + resid hist).
        axes = fig.axes
        assert len(axes) == 2, (
            f"regression overlay should have 2 axes (scatter + "
            f"residual hist); got {len(axes)}")
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
            n_train=0, n_val=0, n_test=0,
            n_train_finite=0, n_val_finite=0, n_test_finite=0,
            extras={},
        )
        out = plot_best_dummy_baseline_overlay(
            empty_report, val_y=None, test_y=None,
            save_path=str(tmp_path / "skip.png"),
            show=False,
        )
        assert out is None


class TestConfigPlotStrongestDefaultOn:
    """User-facing contract: default ON per their repeated request."""

    def test_plot_strongest_default_true(self) -> None:
        cfg = DummyBaselinesConfig()
        assert cfg.plot_strongest is True, (
            "regression: per user feedback this default should be ON; "
            "manual opt-out via plot_strongest=False")


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
            n_train=100, n_val=50, n_test=0,
            n_train_finite=100, n_val_finite=50, n_test_finite=0,
            extras={
                "strongest_val_preds": np.full(50, 10.0),
                # No test preds.
            },
        )
        val_y = np.random.default_rng(0).normal(loc=10, size=50)
        fig = plot_best_dummy_baseline_overlay(
            report, val_y=val_y, test_y=None,
            save_path=str(tmp_path / "val_only.png"),
            show=False,
        )
        assert fig is not None
