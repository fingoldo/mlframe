"""Tests for the 3 chart-spec builders (calibration / regression / temporal)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.reporting.charts import (
    build_calibration_spec, build_regression_panel_spec,
    build_temporal_audit_spec,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    FigureSpec, HistogramPanelSpec, ScatterPanelSpec, LinePanelSpec,
)


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------


class TestCalibrationSpec:
    def test_returns_2_panel_grid_when_histogram_on(self):
        spec = build_calibration_spec(
            freqs_predicted=np.linspace(0.05, 0.95, 10),
            freqs_true=np.linspace(0.0, 1.0, 10),
            hits=np.array([1000, 800, 600, 400, 300, 200, 150, 100, 80, 50]),
            plot_title="test",
            show_prob_histogram=True,
        )
        assert isinstance(spec, FigureSpec)
        # Two rows: scatter + histogram
        assert len(spec.panels) == 2
        assert isinstance(spec.panels[0][0], ScatterPanelSpec)
        assert isinstance(spec.panels[1][0], HistogramPanelSpec)

    def test_returns_1_panel_when_histogram_off(self):
        spec = build_calibration_spec(
            freqs_predicted=np.array([0.1, 0.5, 0.9]),
            freqs_true=np.array([0.05, 0.5, 0.95]),
            hits=np.array([100, 200, 50]),
            show_prob_histogram=False,
        )
        assert len(spec.panels) == 1

    def test_inline_labels_populated(self):
        spec = build_calibration_spec(
            freqs_predicted=np.array([0.1, 0.5]),
            freqs_true=np.array([0.05, 0.5]),
            hits=np.array([1500, 200000]),
            show_inline_population_labels=True,
        )
        scatter = spec.panels[0][0]
        # 1500 -> "1.5K", 200000 -> "200.0K"
        assert scatter.inline_labels is not None
        assert any("K" in t[2] for t in scatter.inline_labels)

    def test_renders_via_matplotlib(self, tmp_path):
        spec = build_calibration_spec(
            freqs_predicted=np.linspace(0.05, 0.95, 10),
            freqs_true=np.linspace(0.0, 1.0, 10),
            hits=np.array([1000, 800, 600, 400, 300, 200, 150, 100, 80, 50]),
        )
        render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"),
                        str(tmp_path / "calib"))
        assert os.path.exists(tmp_path / "calib.png")

    def test_renders_via_plotly(self, tmp_path):
        spec = build_calibration_spec(
            freqs_predicted=np.linspace(0.05, 0.95, 10),
            freqs_true=np.linspace(0.0, 1.0, 10),
            hits=np.array([1000, 800, 600, 400, 300, 200, 150, 100, 80, 50]),
        )
        render_and_save(spec, parse_plot_output_dsl("plotly[html]"),
                        str(tmp_path / "calib"))
        assert os.path.exists(tmp_path / "calib.html")


# ----------------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------------


def _fake_audit():
    """Duck-typed ResidualAudit for testing without computing one."""
    return SimpleNamespace(
        mean=0.05, std=0.5, skew=0.1, excess_kurt=2.5,
        hypothesis="Normal", suggested_loss="MSE (Normal-MLE)",
        hetero_significant=False, hetero_spearman=0.02,
    )


class TestRegressionSpec:
    def test_returns_4_panel_grid(self):
        """The default template restores RESID_VS_PRED and adds ERR_BY_DECILE,
        so the legacy adapter now packs SCATTER + RESID_HIST + RESID_VS_PRED +
        ERR_BY_DECILE into a 2x2 grid (scatter + residual hist on the top row)."""
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(200)
        y_pred = y_true + rng.standard_normal(200) * 0.2
        spec = build_regression_panel_spec(
            y_true, y_pred, audit=_fake_audit(),
            header_str="VAL CB ...", metrics_str="MAE=1.2 R2=0.95",
        )
        assert spec.suptitle == "VAL CB ..."
        n_panels = sum(1 for row in spec.panels for c in row if c is not None)
        assert n_panels == 4
        assert isinstance(spec.panels[0][0], ScatterPanelSpec)
        assert isinstance(spec.panels[0][1], HistogramPanelSpec)

    def test_scatter_title_carries_metrics_and_spearman(self):
        """The Spearman / heteroscedasticity diagnostic moved out of the
        dropped 3rd panel into the scatter title (newline-separated)."""
        spec = build_regression_panel_spec(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 1.9, 3.2]),
            audit=_fake_audit(),
            header_str="suptitle here",
            metrics_str="MAE=0.1 RMSE=0.15",
        )
        title = spec.panels[0][0].title
        assert "MAE=0.1 RMSE=0.15" in title
        # Heteroscedasticity is about residual MAGNITUDE vs prediction, so the
        # diagnostic correlates |resid| (not signed resid) against preds.
        assert "Spearman(|resid|,preds)" in title

    def test_scatter_title_metrics_only_when_audit_missing(self):
        """When the caller passes ``audit=None`` the scatter title is
        exactly the metrics string -- no Spearman line is fabricated."""
        spec = build_regression_panel_spec(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.1, 1.9, 3.2]),
            audit=None,
            header_str="suptitle here",
            metrics_str="MAE=0.1 RMSE=0.15",
        )
        assert spec.panels[0][0].title == "MAE=0.1 RMSE=0.15"

    def test_histogram_title_includes_hypothesis_and_suggested(self):
        spec = build_regression_panel_spec(
            np.random.default_rng(0).standard_normal(50),
            np.random.default_rng(1).standard_normal(50),
            audit=_fake_audit(),
        )
        hist_title = spec.panels[0][1].title
        assert "skew" in hist_title
        assert "excess_kurt" in hist_title
        assert "hypothesis: Normal" in hist_title
        assert "suggested: MSE" in hist_title

    def test_renders_via_plotly(self, tmp_path):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        spec = build_regression_panel_spec(
            y, y + 0.1, audit=_fake_audit(),
            header_str="hdr", metrics_str="MAE=0.1",
        )
        render_and_save(spec, parse_plot_output_dsl("plotly[html]"),
                        str(tmp_path / "reg"))
        assert os.path.exists(tmp_path / "reg.html")


# ----------------------------------------------------------------------------
# Temporal audit
# ----------------------------------------------------------------------------


class TestTemporalSpec:
    def test_returns_single_line_panel(self):
        audit = SimpleNamespace(
            time_bins=np.arange(20),
            rates=np.random.default_rng(0).uniform(0.4, 0.6, 20),
            target_name="cl_total_hired",
            granularity="day",
            target_type="binary_classification",
            segments=[],
        )
        spec = build_temporal_audit_spec(audit)
        assert len(spec.panels) == 1
        assert isinstance(spec.panels[0][0], LinePanelSpec)

    def test_segments_in_title(self):
        rng = np.random.default_rng(0)
        rates = rng.uniform(0.4, 0.6, 20)
        bins = [
            SimpleNamespace(bin_start=float(i), target_rate=float(rates[i]), kept=True)
            for i in range(20)
        ]
        audit = SimpleNamespace(
            bins=bins,
            target_name="x", granularity="day",
            target_type="binary_classification",
            timestamp_col="Time",
            segments=[
                {"start_idx": 0, "end_idx": 10, "mean_rate": 0.45},
                {"start_idx": 10, "end_idx": 20, "mean_rate": 0.55},
            ],
            change_point_indices=[10],
        )
        spec = build_temporal_audit_spec(audit)
        assert "2 segment(s)" in spec.panels[0][0].title

    def test_renders_via_plotly(self, tmp_path):
        audit = SimpleNamespace(
            time_bins=np.arange(20), rates=np.linspace(0.1, 0.5, 20),
            target_name="x", granularity="day",
            target_type="binary_classification",
            segments=[],
        )
        spec = build_temporal_audit_spec(audit)
        render_and_save(spec, parse_plot_output_dsl("plotly[html]"),
                        str(tmp_path / "temp"))
        assert os.path.exists(tmp_path / "temp.html")
