"""Tests for the 3 chart-spec builders (calibration / regression / temporal)."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.reporting.charts import (
    build_calibration_spec,
    build_regression_panel_spec,
    build_temporal_audit_spec,
)
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.renderers import render_and_save
from mlframe.reporting.spec import (
    FigureSpec,
    HistogramPanelSpec,
    ScatterPanelSpec,
    LinePanelSpec,
)


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------


class TestCalibrationSpec:
    """Groups tests for: TestCalibrationSpec."""
    def test_returns_2_panel_grid_when_histogram_on(self):
        """Returns 2 panel grid when histogram on."""
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
        """Returns 1 panel when histogram off."""
        spec = build_calibration_spec(
            freqs_predicted=np.array([0.1, 0.5, 0.9]),
            freqs_true=np.array([0.05, 0.5, 0.95]),
            hits=np.array([100, 200, 50]),
            show_prob_histogram=False,
        )
        assert len(spec.panels) == 1

    def test_inline_labels_populated(self):
        """Inline labels populated."""
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
        """Renders via matplotlib."""
        spec = build_calibration_spec(
            freqs_predicted=np.linspace(0.05, 0.95, 10),
            freqs_true=np.linspace(0.0, 1.0, 10),
            hits=np.array([1000, 800, 600, 400, 300, 200, 150, 100, 80, 50]),
        )
        render_and_save(spec, parse_plot_output_dsl("matplotlib[png]"), str(tmp_path / "calib"))
        assert os.path.exists(tmp_path / "calib.png")

    def test_renders_via_plotly(self, tmp_path):
        """Renders via plotly."""
        spec = build_calibration_spec(
            freqs_predicted=np.linspace(0.05, 0.95, 10),
            freqs_true=np.linspace(0.0, 1.0, 10),
            hits=np.array([1000, 800, 600, 400, 300, 200, 150, 100, 80, 50]),
        )
        render_and_save(spec, parse_plot_output_dsl("plotly[html]"), str(tmp_path / "calib"))
        assert os.path.exists(tmp_path / "calib.html")

    def test_wilson_ci_band_default_on(self):
        """Wilson ci band default on."""
        spec = build_calibration_spec(
            freqs_predicted=np.array([0.1, 0.5, 0.9]),
            freqs_true=np.array([0.08, 0.52, 0.88]),
            hits=np.array([100, 200, 50]),
        )
        scatter = spec.panels[0][0]
        assert scatter.y_err is not None
        lo, hi = scatter.y_err
        assert np.any(lo > 0) and np.any(hi > 0)

    def test_wilson_ci_band_suppressed_when_off(self):
        """Wilson ci band suppressed when off."""
        spec = build_calibration_spec(
            freqs_predicted=np.array([0.1, 0.5, 0.9]),
            freqs_true=np.array([0.08, 0.52, 0.88]),
            hits=np.array([100, 200, 50]),
            show_wilson_ci=False,
        )
        scatter = spec.panels[0][0]
        assert scatter.y_err is None

    def test_bubble_area_capped_for_dominant_bin(self):
        # One bin holds ~99.9% of the mass. The legacy 5000*h/sum scaling would give it an
        # area ~5000, occluding every neighbour. Above MAX_BUBBLE_AREA the area is sqrt-
        # compressed toward the cap, so the dominant bin is far smaller than the raw value
        # yet still the largest point.
        """Bubble area capped for dominant bin."""
        from mlframe.reporting.charts.calibration import MAX_BUBBLE_AREA

        hits = np.array([1, 1, 1_000_000, 1, 1])
        spec = build_calibration_spec(
            freqs_predicted=np.linspace(0.1, 0.9, 5),
            freqs_true=np.linspace(0.1, 0.9, 5),
            hits=hits,
        )
        sizes = np.asarray(spec.panels[0][0].point_size, dtype=float)
        dominant = sizes[2]
        raw_uncapped = 5000.0 * hits[2] / hits.sum()
        expected = MAX_BUBBLE_AREA * np.sqrt(raw_uncapped / MAX_BUBBLE_AREA)
        assert dominant == pytest.approx(expected, rel=1e-9)
        # Compression must materially shrink it relative to the uncapped scaling.
        assert dominant < 0.5 * raw_uncapped
        assert dominant == sizes.max()

    def test_inline_labels_auto_disabled_above_max_bins(self):
        """Inline labels auto disabled above max bins."""
        from mlframe.reporting.charts.calibration import INLINE_LABEL_MAX_BINS

        nbins = INLINE_LABEL_MAX_BINS + 5
        spec = build_calibration_spec(
            freqs_predicted=np.linspace(0.01, 0.99, nbins),
            freqs_true=np.linspace(0.01, 0.99, nbins),
            hits=np.full(nbins, 100),
            show_inline_population_labels=True,
        )
        assert spec.panels[0][0].inline_labels is None

    def test_inline_labels_kept_at_or_below_max_bins(self):
        """Inline labels kept at or below max bins."""
        from mlframe.reporting.charts.calibration import INLINE_LABEL_MAX_BINS

        nbins = INLINE_LABEL_MAX_BINS
        spec = build_calibration_spec(
            freqs_predicted=np.linspace(0.01, 0.99, nbins),
            freqs_true=np.linspace(0.01, 0.99, nbins),
            hits=np.full(nbins, 100),
            show_inline_population_labels=True,
        )
        assert spec.panels[0][0].inline_labels is not None
        assert len(spec.panels[0][0].inline_labels) == nbins


class TestReliabilityCiToggleWiring:
    """G7: ReportingConfig.reliability_show_ci must reach the chart, not be a dead pipe."""

    def _binned(self):
        """Helper: Binned."""
        rng = np.random.default_rng(0)
        y_pred = rng.random(4000)
        y_true = (rng.random(4000) < y_pred).astype(int)
        return y_true, y_pred

    def _spy_build_calibration_spec(self, monkeypatch, captured):
        # build_calibration_spec is imported lazily inside show_calibration_plot; patch it at
        # its definition module so the lazy import resolves to the spy. Capture the ORIGINAL
        # reference before patching so the spy does not call itself (recursion).
        """Helper: Spy build calibration spec."""
        import mlframe.reporting.charts.calibration as cal_mod

        real = cal_mod.build_calibration_spec

        def _spy(*args, **kwargs):
            """Helper: Spy."""
            captured["show_wilson_ci"] = kwargs.get("show_wilson_ci")
            return real(*args, **kwargs)

        monkeypatch.setattr(cal_mod, "build_calibration_spec", _spy)

    def test_fast_calibration_report_off_suppresses_ci(self, tmp_path, monkeypatch):
        # On pre-fix code fast_calibration_report has no reliability_show_ci parameter at all,
        # and the toggle never reaches build_calibration_spec -> show_wilson_ci stays True.
        """Fast calibration report off suppresses ci."""
        from mlframe.metrics.core import fast_calibration_report

        captured = {}
        self._spy_build_calibration_spec(monkeypatch, captured)

        y_true, y_pred = self._binned()
        fast_calibration_report(
            y_true=y_true,
            y_pred=y_pred,
            nbins=10,
            show_plots=False,
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "cal"),
            reliability_show_ci=False,
        )
        assert captured.get("show_wilson_ci") is False

    def test_fast_calibration_report_default_keeps_ci(self, tmp_path, monkeypatch):
        """Fast calibration report default keeps ci."""
        from mlframe.metrics.core import fast_calibration_report

        captured = {}
        self._spy_build_calibration_spec(monkeypatch, captured)

        y_true, y_pred = self._binned()
        fast_calibration_report(
            y_true=y_true,
            y_pred=y_pred,
            nbins=10,
            show_plots=False,
            plot_outputs="matplotlib[png]",
            base_path=str(tmp_path / "cal"),
        )
        assert captured.get("show_wilson_ci") is True


# ----------------------------------------------------------------------------
# Regression
# ----------------------------------------------------------------------------


def _fake_audit():
    """Duck-typed ResidualAudit for testing without computing one."""
    return SimpleNamespace(
        mean=0.05,
        std=0.5,
        skew=0.1,
        excess_kurt=2.5,
        hypothesis="Normal",
        suggested_loss="MSE (Normal-MLE)",
        hetero_significant=False,
        hetero_spearman=0.02,
    )


class TestRegressionSpec:
    """Groups tests for: TestRegressionSpec."""
    def test_returns_4_panel_grid(self):
        """The default template restores RESID_VS_PRED and adds ERR_BY_DECILE,
        so the legacy adapter now packs SCATTER + RESID_HIST + RESID_VS_PRED +
        ERR_BY_DECILE into a 2x2 grid (scatter + residual hist on the top row)."""
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(200)
        y_pred = y_true + rng.standard_normal(200) * 0.2
        spec = build_regression_panel_spec(
            y_true,
            y_pred,
            audit=_fake_audit(),
            header_str="VAL CB ...",
            metrics_str="MAE=1.2 R2=0.95",
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
        """Histogram title includes hypothesis and suggested."""
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
        """Renders via plotly."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(100)
        spec = build_regression_panel_spec(
            y,
            y + 0.1,
            audit=_fake_audit(),
            header_str="hdr",
            metrics_str="MAE=0.1",
        )
        render_and_save(spec, parse_plot_output_dsl("plotly[html]"), str(tmp_path / "reg"))
        assert os.path.exists(tmp_path / "reg.html")


# ----------------------------------------------------------------------------
# Temporal audit
# ----------------------------------------------------------------------------


class TestTemporalSpec:
    """Groups tests for: TestTemporalSpec."""
    def test_returns_single_line_panel(self):
        """Returns single line panel."""
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
        """Segments in title."""
        rng = np.random.default_rng(0)
        rates = rng.uniform(0.4, 0.6, 20)
        bins = [SimpleNamespace(bin_start=float(i), target_rate=float(rates[i]), kept=True) for i in range(20)]
        audit = SimpleNamespace(
            bins=bins,
            target_name="x",
            granularity="day",
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
        """Renders via plotly."""
        audit = SimpleNamespace(
            time_bins=np.arange(20),
            rates=np.linspace(0.1, 0.5, 20),
            target_name="x",
            granularity="day",
            target_type="binary_classification",
            segments=[],
        )
        spec = build_temporal_audit_spec(audit)
        render_and_save(spec, parse_plot_output_dsl("plotly[html]"), str(tmp_path / "temp"))
        assert os.path.exists(tmp_path / "temp.html")
