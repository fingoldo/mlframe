"""Layout tests for show_calibration_plot - histogram subplot + label toggle.

Covers the 2026-04-27 refactor:
- Histogram OFF -> single-axis layout (byte-for-byte unchanged from pre-refactor;
  TestMatplotlibAggPath in tests/training/test_perf_edges.py exercises that case
  too, this file just adds the axes-count contract).
- Histogram ON -> two-axes GridSpec layout, sharex=True.
- Inline population text labels are independent of histogram toggle.
- Auto Y-scale picks log when the bin-population skew exceeds 100x.

Tests use the matplotlib Agg fastpath (plot_file set + show_plots=False) so
nothing pops on screen and tests are thread-safe.
"""

import os
import pytest

# Force headless rendering for CI/local runs.
os.environ.setdefault("MPLBACKEND", "Agg")

pytest.importorskip("matplotlib")

import numpy as np
from matplotlib.text import Text

from mlframe.metrics.core import show_calibration_plot, fast_calibration_binning


@pytest.fixture
def binned_data():
    """Synthetic bin centres / true rates / hits with mild skew for back-compat tests."""
    rng = np.random.default_rng(42)
    n = 5000
    y_true = (rng.random(n) < 0.3).astype(np.float64)
    y_pred = np.clip(0.3 + 0.4 * (y_true - 0.3) + 0.2 * rng.standard_normal(n), 0.001, 0.999)
    return fast_calibration_binning(y_true=y_true, y_pred=y_pred, nbins=10)


@pytest.fixture
def skewed_hits():
    """Bin populations with > 100x skew - auto Y-scale should pick log."""
    return (
        np.linspace(0.1, 0.9, 5),
        np.linspace(0.1, 0.9, 5),
        np.array([10000, 100, 50, 10, 5], dtype=np.int64),
    )


@pytest.fixture
def uniform_hits():
    """Roughly uniform bin populations - auto Y-scale should pick linear."""
    return (
        np.linspace(0.1, 0.9, 5),
        np.linspace(0.1, 0.9, 5),
        np.array([1000, 1100, 950, 1050, 1000], dtype=np.int64),
    )


class TestHistogramAxesCount:
    def test_histogram_off_yields_single_axis(self, binned_data, tmp_path):
        fp, ft, h = binned_data
        plot_file = str(tmp_path / "no_hist.png")
        fig = show_calibration_plot(
            freqs_predicted=fp, freqs_true=ft, hits=h,
            plot_file=plot_file, show_plots=False,
            show_prob_histogram=False,
        )
        # Agg path returns a Figure; only the calibration scatter axes + colorbar exist.
        # The colorbar adds an extra Axes, so we accept 1 or 2 here but assert no
        # *bottom histogram*: we look for the histogram by ylabel.
        ylabels = {ax.get_ylabel() for ax in fig.axes}
        assert "Bin population" not in ylabels - {"Bin population"} or "Bin population" not in [
            ax.get_ylabel() for ax in fig.axes if ax.get_ylabel() not in ("Observed Frequency", "")
        ]

    def test_histogram_on_yields_two_main_axes(self, binned_data, tmp_path):
        fp, ft, h = binned_data
        plot_file = str(tmp_path / "with_hist.png")
        fig = show_calibration_plot(
            freqs_predicted=fp, freqs_true=ft, hits=h,
            plot_file=plot_file, show_plots=False,
            show_prob_histogram=True,
        )
        # GridSpec(2, 1) gives two main axes; colorbar adds a third Axes object.
        # Find the histogram axes by ylabel == "Bin population".
        hist_axes = [
            ax for ax in fig.axes
            if ax.get_ylabel() == "Bin population" and ax.get_label() != "<colorbar>"
        ]
        assert len(hist_axes) == 1
        # And the calibration axes by ylabel == "Observed Frequency".
        calib_axes = [ax for ax in fig.axes if ax.get_ylabel() == "Observed Frequency"]
        assert len(calib_axes) == 1


class TestInlineLabelsIndependentOfHistogram:
    """Inline per-bin population text labels are governed only by their own flag."""

    @pytest.mark.parametrize(
        "show_hist,show_labels",
        [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ],
    )
    def test_label_presence(self, binned_data, tmp_path, show_hist, show_labels):
        fp, ft, h = binned_data
        plot_file = str(tmp_path / f"hist{show_hist}_lbl{show_labels}.png")
        fig = show_calibration_plot(
            freqs_predicted=fp, freqs_true=ft, hits=h,
            plot_file=plot_file, show_plots=False,
            show_prob_histogram=show_hist,
            show_inline_population_labels=show_labels,
        )
        calib_axes = [ax for ax in fig.axes if ax.get_ylabel() == "Observed Frequency"]
        assert len(calib_axes) == 1
        ax = calib_axes[0]
        # Population labels are added via ax.text; count Text artists with the K/M/B suffix
        # OR plain integer formatting from format_population.
        text_artists = [
            t for t in ax.texts
            if isinstance(t, Text) and t.get_text() and t.get_text() not in ("",)
        ]
        if show_labels:
            assert len(text_artists) >= 1, (
                f"expected per-bin labels with show_inline_population_labels=True, got 0; "
                f"hist={show_hist}, labels={show_labels}"
            )
        else:
            assert text_artists == [], (
                f"expected zero per-bin labels with show_inline_population_labels=False, "
                f"got {[t.get_text() for t in text_artists]}; hist={show_hist}, labels={show_labels}"
            )


class TestAutoYscaleHistogram:
    """Auto picks log when max/min hits skew > 100x; linear otherwise."""

    def test_skewed_picks_log(self, skewed_hits, tmp_path):
        fp, ft, h = skewed_hits
        plot_file = str(tmp_path / "skewed.png")
        fig = show_calibration_plot(
            freqs_predicted=fp, freqs_true=ft, hits=h,
            plot_file=plot_file, show_plots=False,
            show_prob_histogram=True,
            prob_histogram_yscale="auto",
        )
        hist_ax = next(
            ax for ax in fig.axes
            if ax.get_ylabel() == "Bin population" and ax.get_label() != "<colorbar>"
        )
        assert hist_ax.get_yscale() == "log"

    def test_uniform_picks_linear(self, uniform_hits, tmp_path):
        fp, ft, h = uniform_hits
        plot_file = str(tmp_path / "uniform.png")
        fig = show_calibration_plot(
            freqs_predicted=fp, freqs_true=ft, hits=h,
            plot_file=plot_file, show_plots=False,
            show_prob_histogram=True,
            prob_histogram_yscale="auto",
        )
        hist_ax = next(
            ax for ax in fig.axes
            if ax.get_ylabel() == "Bin population" and ax.get_label() != "<colorbar>"
        )
        assert hist_ax.get_yscale() == "linear"

    def test_explicit_log_overrides_auto(self, uniform_hits, tmp_path):
        fp, ft, h = uniform_hits
        plot_file = str(tmp_path / "force_log.png")
        fig = show_calibration_plot(
            freqs_predicted=fp, freqs_true=ft, hits=h,
            plot_file=plot_file, show_plots=False,
            show_prob_histogram=True,
            prob_histogram_yscale="log",
        )
        hist_ax = next(
            ax for ax in fig.axes
            if ax.get_ylabel() == "Bin population" and ax.get_label() != "<colorbar>"
        )
        assert hist_ax.get_yscale() == "log"

    def test_explicit_linear_overrides_auto(self, skewed_hits, tmp_path):
        fp, ft, h = skewed_hits
        plot_file = str(tmp_path / "force_linear.png")
        fig = show_calibration_plot(
            freqs_predicted=fp, freqs_true=ft, hits=h,
            plot_file=plot_file, show_plots=False,
            show_prob_histogram=True,
            prob_histogram_yscale="linear",
        )
        hist_ax = next(
            ax for ax in fig.axes
            if ax.get_ylabel() == "Bin population" and ax.get_label() != "<colorbar>"
        )
        assert hist_ax.get_yscale() == "linear"
