"""Wave-5 additive spec vocabulary: error bars, highlight, trend lines, bar orientation/hline, secondary-y,
fill-to-baseline, threshold contours, labeled vspans, datetime vlines, per-series x.

Each new optional field must render through BOTH backends and produce the expected artist / trace, while every
existing spec (no new field set) renders unchanged.
"""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.renderers._trend import robust_fit_endpoints
from mlframe.reporting.spec import (
    BarPanelSpec,
    FigureSpec,
    HeatmapPanelSpec,
    LinePanelSpec,
    ScatterPanelSpec,
)

BACKENDS = ["matplotlib", "plotly"]


def _render(panel, backend):
    return get_renderer(backend).render(FigureSpec(panels=((panel,),), figsize=(6, 4)))


def _scatter_traces(fig):
    return [t for t in fig.data if t.type in ("scatter", "scattergl")]


# ----------------------------------------------------------------------------
# G1 — ScatterPanelSpec.y_err / x_err (Wilson CI bands on the reliability diagram)
# ----------------------------------------------------------------------------


class TestScatterErrorBars:
    @pytest.fixture
    def panel_symmetric(self):
        x = np.linspace(0, 1, 12)
        return ScatterPanelSpec(x=x, y=x, y_err=np.full(12, 0.05))

    @pytest.fixture
    def panel_asymmetric(self):
        # Wilson CIs are asymmetric: separate lower / upper distance arrays.
        x = np.linspace(0, 1, 12)
        return ScatterPanelSpec(x=x, y=x, y_err=(np.full(12, 0.03), np.full(12, 0.07)), x_err=np.full(12, 0.01))

    def test_symmetric_renders_both(self, panel_symmetric):
        for b in BACKENDS:
            assert _render(panel_symmetric, b) is not None

    def test_asymmetric_renders_both(self, panel_asymmetric):
        for b in BACKENDS:
            assert _render(panel_asymmetric, b) is not None

    def test_matplotlib_errorbar_container(self, panel_symmetric):
        fig = _render(panel_symmetric, "matplotlib")
        # ax.errorbar(fmt="none") creates an ErrorbarContainer.
        assert len(fig.axes[0].containers) >= 1

    def test_plotly_error_y_present_symmetric(self, panel_symmetric):
        fig = _render(panel_symmetric, "plotly")
        t0 = _scatter_traces(fig)[0]
        assert t0.error_y is not None and t0.error_y.array is not None
        assert bool(t0.error_y.symmetric) is True

    def test_plotly_error_asymmetric_and_x(self, panel_asymmetric):
        fig = _render(panel_asymmetric, "plotly")
        t0 = _scatter_traces(fig)[0]
        assert bool(t0.error_y.symmetric) is False
        assert t0.error_y.arrayminus is not None
        assert t0.error_x is not None and t0.error_x.array is not None


# ----------------------------------------------------------------------------
# G9 / R-9 — ScatterPanelSpec.highlight_indices + highlight_color (worst-K overlay)
# ----------------------------------------------------------------------------


class TestScatterHighlight:
    @pytest.fixture
    def panel(self):
        x = np.linspace(0, 10, 50)
        return ScatterPanelSpec(x=x, y=x, highlight_indices=np.array([0, 25, 49]), highlight_color="red")

    def test_renders_both(self, panel):
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_plotly_emits_worst_k_trace(self, panel):
        fig = _render(panel, "plotly")
        names = [t.name for t in fig.data]
        assert "worst-K" in names

    def test_matplotlib_extra_scatter_collection(self, panel):
        # base scatter + highlight scatter => >= 2 PathCollections.
        fig = _render(panel, "matplotlib")
        assert len(fig.axes[0].collections) >= 2

    def test_out_of_range_indices_ignored(self):
        x = np.linspace(0, 1, 5)
        panel = ScatterPanelSpec(x=x, y=x, highlight_indices=np.array([-1, 99, 2]))
        for b in BACKENDS:
            assert _render(panel, b) is not None  # only index 2 is valid; no crash


# ----------------------------------------------------------------------------
# G5 / R-15 — trend_line on Scatter + Heatmap
# ----------------------------------------------------------------------------


class TestTrendLine:
    @pytest.mark.parametrize("method", ["theil-sen", "huber"])
    def test_scatter_trend_renders_both(self, method):
        x = np.linspace(0, 10, 60)
        y = 2.0 * x + np.random.default_rng(0).standard_normal(60)
        panel = ScatterPanelSpec(x=x, y=y, trend_line=method, perfect_fit_line=True)
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_scatter_trend_labeled_line_matplotlib(self):
        x = np.linspace(0, 10, 60)
        panel = ScatterPanelSpec(x=x, y=2.0 * x, trend_line="theil-sen")
        fig = _render(panel, "matplotlib")
        labels = [ln.get_label() for ln in fig.axes[0].lines]
        assert any("robust fit" in (lb or "") for lb in labels)

    def test_scatter_trend_named_trace_plotly(self):
        x = np.linspace(0, 10, 60)
        panel = ScatterPanelSpec(x=x, y=2.0 * x, trend_line="huber")
        fig = _render(panel, "plotly")
        assert any("robust fit" in (t.name or "") for t in fig.data)

    def test_heatmap_trend_renders_both(self):
        m = np.random.default_rng(1).random((6, 6))
        x = np.linspace(0, 1, 40)
        y = x + np.random.default_rng(2).standard_normal(40) * 0.1
        panel = HeatmapPanelSpec(matrix=m, row_labels=tuple("abcdef"), col_labels=tuple("123456"), trend_line="theil-sen", trend_xy=(x, y))
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_robust_fit_recovers_slope(self):
        # Theil-Sen should recover ~2x slope despite outliers; endpoints span the data x-range.
        rng = np.random.default_rng(3)
        x = np.linspace(0, 10, 200)
        y = 2.0 * x + rng.standard_normal(200) * 0.1
        y[::20] += 50.0  # gross outliers an OLS line would chase
        ends = robust_fit_endpoints(x, y, "theil-sen")
        (x0, y0), (x1, y1) = ends
        slope = (y1 - y0) / (x1 - x0)
        assert abs(slope - 2.0) < 0.3, f"robust slope {slope} should be near 2.0 despite outliers"

    def test_robust_fit_degenerate_returns_none(self):
        assert robust_fit_endpoints(np.ones(5), np.arange(5.0), "theil-sen") is None  # all x identical
        assert robust_fit_endpoints(np.array([1.0]), np.array([1.0]), "huber") is None  # < 2 points

    def test_robust_fit_large_n_is_bounded(self):
        # Default-ON hexbin overlay can feed millions of points; the fit-cap keeps it fast and still slope-correct.
        import time

        rng = np.random.default_rng(7)
        n = 2_000_000
        x = rng.standard_normal(n) * 5.0
        y = 2.0 * x + rng.standard_normal(n)
        t0 = time.perf_counter()
        ends = robust_fit_endpoints(x, y, "theil-sen")
        elapsed = time.perf_counter() - t0
        assert ends is not None
        (x0, y0), (x1, y1) = ends
        slope = (y1 - y0) / (x1 - x0)
        assert abs(slope - 2.0) < 0.2, f"capped fit slope {slope} should still recover ~2.0"
        assert elapsed < 5.0, f"capped robust fit took {elapsed:.2f}s; should stay well under 5s at 2M points"


# ----------------------------------------------------------------------------
# G2 — BarPanelSpec.orientation
# ----------------------------------------------------------------------------


class TestBarOrientation:
    @pytest.fixture
    def panel(self):
        return BarPanelSpec(categories=("A->B", "C->D", "E->F"), values=np.array([0.4, 0.25, 0.1]), orientation="horizontal")

    def test_renders_both(self, panel):
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_plotly_orientation_h(self, panel):
        fig = _render(panel, "plotly")
        assert fig.data[0].orientation == "h"

    def test_matplotlib_barh_geometry(self, panel):
        # Horizontal bar: width == the value, height == the fixed category thickness (0.8). A vertical bar of
        # the same data has the axes swapped, so comparing the two distinguishes the orientation unambiguously.
        h_patch = _render(panel, "matplotlib").axes[0].patches[0]
        v_panel = BarPanelSpec(categories=panel.categories, values=panel.values)
        v_patch = _render(v_panel, "matplotlib").axes[0].patches[0]
        assert h_patch.get_width() == pytest.approx(panel.values[0])  # value on the x axis
        assert v_patch.get_height() == pytest.approx(panel.values[0])  # value on the y axis

    def test_vertical_default_unchanged(self):
        panel = BarPanelSpec(categories=("a", "b"), values=np.array([1.0, 2.0]))
        fig = _render(panel, "plotly")
        # default orientation v (plotly leaves orientation None/"v")
        assert fig.data[0].orientation in (None, "v")

    def test_grouped_horizontal(self):
        panel = BarPanelSpec(
            categories=("a", "b"), values=(np.array([1.0, 2.0]), np.array([0.5, 1.5])), series_labels=("seg", "global"), orientation="horizontal"
        )
        for b in BACKENDS:
            assert _render(panel, b) is not None


# ----------------------------------------------------------------------------
# G8 — BarPanelSpec.hline (global-metric reference on segments_bar)
# ----------------------------------------------------------------------------


class TestBarHline:
    def test_vertical_hline_renders_both(self):
        panel = BarPanelSpec(categories=("a", "b", "c"), values=np.array([1.0, 2.0, 3.0]), hline=(2.0, "black", "global"))
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_horizontal_hline_renders_both(self):
        panel = BarPanelSpec(categories=("a", "b"), values=np.array([1.0, 2.0]), orientation="horizontal", hline=(1.5, "red", "ref"))
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_matplotlib_reference_line_added(self):
        panel = BarPanelSpec(categories=("a", "b"), values=np.array([1.0, 2.0]), hline=(1.5, "black", "global"))
        fig = _render(panel, "matplotlib")
        # axhline adds a Line2D to ax.lines.
        assert len(fig.axes[0].lines) >= 1

    def test_plotly_hline_shape_added(self):
        panel = BarPanelSpec(categories=("a", "b"), values=np.array([1.0, 2.0]), hline=(1.5, "black", "global"))
        fig = _render(panel, "plotly")
        assert len(fig.layout.shapes) >= 1


# ----------------------------------------------------------------------------
# G3 — LinePanelSpec.secondary_y (COVERAGE width / THRESHOLD queue-rate on a 2nd axis)
# ----------------------------------------------------------------------------


class TestSecondaryY:
    @pytest.fixture
    def panel(self):
        return LinePanelSpec(
            x=np.arange(10),
            y=(np.arange(10.0), np.arange(10.0) * 3.0),
            secondary_y=(False, True),
            secondary_ylabel="queue-rate",
            series_labels=("recall", "queue"),
        )

    def test_renders_both(self, panel):
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_matplotlib_twinx_axis(self, panel):
        fig = _render(panel, "matplotlib")
        # twinx adds a second Axes for the single panel.
        assert len(fig.axes) == 2
        assert fig.axes[1].get_ylabel() == "queue-rate"

    def test_plotly_yaxis2_present(self, panel):
        fig = _render(panel, "plotly")
        assert "yaxis2" in fig.layout
        yaxes = [t.yaxis for t in fig.data]
        assert "y2" in yaxes

    def test_single_bool_secondary_y(self):
        panel = LinePanelSpec(x=np.arange(10), y=np.arange(10.0), secondary_y=True)
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_no_secondary_y_single_axis(self):
        panel = LinePanelSpec(x=np.arange(10), y=np.arange(10.0))
        fig = _render(panel, "matplotlib")
        assert len(fig.axes) == 1  # no twinx when not requested


# ----------------------------------------------------------------------------
# G10 — LinePanelSpec.fill_to_baseline / step_fill (GAIN curve, SCORE_DIST)
# ----------------------------------------------------------------------------


class TestFillToBaseline:
    def test_linear_fill_renders_both(self):
        panel = LinePanelSpec(x=np.arange(10), y=np.arange(10.0), fill_to_baseline=True)
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_step_fill_renders_both(self):
        panel = LinePanelSpec(x=np.arange(10), y=np.arange(10.0), fill_to_baseline=True, step_fill=True)
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_matplotlib_fill_collection(self):
        panel = LinePanelSpec(x=np.arange(10), y=np.arange(10.0), fill_to_baseline=True)
        fig = _render(panel, "matplotlib")
        assert len(fig.axes[0].collections) >= 1  # fill_between -> PolyCollection

    def test_plotly_fill_attr(self):
        panel = LinePanelSpec(x=np.arange(10), y=np.arange(10.0), fill_to_baseline=True)
        fig = _render(panel, "plotly")
        assert any(t.fill is not None for t in fig.data)

    def test_per_series_fill(self):
        panel = LinePanelSpec(x=np.arange(10), y=(np.arange(10.0), np.arange(10.0)), fill_to_baseline=(True, False))
        for b in BACKENDS:
            assert _render(panel, b) is not None


# ----------------------------------------------------------------------------
# G12 — HeatmapPanelSpec.threshold_contours (PSI 0.10 / 0.25 on the drift heatmap)
# ----------------------------------------------------------------------------


class TestThresholdContours:
    @pytest.fixture
    def panel(self):
        m = np.random.default_rng(0).random((6, 6)) * 0.4  # spans below+above 0.10/0.25
        return HeatmapPanelSpec(matrix=m, row_labels=tuple("abcdef"), col_labels=tuple("123456"), threshold_contours=((0.10, "blue"), (0.25, "red")))

    def test_renders_both(self, panel):
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_plotly_contour_traces(self, panel):
        fig = _render(panel, "plotly")
        assert [t.type for t in fig.data].count("contour") == 2

    def test_matplotlib_contour_collection(self, panel):
        fig = _render(panel, "matplotlib")
        # imshow image + at least one contour QuadContourSet collection.
        assert len(fig.axes[0].collections) >= 1

    def test_level_outside_range_skipped(self):
        m = np.full((4, 4), 0.05)  # all below both thresholds -> no contour crosses
        panel = HeatmapPanelSpec(matrix=m, row_labels=tuple("abcd"), col_labels=tuple("1234"), threshold_contours=((0.10, "blue"), (0.25, "red")))
        fig = _render(panel, "plotly")
        assert [t.type for t in fig.data].count("contour") == 0


# ----------------------------------------------------------------------------
# G13 — labeled vspans (regime / split legend label)
# ----------------------------------------------------------------------------


class TestVspanLabels:
    def test_labeled_vspan_renders_both(self):
        panel = LinePanelSpec(x=np.arange(20), y=np.arange(20.0), vspans=((2, 6, "green", 0.2, "train"), (10, 15, "orange", 0.2, "test")))
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_backcompat_unlabeled_4tuple(self):
        panel = LinePanelSpec(x=np.arange(20), y=np.arange(20.0), vspans=((2, 6, "green", 0.2),))
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_plotly_legend_proxy_for_label(self):
        panel = LinePanelSpec(x=np.arange(20), y=np.arange(20.0), vspans=((2, 6, "green", 0.2, "train"),))
        fig = _render(panel, "plotly")
        assert "train" in [t.name for t in fig.data]

    def test_matplotlib_legend_has_label(self):
        panel = LinePanelSpec(x=np.arange(20), y=np.arange(20.0), vspans=((2, 6, "green", 0.2, "regime A"),))
        fig = _render(panel, "matplotlib")
        legend = fig.axes[0].get_legend()
        assert legend is not None
        assert any("regime A" in t.get_text() for t in legend.get_texts())


# ----------------------------------------------------------------------------
# G4 — plotly vline on a datetime x-axis (mpl axvline must still work too)
# ----------------------------------------------------------------------------


class TestDatetimeVline:
    @pytest.fixture
    def dt_panel(self):
        dt = np.array([np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(15)])
        return LinePanelSpec(x=dt, y=np.arange(15.0), x_is_time=True, vlines=((dt[7], "red", "change-point"),))

    def test_plotly_datetime_vline_no_raise(self, dt_panel):
        # add_vline raises on datetime; the shape-based path must render cleanly.
        fig = _render(dt_panel, "plotly")
        assert len(fig.layout.shapes) >= 1

    def test_plotly_datetime_vline_annotation(self, dt_panel):
        fig = _render(dt_panel, "plotly")
        assert any("change-point" == (a.text or "") for a in fig.layout.annotations)

    def test_matplotlib_datetime_axvline_still_works(self, dt_panel):
        fig = _render(dt_panel, "matplotlib")
        assert len(fig.axes[0].lines) >= 1

    def test_plotly_numeric_vline_unchanged(self):
        panel = LinePanelSpec(x=np.arange(20), y=np.arange(20.0), vlines=((5.0, "red", "split"),))
        fig = _render(panel, "plotly")
        assert fig is not None  # numeric add_vline path still works

    def test_python_datetime_vline(self):
        dt = np.array([datetime.datetime(2021, 1, 1) + datetime.timedelta(days=i) for i in range(10)])
        panel = LinePanelSpec(x=dt, y=np.arange(10.0), x_is_time=True, vlines=((dt[5], "blue", "cp"),))
        fig = _render(panel, "plotly")
        assert len(fig.layout.shapes) >= 1


# ----------------------------------------------------------------------------
# G11 — multi-series line with per-series x arrays (adversarial ROC overlay)
# ----------------------------------------------------------------------------


class TestPerSeriesX:
    @pytest.fixture
    def panel(self):
        # Two ROC curves with DIFFERENT fpr grids (train vs test) + a chance diagonal.
        fpr_train = np.array([0.0, 0.1, 0.3, 1.0])
        fpr_test = np.array([0.0, 0.4, 0.7, 1.0])
        chance = np.array([0.0, 1.0])
        return LinePanelSpec(
            x=(fpr_train, fpr_test, chance),
            y=(np.array([0.0, 0.6, 0.85, 1.0]), np.array([0.0, 0.5, 0.75, 1.0]), np.array([0.0, 1.0])),
            series_labels=("train-vs-test", "train-vs-val", "chance"),
            line_styles=("-", "-", "--"),
        )

    def test_renders_both(self, panel):
        for b in BACKENDS:
            assert _render(panel, b) is not None

    def test_plotly_each_series_keeps_own_x(self, panel):
        fig = _render(panel, "plotly")
        line_traces = [t for t in fig.data if t.type in ("scatter", "scattergl")]
        assert len(line_traces) == 3
        # The second series must keep the test fpr grid, not the train one.
        np.testing.assert_array_equal(np.asarray(line_traces[1].x), np.array([0.0, 0.4, 0.7, 1.0]))

    def test_matplotlib_three_lines(self, panel):
        fig = _render(panel, "matplotlib")
        assert len(fig.axes[0].lines) == 3

    def test_shared_x_still_works(self):
        # Back-compat: a single shared x array across two series.
        panel = LinePanelSpec(x=np.arange(10), y=(np.arange(10.0), np.arange(10.0) * 2), series_labels=("a", "b"))
        fig = _render(panel, "matplotlib")
        assert len(fig.axes[0].lines) == 2
