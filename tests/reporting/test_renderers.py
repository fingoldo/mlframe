"""Tests for matplotlib + plotly renderers.

Each renderer must build a native figure handle from a FigureSpec and
preserve the structural content (panel count, axis labels, titles,
data values). Pixel equivalence between backends is NOT asserted (font
metrics differ); structural correctness is.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.spec import (
    BarPanelSpec,
    FigureSpec,
    HeatmapPanelSpec,
    HistogramPanelSpec,
    LinePanelSpec,
    ScatterPanelSpec,
    ViolinPanelSpec,
)

# ----------------------------------------------------------------------------
# Common fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def rng():
    """Rng."""
    return np.random.default_rng(42)


@pytest.fixture
def scatter_panel(rng):
    """Scatter panel."""
    x = rng.standard_normal(100)
    y = x * 1.5 + rng.standard_normal(100) * 0.2
    return ScatterPanelSpec(
        x=x,
        y=y,
        title="scatter",
        xlabel="x",
        ylabel="y",
        perfect_fit_line=True,
        point_alpha=0.5,
    )


@pytest.fixture
def histogram_panel(rng):
    """Histogram panel."""
    return HistogramPanelSpec(
        values=rng.standard_normal(500),
        bins=30,
        title="hist",
        xlabel="x",
        overlay_normal=(0.0, 1.0),
    )


@pytest.fixture
def heatmap_panel():
    """Heatmap panel."""
    matrix = np.array([[0.9, 0.1], [0.05, 0.95]])
    return HeatmapPanelSpec(
        matrix=matrix,
        row_labels=("class A", "class B"),
        col_labels=("pred A", "pred B"),
        title="heatmap",
        cell_text=matrix,
        text_format=".2f",
        colorbar_label="prob",
    )


@pytest.fixture
def bar_panel():
    """Bar panel."""
    return BarPanelSpec(
        categories=("A", "B", "C"),
        values=np.array([1.0, 2.0, 1.5]),
        title="bar",
        xlabel="cat",
        ylabel="val",
    )


@pytest.fixture
def line_panel():
    """Line panel."""
    x = np.arange(10)
    return LinePanelSpec(
        x=x,
        y=(x.astype(float), x.astype(float) * 2),
        series_labels=("s1", "s2"),
        title="line",
        xlabel="x",
        ylabel="y",
    )


@pytest.fixture
def violin_panel(rng):
    """Violin panel."""
    return ViolinPanelSpec(
        groups=(rng.standard_normal(50), rng.standard_normal(50) + 1),
        group_labels=("g1", "g2"),
        title="violin",
    )


# ----------------------------------------------------------------------------
# Matplotlib renderer
# ----------------------------------------------------------------------------


class TestMatplotlibRenderer:
    """Groups tests for: TestMatplotlibRenderer."""
    def test_render_single_scatter(self, scatter_panel):
        """Render single scatter."""
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(suptitle="t", panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        # Native matplotlib Figure
        assert hasattr(fig, "savefig")
        assert hasattr(fig, "axes")
        assert len(fig.axes) >= 1  # >=1 because legend may add an axis

    def test_render_2x2_grid(self, scatter_panel, histogram_panel, heatmap_panel, bar_panel):
        """Render 2x2 grid."""
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(
            panels=((scatter_panel, histogram_panel), (heatmap_panel, bar_panel)),
            figsize=(12, 8),
        )
        fig = renderer.render(spec)
        # Heatmap adds a colorbar axis; expect ≥4 main axes.
        assert len(fig.axes) >= 4

    def test_render_line_panel(self, line_panel):
        """Render line panel."""
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((line_panel,),), figsize=(8, 4))
        fig = renderer.render(spec)
        ax = fig.axes[0]
        # Two lines + legend artists
        assert len(ax.lines) >= 2

    def test_render_violin(self, violin_panel):
        """Render violin."""
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((violin_panel,),), figsize=(8, 4))
        fig = renderer.render(spec)
        # Violin produces PolyCollections.
        assert len(fig.axes) >= 1

    def test_save_png(self, scatter_panel, tmp_path):
        """Save png."""
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        out = str(tmp_path / "scatter.png")
        renderer.save(fig, out, "png")
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_save_pdf(self, scatter_panel, tmp_path):
        """Save pdf."""
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        out = str(tmp_path / "scatter.pdf")
        renderer.save(fig, out, "pdf")
        assert os.path.exists(out)

    def test_save_unknown_format_raises(self, scatter_panel, tmp_path):
        """Save unknown format raises."""
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        with pytest.raises(ValueError, match="doesn't support"):
            renderer.save(fig, str(tmp_path / "x.html"), "html")

    def test_empty_panels_raises(self):
        """Empty panels raises."""
        renderer = get_renderer("matplotlib")
        with pytest.raises(ValueError, match="no panels"):
            renderer.render(FigureSpec(panels=(), figsize=(6, 4)))


# ----------------------------------------------------------------------------
# Plotly renderer
# ----------------------------------------------------------------------------


class TestPlotlyRenderer:
    """Groups tests for: TestPlotlyRenderer."""
    def test_render_single_scatter(self, scatter_panel):
        """Render single scatter."""
        renderer = get_renderer("plotly")
        spec = FigureSpec(suptitle="t", panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        # plotly Figure
        assert hasattr(fig, "to_html")
        assert hasattr(fig, "data")
        # scatter trace + perfect-fit line trace
        assert len(fig.data) >= 1

    def test_render_2x2_grid(self, scatter_panel, histogram_panel, heatmap_panel, bar_panel):
        """Render 2x2 grid."""
        renderer = get_renderer("plotly")
        spec = FigureSpec(
            panels=((scatter_panel, histogram_panel), (heatmap_panel, bar_panel)),
            figsize=(12, 8),
        )
        fig = renderer.render(spec)
        # 4 panels each produce ≥1 trace.
        assert len(fig.data) >= 4

    def test_render_line_panel(self, line_panel):
        """Render line panel."""
        renderer = get_renderer("plotly")
        spec = FigureSpec(panels=((line_panel,),), figsize=(8, 4))
        fig = renderer.render(spec)
        # Two line traces
        assert sum(1 for t in fig.data if t.type == "scatter" and t.mode == "lines") == 2

    def test_save_html(self, scatter_panel, tmp_path):
        """Save html."""
        renderer = get_renderer("plotly")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        out = str(tmp_path / "scatter.html")
        renderer.save(fig, out, "html")
        assert os.path.exists(out)
        # Sanity: contains plotly.js include.
        with open(out, encoding="utf-8") as f:
            content = f.read()
        assert "plotly" in content.lower()

    def test_save_json(self, scatter_panel, tmp_path):
        """Save json."""
        renderer = get_renderer("plotly")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        out = str(tmp_path / "scatter.json")
        renderer.save(fig, out, "json")
        assert os.path.exists(out)
        # Valid JSON with data + layout. mlframe rule: orjson over json,
        # so use orjson.loads on the file bytes. orjson has no streaming
        # load() so we read the whole file (~few KB for this test) first.
        import orjson

        with open(out, "rb") as f:
            obj = orjson.loads(f.read())
        assert "data" in obj
        assert "layout" in obj

    def test_save_unknown_format_raises(self, scatter_panel, tmp_path):
        """Save unknown format raises."""
        renderer = get_renderer("plotly")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        with pytest.raises(ValueError, match="doesn't support"):
            renderer.save(fig, str(tmp_path / "x.jpg"), "jpg")


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


class TestRendererFactory:
    """Groups tests for: TestRendererFactory."""
    def test_get_matplotlib(self):
        """Get matplotlib."""
        r = get_renderer("matplotlib")
        assert r.backend == "matplotlib"

    def test_get_plotly(self):
        """Get plotly."""
        r = get_renderer("plotly")
        assert r.backend == "plotly"

    def test_unknown_backend_raises(self):
        """Unknown backend raises."""
        with pytest.raises(ValueError, match="unknown renderer"):
            get_renderer("bokeh")

    def test_case_insensitive(self):
        """Case insensitive."""
        assert get_renderer("PLOTLY").backend == "plotly"


_LONG_METRIC_TITLE = (
    "VAL (DUMMY) DummyBaseline:oracle_prior cl_act_total_hired_above_1 MTTR=0.43 "
    "[68F/242.4K rows] ICE=0.15, BR=24.4%(RL0.0%+U24.4%-RS-0.0%), ECE=0.0%, "
    "CMAEW=0.0%, LL=0.68 ROC AUC=0.50, PR AUC=0.42, KS=0.00, MCC=0.00, BSS=-0.00"
)


def _one_panel_spec(figsize):
    """Single-panel FigureSpec carrying the long diagnostic-metric title."""
    return FigureSpec(
        panels=[[LinePanelSpec(x=np.arange(10), y=np.arange(10), title=_LONG_METRIC_TITLE)]],
        figsize=figsize,
    )


def test_plotly_panel_title_fills_a_wide_panel_instead_of_a_narrow_column():
    """A wide panel must pack its title to the panel's real width.

    The wrap budget was a flat 46 chars/line calibrated for a ~6-inch panel but applied at any width, so
    a wide single-panel figure folded a long metric title into a tall ragged column using a fraction of
    the space. Asserting the LINE COUNT drops as the panel widens: a width-agnostic implementation
    produces the same number of lines for both figure sizes.
    """
    from mlframe.reporting.renderers._shared_helpers import panel_title_wrap_chars

    narrow = panel_title_wrap_chars((6, 4), 1)
    wide = panel_title_wrap_chars((15, 6), 1)
    assert narrow == 46, narrow  # unchanged for the width it was calibrated against
    assert wide > 2 * narrow, (narrow, wide)

    # Per-column split: two panels on a 15-inch figure each get roughly half the single-panel budget.
    assert panel_title_wrap_chars((15, 6), 2) < wide
    # Degenerate/missing figsize must not raise.
    assert panel_title_wrap_chars(None, 1) == 46


def test_plotly_panel_title_preserves_explicit_line_breaks():
    """Explicit ``\n`` breaks in a caller-supplied title are deliberate and must survive wrapping."""
    pytest.importorskip("plotly")
    from mlframe.reporting.renderers.plotly import _wrap_text

    out = _wrap_text("alpha\nbeta\ngamma", 200)
    assert out == "alpha<br>beta<br>gamma", out


def test_matplotlib_panel_title_preserves_explicit_line_breaks():
    """``textwrap.wrap`` treats a newline as ordinary whitespace, so the pre-fix matplotlib title path
    silently collapsed and re-flowed any explicit break the caller wrote."""
    fig = get_renderer("matplotlib").render(FigureSpec(panels=[[LinePanelSpec(x=np.arange(5), y=np.arange(5), title="alpha\nbeta\ngamma")]], figsize=(12, 4)))
    assert fig.axes[0].get_title() == "alpha\nbeta\ngamma"


def test_matplotlib_wide_panel_title_uses_fewer_lines_than_a_narrow_one():
    """Same width-scaling contract as the plotly twin, asserted through the real rendered title."""
    narrow_title = get_renderer("matplotlib").render(_one_panel_spec((6, 4))).axes[0].get_title()
    wide_title = get_renderer("matplotlib").render(_one_panel_spec((18, 6))).axes[0].get_title()
    assert wide_title.count("\n") < narrow_title.count("\n"), (narrow_title, wide_title)


def test_plotly_figure_size_matches_the_matplotlib_twin():
    """The same FigureSpec must yield the same figure size in both backends.

    ``figsize`` is in matplotlib inches and matplotlib renders at 100 dpi, so plotly's former 80 px/inch
    made every plotly figure 20% smaller than its matplotlib counterpart built from the identical spec --
    the "the plotly version looks cramped" difference. Plot-area width is compared directly; plotly's
    height additionally carries the reserved title band, so the check is >= the matplotlib height.
    """
    pytest.importorskip("plotly")
    spec = FigureSpec(panels=((LinePanelSpec(x=np.arange(5), y=np.arange(5), title="t"),),), figsize=(12, 6))

    pf = get_renderer("plotly").render(spec)
    mf = get_renderer("matplotlib").render(spec)
    mpl_w, mpl_h = (int(v * mf.get_dpi()) for v in mf.get_size_inches())

    assert pf.layout.width == mpl_w, (pf.layout.width, mpl_w)
    assert pf.layout.height >= mpl_h, (pf.layout.height, mpl_h)


def test_plotly_top_margin_reserves_room_for_row1_panel_titles():
    """A multi-line panel title must not land on the suptitle.

    plotly stamps each subplot title as an annotation just ABOVE its subplot domain -- i.e. inside the top
    margin. Sizing that margin from the suptitle alone let the two collide, which is what produced the
    overlapping title text on every wide multi-panel diagnostic figure.
    """
    pytest.importorskip("plotly")
    long_panel_title = "Adversarial validation: train-vs-test AUC=1.000 (shift => CV may NOT transfer)"
    suptitle = "line one of the run identity\nline two\nline three"

    with_titles = get_renderer("plotly").render(
        FigureSpec(suptitle=suptitle, panels=((LinePanelSpec(x=np.arange(5), y=np.arange(5), title=long_panel_title),),), figsize=(6, 4))
    )
    without_titles = get_renderer("plotly").render(
        FigureSpec(suptitle=suptitle, panels=((LinePanelSpec(x=np.arange(5), y=np.arange(5), title=""),),), figsize=(6, 4))
    )
    assert with_titles.layout.margin.t > without_titles.layout.margin.t, (with_titles.layout.margin.t, without_titles.layout.margin.t)


def test_plotly_does_not_truncate_ordinary_feature_names():
    """Long-but-ordinary feature names must render in full, as the matplotlib renderer already does.

    The 24-char cap turned "job_posted_at_day_of_year_cos" into "job_posted_at_day_of_ye...", making the
    plotly twin strictly less informative than its matplotlib counterpart at the same figure size.
    """
    pytest.importorskip("plotly")
    from mlframe.reporting.renderers.plotly import _truncate_label

    assert _truncate_label("job_posted_at_day_of_year_cos") == "job_posted_at_day_of_year_cos"
    # The cap survives as a safety valve for a pathological generated name.
    assert _truncate_label("x" * 200).endswith("...")


def _time_axis_spec():
    """Single line panel whose x carries epoch nanoseconds and declares ``x_is_time``."""
    import pandas as pd

    x = pd.date_range("2021-07-01", "2026-07-31", periods=120).values.astype("datetime64[ns]").astype(np.int64).astype(float)
    return FigureSpec(
        panels=((LinePanelSpec(x=x, y=np.linspace(0.95, 0.70, 120), series_labels=("roc_auc",), title="roc_auc over time", xlabel="time", ylabel="roc_auc", x_is_time=True),),),
        figsize=(10, 4),
    )


def test_time_axis_renders_dates_not_epoch_nanoseconds_plotly():
    """``x_is_time`` must format the axis as dates, not merely rotate the labels.

    Spec builders pass epoch NANOSECONDS as a numeric x (so vspans/regime shading share one coordinate
    space) and set ``x_is_time`` to mean "these are timestamps". That flag only rotated the labels, so a
    five-year metric-over-time chart rendered its axis as ``1.62e18 ... 1.78e18`` -- no usable information.
    """
    pytest.importorskip("plotly")
    fig = get_renderer("plotly").render(_time_axis_spec())

    ticktext = list(fig.layout.xaxis.ticktext or [])
    assert ticktext, "time axis left unformatted"
    assert all("e+" not in t and "e18" not in t for t in ticktext), ticktext
    assert ticktext[0].startswith("2021"), ticktext


def test_time_axis_renders_dates_not_epoch_nanoseconds_matplotlib():
    """matplotlib twin of the time-axis contract: ``autofmt_xdate`` alone cannot convert a float axis."""
    fig = get_renderer("matplotlib").render(_time_axis_spec())
    fig.canvas.draw()

    labels = [t.get_text() for t in fig.axes[0].get_xticklabels() if t.get_text()]
    assert labels, "time axis left unformatted"
    assert all("e+" not in lab and "1e18" not in lab for lab in labels), labels
    assert any(lab.startswith("2021") for lab in labels), labels


def test_epoch_ns_ticks_is_safe_on_degenerate_input():
    """Empty / all-NaN input must leave the axis alone rather than raising inside a renderer."""
    from mlframe.reporting.renderers._shared_helpers import epoch_ns_ticks

    assert epoch_ns_ticks([]) == (None, None)
    assert epoch_ns_ticks([np.nan, np.nan]) == (None, None)
    # A single-instant series still yields usable ticks rather than a zero-width range.
    tickvals, ticktext = epoch_ns_ticks([1.7e18])
    assert tickvals is not None and len(ticktext) >= 2


def test_plotly_single_labelled_panel_keeps_its_legend():
    """A one-panel figure whose series are named must show a legend even in interactive HTML.

    The legend was suppressed for every non-static export to avoid multi-panel legend soup (all panels'
    series pooled into one list). With a single panel there is no soup, and without the legend a chart like
    the decision curve renders three unlabelled lines a reader cannot tell apart -- the model, "treat all"
    and "treat none" look identical at a glance.
    """
    pytest.importorskip("plotly")
    x = np.linspace(0, 0.6, 40)
    panel = LinePanelSpec(
        x=x, y=(0.4 - x, 0.4 - 2 * x, np.zeros_like(x)),
        series_labels=("model", "treat all", "treat none"), title="Decision-curve analysis",
    )
    fig = get_renderer("plotly").render(FigureSpec(panels=((panel,),), figsize=(8, 4)))
    assert fig.layout.showlegend is True


def test_plotly_multi_panel_still_suppresses_the_interactive_legend():
    """The soup-avoidance rule is unchanged where it actually applies."""
    pytest.importorskip("plotly")
    x = np.linspace(0, 0.6, 40)
    panel = LinePanelSpec(x=x, y=(0.4 - x,), series_labels=("model",), title="p")
    fig = get_renderer("plotly").render(FigureSpec(panels=((panel, panel),), figsize=(12, 4)))
    assert not fig.layout.showlegend


def test_plotly_single_unlabelled_panel_gets_no_legend():
    """Without ``series_labels`` a legend would only show plotly's auto names (trace 0, trace 1, ...)."""
    pytest.importorskip("plotly")
    x = np.linspace(0, 0.6, 40)
    panel = LinePanelSpec(x=x, y=(0.4 - x,), title="p")
    fig = get_renderer("plotly").render(FigureSpec(panels=((panel,),), figsize=(8, 4)))
    assert not fig.layout.showlegend
