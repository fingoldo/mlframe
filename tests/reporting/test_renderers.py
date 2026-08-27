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
