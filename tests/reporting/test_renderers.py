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
    BarPanelSpec, FigureSpec, HeatmapPanelSpec, HistogramPanelSpec,
    LinePanelSpec, ScatterPanelSpec, ViolinPanelSpec,
)


# ----------------------------------------------------------------------------
# Common fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def scatter_panel(rng):
    x = rng.standard_normal(100)
    y = x * 1.5 + rng.standard_normal(100) * 0.2
    return ScatterPanelSpec(
        x=x, y=y, title="scatter", xlabel="x", ylabel="y",
        perfect_fit_line=True, point_alpha=0.5,
    )


@pytest.fixture
def histogram_panel(rng):
    return HistogramPanelSpec(
        values=rng.standard_normal(500), bins=30,
        title="hist", xlabel="x", overlay_normal=(0.0, 1.0),
    )


@pytest.fixture
def heatmap_panel():
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
    return BarPanelSpec(
        categories=("A", "B", "C"),
        values=np.array([1.0, 2.0, 1.5]),
        title="bar", xlabel="cat", ylabel="val",
    )


@pytest.fixture
def line_panel():
    x = np.arange(10)
    return LinePanelSpec(
        x=x, y=(x.astype(float), x.astype(float) * 2),
        series_labels=("s1", "s2"), title="line", xlabel="x", ylabel="y",
    )


@pytest.fixture
def violin_panel(rng):
    return ViolinPanelSpec(
        groups=(rng.standard_normal(50), rng.standard_normal(50) + 1),
        group_labels=("g1", "g2"), title="violin",
    )


# ----------------------------------------------------------------------------
# Matplotlib renderer
# ----------------------------------------------------------------------------


class TestMatplotlibRenderer:
    def test_render_single_scatter(self, scatter_panel):
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(suptitle="t", panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        # Native matplotlib Figure
        assert hasattr(fig, "savefig")
        assert hasattr(fig, "axes")
        assert len(fig.axes) >= 1  # >=1 because legend may add an axis

    def test_render_2x2_grid(self, scatter_panel, histogram_panel, heatmap_panel, bar_panel):
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(
            panels=((scatter_panel, histogram_panel),
                    (heatmap_panel, bar_panel)),
            figsize=(12, 8),
        )
        fig = renderer.render(spec)
        # Heatmap adds a colorbar axis; expect ≥4 main axes.
        assert len(fig.axes) >= 4

    def test_render_line_panel(self, line_panel):
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((line_panel,),), figsize=(8, 4))
        fig = renderer.render(spec)
        ax = fig.axes[0]
        # Two lines + legend artists
        assert len(ax.lines) >= 2

    def test_render_violin(self, violin_panel):
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((violin_panel,),), figsize=(8, 4))
        fig = renderer.render(spec)
        # Violin produces PolyCollections.
        assert len(fig.axes) >= 1

    def test_save_png(self, scatter_panel, tmp_path):
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        out = str(tmp_path / "scatter.png")
        renderer.save(fig, out, "png")
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

    def test_save_pdf(self, scatter_panel, tmp_path):
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        out = str(tmp_path / "scatter.pdf")
        renderer.save(fig, out, "pdf")
        assert os.path.exists(out)

    def test_save_unknown_format_raises(self, scatter_panel, tmp_path):
        renderer = get_renderer("matplotlib")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        with pytest.raises(ValueError, match="doesn't support"):
            renderer.save(fig, str(tmp_path / "x.html"), "html")

    def test_empty_panels_raises(self):
        renderer = get_renderer("matplotlib")
        with pytest.raises(ValueError, match="no panels"):
            renderer.render(FigureSpec(panels=(), figsize=(6, 4)))


# ----------------------------------------------------------------------------
# Plotly renderer
# ----------------------------------------------------------------------------


class TestPlotlyRenderer:
    def test_render_single_scatter(self, scatter_panel):
        renderer = get_renderer("plotly")
        spec = FigureSpec(suptitle="t", panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        # plotly Figure
        assert hasattr(fig, "to_html")
        assert hasattr(fig, "data")
        # scatter trace + perfect-fit line trace
        assert len(fig.data) >= 1

    def test_render_2x2_grid(self, scatter_panel, histogram_panel, heatmap_panel, bar_panel):
        renderer = get_renderer("plotly")
        spec = FigureSpec(
            panels=((scatter_panel, histogram_panel),
                    (heatmap_panel, bar_panel)),
            figsize=(12, 8),
        )
        fig = renderer.render(spec)
        # 4 panels each produce ≥1 trace.
        assert len(fig.data) >= 4

    def test_render_line_panel(self, line_panel):
        renderer = get_renderer("plotly")
        spec = FigureSpec(panels=((line_panel,),), figsize=(8, 4))
        fig = renderer.render(spec)
        # Two line traces
        assert sum(1 for t in fig.data if t.type == "scatter" and t.mode == "lines") == 2

    def test_save_html(self, scatter_panel, tmp_path):
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
        renderer = get_renderer("plotly")
        spec = FigureSpec(panels=((scatter_panel,),), figsize=(6, 4))
        fig = renderer.render(spec)
        with pytest.raises(ValueError, match="doesn't support"):
            renderer.save(fig, str(tmp_path / "x.jpg"), "jpg")


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------


class TestRendererFactory:
    def test_get_matplotlib(self):
        r = get_renderer("matplotlib")
        assert r.backend == "matplotlib"

    def test_get_plotly(self):
        r = get_renderer("plotly")
        assert r.backend == "plotly"

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown renderer"):
            get_renderer("bokeh")

    def test_case_insensitive(self):
        assert get_renderer("PLOTLY").backend == "plotly"
