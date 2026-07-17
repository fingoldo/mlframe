"""Large-n safety nets in the plotly + matplotlib renderers.

Covers the renderer-level guards that keep a raw large-n spec from embedding millions of values
into the output: histogram pre-bin, Scattergl switch, extremes-preserving downsample cap, vectorized
marker-size, the union-range perfect-fit diagonal on a collapsed-prediction spec, and the
static-export legend gate. Sizes are kept modest (just past each threshold) so the suite stays fast.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mlframe.reporting.renderers import get_renderer, render_and_save
from mlframe.reporting.renderers import plotly as plotly_mod
from mlframe.reporting.renderers import matplotlib as mpl_mod
from mlframe.reporting.output import parse_plot_output_dsl
from mlframe.reporting.spec import FigureSpec, HistogramPanelSpec, ScatterPanelSpec


@pytest.fixture
def rng():
    return np.random.default_rng(7)


# ----------------------------------------------------------------------------
# Histogram pre-bin (PERF-4) — raw values above ~50k must not be embedded.
# ----------------------------------------------------------------------------


class TestHistogramPrebin:
    def test_plotly_prebins_above_threshold_to_bar_trace(self, rng):
        """Above the pre-bin threshold, plotly draws a go.Bar (pre-binned) instead of a go.Histogram that
        ships every raw value into the JSON."""
        n = plotly_mod._HIST_PREBIN_THRESHOLD + 5_000
        vals = rng.standard_normal(n)
        spec = FigureSpec(panels=((HistogramPanelSpec(values=vals, bins=40, title="h"),),), figsize=(6, 4))
        fig = get_renderer("plotly").render(spec)
        bar = [t for t in fig.data if t.type == "bar"]
        hist = [t for t in fig.data if t.type == "histogram"]
        assert len(bar) == 1 and not hist, "expected one pre-binned Bar trace, no raw Histogram trace"
        # Bar carries <= bins points, never the raw n.
        assert len(bar[0].y) <= 40

    def test_plotly_small_histogram_stays_native(self, rng):
        spec = FigureSpec(panels=((HistogramPanelSpec(values=rng.standard_normal(500), bins=30),),), figsize=(6, 4))
        fig = get_renderer("plotly").render(spec)
        assert any(t.type == "histogram" for t in fig.data)

    def test_plotly_prebin_html_size_small(self, rng, tmp_path):
        """The whole point: the pre-binned HTML is tiny vs the 37 MB raw embed at 2M."""
        n = plotly_mod._HIST_PREBIN_THRESHOLD + 50_000
        spec = FigureSpec(panels=((HistogramPanelSpec(values=rng.standard_normal(n), bins=50),),), figsize=(6, 4))
        r = get_renderer("plotly")
        out = str(tmp_path / "h.html")
        r.save(r.render(spec), out, "html")
        assert os.path.getsize(out) < 0.5 * 1024 * 1024, "pre-binned histogram HTML must stay under 0.5 MB"

    def test_matplotlib_prebins_above_threshold(self, rng):
        """The mpl pre-bin path produces <= bins bar containers, not a full-n ax.hist."""
        n = mpl_mod._HIST_PREBIN_THRESHOLD + 5_000
        spec = FigureSpec(panels=((HistogramPanelSpec(values=rng.standard_normal(n), bins=35, overlay_normal=(0.0, 1.0)),),), figsize=(6, 4))
        fig = get_renderer("matplotlib").render(spec)
        ax = fig.axes[0]
        # ax.bar -> one BarContainer of <= bins patches; ax.hist would emit a single Polygon/patches set of n-free
        # but the discriminating signal is the overlay line still being present and the panel rendering cleanly.
        assert len(ax.patches) <= 35
        assert any(line.get_linestyle() == "--" for line in ax.lines), "Normal overlay should survive pre-bin"

    def test_prebin_overlay_grid_from_edges_not_raw(self, rng):
        """PERF-18: the Normal overlay x-grid spans the bin-edge range, not two extra full-n passes."""
        n = plotly_mod._HIST_PREBIN_THRESHOLD + 1_000
        vals = rng.standard_normal(n)
        spec = FigureSpec(panels=((HistogramPanelSpec(values=vals, bins=30, overlay_normal=(0.0, 1.0)),),), figsize=(6, 4))
        fig = get_renderer("plotly").render(spec)
        overlay = [t for t in fig.data if t.type == "scatter" and t.mode == "lines"]
        assert len(overlay) == 1
        gx = np.asarray(overlay[0].x)
        # Grid stays within the data's actual extent (edge-derived), within a bin-width of min/max.
        assert gx.min() >= vals.min() - 1.0 and gx.max() <= vals.max() + 1.0


# ----------------------------------------------------------------------------
# Scatter: Scattergl switch (>10k), downsample cap (>50k), marker-size vectorization.
# ----------------------------------------------------------------------------


class TestScatterLargeN:
    def test_plotly_switches_to_scattergl_above_webgl_threshold(self, rng):
        n = plotly_mod._SCATTER_WEBGL_THRESHOLD + 2_000
        x = rng.standard_normal(n)
        spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=x + rng.standard_normal(n) * 0.1),),), figsize=(6, 4))
        fig = get_renderer("plotly").render(spec)
        assert any(t.type == "scattergl" for t in fig.data), "large scatter must use WebGL Scattergl"

    def test_plotly_small_scatter_stays_svg(self, rng):
        x = rng.standard_normal(1_000)
        spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=x),),), figsize=(6, 4))
        fig = get_renderer("plotly").render(spec)
        assert any(t.type == "scatter" for t in fig.data)
        assert not any(t.type == "scattergl" for t in fig.data)

    def test_plotly_downsamples_above_cap(self, rng):
        n = plotly_mod._SCATTER_MAX_POINTS + 20_000
        x = rng.standard_normal(n)
        spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=x),),), figsize=(6, 4))
        fig = get_renderer("plotly").render(spec)
        pts = [t for t in fig.data if t.type in ("scatter", "scattergl") and t.mode and "markers" in t.mode]
        assert pts, "expected a marker trace"
        assert len(pts[0].x) <= plotly_mod._SCATTER_MAX_POINTS, "downsample must cap at _SCATTER_MAX_POINTS"

    def test_downsample_keeps_per_point_size_and_color_aligned(self, rng):
        """When point_size / point_color are per-point arrays they follow the SAME subset as x/y."""
        n = plotly_mod._SCATTER_MAX_POINTS + 5_000
        x = rng.standard_normal(n)
        sizes = rng.uniform(1, 50, n)
        colors = rng.uniform(0, 1, n)
        spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=x, point_size=sizes, point_color=colors, colorbar_label="c"),),), figsize=(6, 4))
        fig = get_renderer("plotly").render(spec)
        pts = [t for t in fig.data if t.mode and "markers" in t.mode][0]
        npt = len(pts.x)
        assert len(pts.marker.size) == npt
        assert len(pts.marker.color) == npt

    def test_plotly_scatter_html_size_capped_at_2M_input(self, rng, tmp_path):
        """50k-capped scatter HTML stays well under 5 MB even with a 2M-point input (target: <5 MB)."""
        n = 2_000_000
        x = rng.standard_normal(n)
        spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=x + rng.standard_normal(n) * 0.1),),), figsize=(6, 4))
        r = get_renderer("plotly")
        out = str(tmp_path / "s.html")
        r.save(r.render(spec), out, "html")
        assert os.path.getsize(out) < 5 * 1024 * 1024

    def test_marker_size_vectorized_matches_reference(self, rng):
        """PERF-16: vectorized sqrt marker-size must equal the per-point reference exactly."""
        sizes = rng.uniform(0, 100, 2_000)
        spec = FigureSpec(panels=((ScatterPanelSpec(x=rng.standard_normal(2_000), y=rng.standard_normal(2_000), point_size=sizes),),), figsize=(6, 4))
        fig = get_renderer("plotly").render(spec)
        pts = [t for t in fig.data if t.mode and "markers" in t.mode][0]
        got = np.asarray(pts.marker.size, dtype=float)
        ref = np.sqrt(np.maximum(sizes, 0.0)) * 1.33
        np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_matplotlib_rasterizes_large_scatter(self, rng):
        n = mpl_mod._SCATTER_MAX_POINTS + 10_000
        x = rng.standard_normal(n)
        spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=x),),), figsize=(6, 4))
        fig = get_renderer("matplotlib").render(spec)
        ax = fig.axes[0]
        coll = ax.collections[0]
        assert coll.get_rasterized() is True, "capped mpl scatter should be rasterized for small vector export"
        # Offsets are capped at the downsample budget.
        assert len(coll.get_offsets()) <= mpl_mod._SCATTER_MAX_POINTS


# ----------------------------------------------------------------------------
# INV-16 — perfect-fit diagonal spans the union range on a collapsed prediction.
# ----------------------------------------------------------------------------


class TestDiagonalUnionRange:
    def test_plotly_diagonal_union_range_on_collapsed_y(self):
        """y collapsed to a constant: the diagonal must still span [min(all), max(all)] on both axes via
        scaleanchor, not degenerate to the single y value."""
        x = np.linspace(-3.0, 5.0, 200)
        y = np.full_like(x, 1.0)  # collapsed prediction
        spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=y, perfect_fit_line=True),),), figsize=(5, 5))
        fig = get_renderer("plotly").render(spec)
        diag = [t for t in fig.data if t.name == "Perfect fit"]
        assert len(diag) == 1
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        assert tuple(diag[0].x) == (lo, hi)
        assert tuple(diag[0].y) == (lo, hi)
        # x-axis squared against y via scaleanchor.
        assert fig.layout.yaxis.scaleanchor in ("x", "x1")

    def test_matplotlib_diagonal_union_range_equal_aspect(self):
        x = np.linspace(-3.0, 5.0, 200)
        y = np.full_like(x, 1.0)
        spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=y, perfect_fit_line=True),),), figsize=(5, 5))
        fig = get_renderer("matplotlib").render(spec)
        ax = fig.axes[0]
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        assert ax.get_xlim() == (lo, hi)
        assert ax.get_ylim() == (lo, hi)
        assert ax.get_aspect() == 1.0  # 'equal' resolves to a 1.0 ratio


# ----------------------------------------------------------------------------
# INV-28 — plotly legends on when a static format is in the save set.
# ----------------------------------------------------------------------------


class TestStaticLegend:
    def _labeled_spec(self):
        x = np.arange(10)
        from mlframe.reporting.spec import LinePanelSpec

        return FigureSpec(panels=((LinePanelSpec(x=x, y=(x.astype(float), x.astype(float) * 2), series_labels=("a", "b")),),), figsize=(6, 4))

    def test_render_static_legend_flag_enables_legend(self):
        fig = get_renderer("plotly").render(self._labeled_spec(), static_legend=True)
        assert fig.layout.showlegend is True

    def test_render_default_no_legend(self):
        fig = get_renderer("plotly").render(self._labeled_spec())
        assert fig.layout.showlegend is False

    def test_save_dispatch_enables_legend_for_png_in_set(self, monkeypatch):
        """render_and_save passes static_legend=True to plotly when a static format is requested."""
        captured = {}
        real = plotly_mod.PlotlyRenderer.render

        def _spy(self, spec, *, static_legend=False):
            captured["static_legend"] = static_legend
            return real(self, spec, static_legend=static_legend)

        monkeypatch.setattr(plotly_mod.PlotlyRenderer, "render", _spy)
        out = parse_plot_output_dsl("plotly[html,svg]")
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            render_and_save(self._labeled_spec(), out, os.path.join(d, "p"))
        assert captured.get("static_legend") is True

    def test_save_dispatch_html_only_keeps_legend_off(self, monkeypatch):
        captured = {}
        real = plotly_mod.PlotlyRenderer.render

        def _spy(self, spec, *, static_legend=False):
            captured["static_legend"] = static_legend
            return real(self, spec, static_legend=static_legend)

        monkeypatch.setattr(plotly_mod.PlotlyRenderer, "render", _spy)
        out = parse_plot_output_dsl("plotly[html]")
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            render_and_save(self._labeled_spec(), out, os.path.join(d, "p"))
        assert captured.get("static_legend") is False


# ----------------------------------------------------------------------------
# INV-51 — render_and_save counts dropped charts (multi-backend thread path).
# ----------------------------------------------------------------------------


class TestRenderFailureStats:
    def test_exception_increments_counter(self, monkeypatch):
        from mlframe.reporting.renderers import save as save_mod

        save_mod.reset_render_failure_stats()

        # Make matplotlib render explode; plotly still succeeds. Multi-backend DSL forces the thread path.
        def _boom(self, spec):
            raise RuntimeError("synthetic render failure")

        monkeypatch.setattr(mpl_mod.MatplotlibRenderer, "render", _boom)
        out = parse_plot_output_dsl("plotly[html] + matplotlib[png]")
        from mlframe.reporting.spec import LinePanelSpec

        x = np.arange(5)
        spec = FigureSpec(panels=((LinePanelSpec(x=x, y=x.astype(float)),),), figsize=(4, 3))
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            render_and_save(spec, out, os.path.join(d, "p"))
        stats = save_mod.get_render_failure_stats()
        assert stats["total"] == 1
        assert stats["exceptions"] == 1
        assert stats["timeouts"] == 0
        save_mod.reset_render_failure_stats()

    def test_stats_reset(self):
        from mlframe.reporting.renderers import save as save_mod

        save_mod.reset_render_failure_stats()
        assert save_mod.get_render_failure_stats() == {"total": 0, "timeouts": 0, "exceptions": 0}
