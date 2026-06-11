"""Colourblind-safe heatmap colormap defaults + LINE_PALETTE byte-stability.

Heatmaps must default to a perceptually-uniform / CB-safe sequential colormap (viridis) on BOTH backends when
the spec leaves the colormap at the generic placeholder; LINE_PALETTE must stay byte-stable (snapshot-pinned).
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.colors import (
    DIVERGING_CMAP, HEATMAP_CMAP, HEATMAP_GENERIC, LINE_PALETTE,
    resolve_heatmap_cmap,
)
from mlframe.reporting.renderers.plotly import _mpl_to_plotly_cmap
from mlframe.reporting.spec import HeatmapPanelSpec


_TAB10 = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
_TAB20_EXT = (
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
)


class TestCBDefaults:
    def test_heatmap_cmap_is_cb_safe_sequential(self):
        assert HEATMAP_CMAP == "viridis"

    def test_diverging_cmap_is_signed_safe(self):
        # Diverging default must be a recognised signed scheme (RdBu_r) so renderers map it cleanly.
        assert DIVERGING_CMAP in ("RdBu_r", "RdBu", "cividis")

    def test_resolver_maps_generic_placeholder_to_cb_safe(self):
        assert resolve_heatmap_cmap(HEATMAP_GENERIC) == HEATMAP_CMAP
        assert resolve_heatmap_cmap(None) == HEATMAP_CMAP

    def test_resolver_honours_explicit_override(self):
        assert resolve_heatmap_cmap("RdBu_r") == "RdBu_r"
        assert resolve_heatmap_cmap("plasma") == "plasma"

    def test_default_heatmap_spec_resolves_cb_safe(self):
        # An un-overridden HeatmapPanelSpec keeps HEATMAP_GENERIC as its field default; the resolver turns it CB-safe.
        spec = HeatmapPanelSpec(matrix=np.eye(3), row_labels=("a", "b", "c"), col_labels=("a", "b", "c"))
        assert spec.colormap == HEATMAP_GENERIC
        assert resolve_heatmap_cmap(spec.colormap) == HEATMAP_CMAP


class TestRendererCBDefault:
    def _spec(self):
        m = np.array([[0.1, 0.4, 0.9], [0.3, 0.7, 0.2], [0.5, 0.6, 0.8]])
        return HeatmapPanelSpec(matrix=m, row_labels=("r0", "r1", "r2"),
                                col_labels=("c0", "c1", "c2"), title="t",
                                cell_text=m, colorbar_label="v")

    def test_matplotlib_uses_cb_safe_cmap_by_default(self):
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(1, 1, 1)
        MatplotlibRenderer()._heatmap(ax, self._spec(), fig)
        # The drawn image carries the resolved (CB-safe) colormap, NOT the generic Blues placeholder.
        images = ax.get_images()
        assert images, "heatmap drew no image"
        assert images[0].get_cmap().name == matplotlib.colormaps[HEATMAP_CMAP].name

    def test_plotly_uses_cb_safe_colorscale_by_default(self):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from mlframe.reporting.renderers.plotly import PlotlyRenderer

        fig = make_subplots(rows=1, cols=1)
        PlotlyRenderer()._heatmap(fig, self._spec(), 1, 1)
        heatmaps = [t for t in fig.data if t.type == "heatmap"]
        assert heatmaps, "no heatmap trace emitted"
        # plotly normalises a named scale into an explicit (stop, color) list on the trace; compare against the
        # same normalisation of the CB-safe name (Viridis) and assert it is NOT the generic Blues placeholder.
        viridis = go.Heatmap(colorscale=_mpl_to_plotly_cmap(HEATMAP_CMAP)).colorscale
        blues = go.Heatmap(colorscale=_mpl_to_plotly_cmap(HEATMAP_GENERIC)).colorscale
        assert heatmaps[0].colorscale == viridis
        assert heatmaps[0].colorscale != blues


class TestLinePaletteUnchanged:
    def test_first_ten_are_tab10(self):
        assert LINE_PALETTE[:10] == _TAB10

    def test_full_palette_byte_stable(self):
        assert LINE_PALETTE == _TAB10 + _TAB20_EXT
