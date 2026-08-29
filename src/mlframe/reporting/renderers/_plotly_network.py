"""Network/graph panel renderer for the plotly backend.

Carved out of ``plotly.py`` (which had grown past the 1000-LOC house limit) and bound back onto
``PlotlyRenderer`` at that module's bottom via the project's method-rebinding pattern, so ``self`` stays
the first argument and the class's public surface is unchanged. This panel type is entirely self-contained
-- nothing else in the renderer references it -- which makes it the natural seam.
"""
from __future__ import annotations

from typing import List

import numpy as np

from mlframe.reporting.spec import NetworkPanelSpec

from ._plotly_color import _mpl_to_plotly_cmap

import logging

logger = logging.getLogger(__name__)


# Cap on directed-edge arrow annotations. Each arrow is one layout
# annotation; beyond this the topology still renders (lines + nodes) but
# arrowheads are skipped so a large opt-in graph doesn't bloat the layout.
_NETWORK_MAX_ARROWS = 500


def _network(self, fig, p: NetworkPanelSpec, row: int, col: int) -> None:
    """Render a network/graph panel: edges are binned by weight into a handful of width/color buckets and drawn as one ``Scattergl`` line trace per bucket (keeps trace count O(bins) regardless of edge count), plus an invisible edge-midpoint marker trace carrying the continuous weight colorbar and hover text, optional directed-edge arrowheads as data-space annotations (capped at ``_NETWORK_MAX_ARROWS``, silently skipped on axis-ref resolution failure), and a node marker trace with mpl-style area-based sizing."""
    # Lazy, function-local: the parent module imports THIS one at its own bottom, so a module-level
    # ``from .plotly import _go`` would be a hard cycle. By call time the parent is fully loaded.
    from .plotly import _go

    go = _go()

    node_x = np.asarray(p.node_x, dtype=float)
    node_y = np.asarray(p.node_y, dtype=float)
    e_src = np.asarray(p.edge_src, dtype=np.int64)
    e_dst = np.asarray(p.edge_dst, dtype=np.int64)
    weights = np.asarray(p.edge_weight, dtype=float)

    if e_src.size:
        wmin, wmax = float(weights.min()), float(weights.max())
        raw_wspan = wmax - wmin
        # All edges carry the same weight: span is 0, not a caller-supplied falsy value, and dividing by
        # it below would raise. Substitute 1.0 explicitly rather than via `or`, which a falsy-but-legitimate
        # span could never actually pass here since it's always >= 0.
        wspan = raw_wspan if raw_wspan else 1.0
        lo, hi = p.edge_width_range
        colorscale = _mpl_to_plotly_cmap(p.colormap)
        # Bin edges by MI into a handful of width/color buckets: one Scattergl
        # line trace per non-empty bucket keeps trace count O(bins) regardless
        # of edge count (a single line trace can't vary width/color per segment).
        n_bins = min(8, max(1, e_src.size))
        bin_idx = np.minimum(((weights - wmin) / wspan * n_bins).astype(int), n_bins - 1)
        from plotly.colors import sample_colorscale
        for b in range(n_bins):
            mask = bin_idx == b
            if not mask.any():
                continue
            frac = (b + 0.5) / n_bins
            width = lo + frac * (hi - lo)
            color = sample_colorscale(colorscale, [frac])[0]
            xs: List = []
            ys: List = []
            for a, d in zip(e_src[mask], e_dst[mask]):
                xs.extend([node_x[a], node_x[d], None])
                ys.extend([node_y[a], node_y[d], None])
            fig.add_trace(
                go.Scattergl(x=xs, y=ys, mode="lines", line=dict(width=width, color=color), hoverinfo="skip", showlegend=False),
                row=row,
                col=col,
            )
            _label = p.colorbar_label if p.colorbar_label else "edge weight"
            fig.add_trace(
                go.Scattergl(
                    x=[(node_x[a] + node_x[d]) / 2.0 for a, d in zip(e_src[mask], e_dst[mask])],
                    y=[(node_y[a] + node_y[d]) / 2.0 for a, d in zip(e_src[mask], e_dst[mask])],
                    mode="markers", marker=dict(size=6, color=color, opacity=0.01),
                    hovertext=[f"{p.node_label[a]} - {p.node_label[d]}<br>{_label}={w:.4g}"
                               for a, d, w in zip(e_src[mask], e_dst[mask], weights[mask])],
                    hoverinfo="text", showlegend=False),
                row=row, col=col,
            )

        # Invisible marker trace at edge midpoints carries the continuous MI
        # colorbar and a per-edge hover readout without cluttering the plot.
        mid_x = (node_x[e_src] + node_x[e_dst]) / 2.0
        mid_y = (node_y[e_src] + node_y[e_dst]) / 2.0
        fig.add_trace(
            go.Scattergl(
                x=mid_x, y=mid_y, mode="markers",
                marker=dict(size=0.1, color=weights, colorscale=colorscale,
                            showscale=True,
                            colorbar=dict(title=p.colorbar_label) if p.colorbar_label else None),
                text=[f"MI={w:.4f}" for w in weights],
                hoverinfo="text", showlegend=False),
            row=row, col=col,
        )

        # Directed-edge arrowheads via data-space annotations. Axis refs are
        # derived from the subplot grid so multi-panel figures stay correct;
        # any failure falls back to no arrows (lines already convey topology).
        directed = p.edge_directed
        if np.isscalar(directed):
            directed = np.full(e_src.shape, bool(directed))
        else:
            directed = np.asarray(directed, dtype=bool)
        if directed.any() and int(directed.sum()) <= self._NETWORK_MAX_ARROWS:
            try:
                n_cols = len(fig._grid_ref[0])
                idx = (row - 1) * n_cols + col
                suffix = "" if idx == 1 else str(idx)
                xref, yref = f"x{suffix}", f"y{suffix}"
                # ``fig.add_annotation`` re-validates the whole growing ``layout.annotations`` tuple per
                # call (O(n) per mutation -> O(n^2) over a loop, measured 534x at 400 calls in
                # bench_annotation.py / the sibling heatmap fix above); xref/yref are already resolved
                # here (constant for this subplot), so batch every arrow into ONE tuple assignment.
                arrows = [
                    go.layout.Annotation(
                        x=node_x[d], y=node_y[d], ax=node_x[a], ay=node_y[a],
                        xref=xref, yref=yref, axref=xref, ayref=yref,
                        showarrow=True, arrowhead=2, arrowsize=1.2,
                        arrowwidth=1.0, arrowcolor="rgba(80,80,80,0.6)",
                        standoff=6, startstandoff=6,
                    )
                    for a, d, dirn in zip(e_src, e_dst, directed)
                    if dirn
                ]
                if arrows:
                    fig.layout.annotations = fig.layout.annotations + tuple(arrows)
            except Exception:
                logger.debug("network arrows skipped (subplot axis-ref resolution failed)", exc_info=True)

    # Nodes: one marker trace. size follows matplotlib ``scatter(s=)`` area
    # semantics; convert to plotly's pixel diameter (sqrt(area) * 1.33).
    sizes = np.sqrt(np.maximum(np.asarray(p.node_size, dtype=float), 0.0)) * 1.33
    hovertext = list(p.node_hovertext) if p.node_hovertext else list(p.node_label)
    fig.add_trace(
        go.Scattergl(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(size=sizes, color=list(p.node_color),
                        line=dict(width=0.5, color="black")),
            text=list(p.node_label), textposition="top center", textfont=dict(size=8),
            hovertext=hovertext, hoverinfo="text", showlegend=False),
        row=row, col=col,
    )

    for _lbl, _col in p.node_legend or ():
        fig.add_trace(
            go.Scattergl(x=[None], y=[None], mode="markers", marker=dict(size=9, color=_col), name=_lbl, showlegend=True, hoverinfo="skip"),
            row=row,
            col=col,
        )
    if p.node_legend:
        fig.update_layout(showlegend=True)

    fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=False, zeroline=False, showticklabels=False)
