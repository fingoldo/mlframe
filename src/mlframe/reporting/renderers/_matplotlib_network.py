"""``MatplotlibRenderer._network``, carved out to keep ``matplotlib.py`` under the 1000-LOC house limit.

Bound back onto the class at the bottom of ``matplotlib.py``, so the ``_render_panel`` dispatch and any external
``MatplotlibRenderer._network`` reference keep resolving unchanged.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.reporting.colors import resolve_heatmap_cmap
from mlframe.reporting.spec import NetworkPanelSpec

# Label cap and directed-edge arrowhead ceiling, shared with the plotly renderer so one spec draws the same network on both
# backends. Imported from the sibling that owns them rather than restated.
from ._plotly_network import _NETWORK_LABEL_MAXLEN, _NETWORK_MAX_ARROWS
from ._shared_helpers import _TITLE_REF_WIDTH_IN, network_label_indices, non_colliding_label_indices, truncate_bar_label

# The parent module's logger name: these lines predate the split, and log filters select them by that name.
logger = logging.getLogger("mlframe.reporting.renderers.matplotlib")


def _network(self, ax, p: NetworkPanelSpec, fig) -> None:
    """Render a node-link network panel: edges as a single ``LineCollection`` (one draw call for O(E) edges, width+color both encoding weight), per-edge arrows for directed edges, then nodes on top with labels and an optional node-color legend."""
    from .matplotlib import _measured_axis_in, _place_legend, _set_panel_title  # the parent binds this function at its import
    import matplotlib
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    nx_pos = np.column_stack([np.asarray(p.node_x, dtype=float), np.asarray(p.node_y, dtype=float)])
    e_src = np.asarray(p.edge_src, dtype=np.int64)
    e_dst = np.asarray(p.edge_dst, dtype=np.int64)
    weights = np.asarray(p.edge_weight, dtype=float)

    # Edges as a single LineCollection: O(E) artists collapse to one draw
    # call, so thousands of edges stay cheap. Width + color both encode MI.
    if e_src.size:
        segments = [[tuple(nx_pos[a]), tuple(nx_pos[b])] for a, b in zip(e_src, e_dst)]
        wmin, wmax = float(weights.min()), float(weights.max())
        norm = Normalize(vmin=wmin, vmax=wmax if wmax > wmin else wmin + 1e-9)
        # Through the resolver, like every other colormap lookup here: ``NetworkPanelSpec.colormap``
        # defaults to the HEATMAP_GENERIC sentinel, which is a token rather than a colormap name, so the
        # raw subscript raised KeyError('__mlframe_default_heatmap__') on any network panel that did not
        # name a colormap -- while the plotly twin silently fell back to Viridis with a warning.
        cmap = matplotlib.colormaps[resolve_heatmap_cmap(p.colormap)]
        lo, hi = p.edge_width_range
        if wmax > wmin:
            lws = lo + (weights - wmin) / (wmax - wmin) * (hi - lo)
        else:
            lws = np.full_like(weights, (lo + hi) / 2.0)
        lc = LineCollection(segments, linewidths=lws.tolist(), colors=cmap(norm(weights)), alpha=0.8, zorder=1)
        ax.add_collection(lc)

        # Arrows for directed edges, under the SAME ceiling the plotly twin applies. The old comment
        # leaned on "the friend-graph max_nodes guard keeps edge counts modest", which is a property of
        # one caller rather than of the renderer -- and the two backends then disagreed about what the
        # figure shows: past 500 directed edges plotly drew no arrowheads and matplotlib drew all of
        # them, from one spec. (``Axes.annotate`` appends in O(1), unlike plotly's, so this bounds an
        # unbounded constant rather than a complexity class.)
        directed = p.edge_directed
        if np.isscalar(directed):
            directed = np.full(e_src.shape, bool(directed))
        else:
            directed = np.asarray(directed, dtype=bool)
        if int(directed.sum()) > _NETWORK_MAX_ARROWS:
            logger.debug("network panel has %d directed edges; arrowheads skipped past the %d cap, matching the plotly renderer",
                         int(directed.sum()), _NETWORK_MAX_ARROWS)
            directed = np.zeros_like(directed)
        for a, b, d in zip(e_src, e_dst, directed):
            if d:
                ax.annotate("", xy=tuple(nx_pos[b]), xytext=tuple(nx_pos[a]),
                            arrowprops=dict(arrowstyle="-|>", color="0.35",
                                            alpha=0.6, shrinkA=8, shrinkB=8),
                            zorder=2)

        # A colorbar over edge weights that are all the same encodes nothing, and matplotlib renders the
        # degenerate range with an offset exponent ("1e-9+1") that reads as a real scale. The spectral
        # embedding passes a constant weight vector, so it drew exactly that beside every figure.
        if wmax > wmin:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            if p.colorbar_label:
                cbar.set_label(p.colorbar_label)

    ax.scatter(nx_pos[:, 0], nx_pos[:, 1], s=np.asarray(p.node_size, dtype=float), c=list(p.node_color), edgecolors="black", linewidths=0.5, zorder=3)
    # Only the biggest nodes are named, and the names are truncated. Every node used to be labelled at
    # full length, centred on its own marker, so a friend graph rendered its names as a single illegible
    # mat over the middle of the panel. The unlabelled nodes keep their marker, their size and their
    # colour -- the graph still shows them, it just does not try to name all 200.
    # Capping the COUNT is not the same as keeping them readable: a spectral embedding whose nodes
    # collapse into one region printed its surviving names on top of each other as a black smudge. Of
    # the biggest nodes, keep the ones whose label boxes actually clear each other.
    _cand = network_label_indices(p.node_size)
    _texts = [truncate_bar_label(p.node_label[_i], _NETWORK_LABEL_MAXLEN) for _i in _cand]
    _x_lo, _x_hi = float(np.min(nx_pos[:, 0])), float(np.max(nx_pos[:, 0]))
    _y_lo, _y_hi = float(np.min(nx_pos[:, 1])), float(np.max(nx_pos[:, 1]))
    _sizes = np.asarray(p.node_size, dtype=float).ravel()
    # Explicit None checks rather than ``measured or REF``: a measured extent of exactly 0 is a
    # degenerate axes, not a failed measurement, and the two deserve different treatment for different
    # reasons -- the same trap the panel-title width budget documents a few hundred lines up.
    _panel_w = _measured_axis_in(ax, horizontal=False)
    _panel_h = _measured_axis_in(ax, horizontal=True)
    _keep_lbl = non_colliding_label_indices(
        nx_pos[_cand, 0], nx_pos[_cand, 1], _texts, fontsize=7,
        x_span=max(_x_hi - _x_lo, 1e-9), y_span=max(_y_hi - _y_lo, 1e-9),
        width_in=_TITLE_REF_WIDTH_IN if _panel_w is None else _panel_w,
        height_in=_TITLE_REF_WIDTH_IN if _panel_h is None else _panel_h,
        priority=[_sizes[_i] for _i in _cand] if _sizes.size == len(p.node_label) else None,
    )
    for _j in _keep_lbl:
        _x, _y = nx_pos[_cand[_j]]
        ax.annotate(_texts[_j], (_x, _y), fontsize=7, ha="center", va="center", zorder=4)

    if p.node_legend:
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=8, label=lbl) for lbl, col in p.node_legend]
        _place_legend(ax, handles, [h.get_label() for h in handles])

    _set_panel_title(ax, p.title)
    ax.set_xlabel(p.xlabel)
    ax.set_ylabel(p.ylabel)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.12)
