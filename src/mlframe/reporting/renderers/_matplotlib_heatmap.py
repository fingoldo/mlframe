"""The matplotlib heatmap panels, carved out of ``matplotlib.py`` for the 1000-LOC ceiling.

``_heatmap`` and ``_confusion_margins`` are the two largest panel builders in that renderer and they share
the same machinery -- a colour scale, per-cell text whose colour is flipped by luminance, and a colorbar
that shrinks the axes it attaches to. They are bound back onto ``MatplotlibRenderer`` at the bottom of
``matplotlib.py``, which mirrors how ``_plotly_heatmap.py`` already relates to ``plotly.py``.

Both keep their ``self`` parameter: they are methods that happen to live in another module, not free
functions, and they call back into sibling methods through it.
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional

import numpy as np

from mlframe.reporting.colors import CONFUSION_COL_MARGIN, CONFUSION_ROW_MARGIN, TREND_LINE, resolve_heatmap_cmap
from mlframe.reporting.spec import ConfusionMarginsPanelSpec, HeatmapPanelSpec

from ._shared_helpers import (
    _HEATMAP_CELL_TEXT_MAX,
    _finite_range,
    _thin_tick_positions,
    heatmap_value_to_index,
    rotated_tick_pitch_in,
    ticks_that_fit,
    truncate_bar_label,
)

logger = logging.getLogger("mlframe.reporting.renderers._matplotlib_heatmap")


#: Spec dash names mapped to matplotlib linestyles. Typed as the Literal matplotlib actually accepts so
#: the value survives a ``.get()`` without mypy losing it -- the runtime value is always one of these.
_DASH_STYLES: Dict[str, Literal["-", "--", ":", "-."]] = {"solid": "-", "dash": "--", "dot": ":", "dashdot": "-."}


def _calmest_corner(matrix: np.ndarray) -> str:
    """The legend ``loc`` whose quadrant carries the least extreme values.

    A fixed corner covers whatever happens to be there, and on a drift heatmap the rows are ordered by peak
    PSI and the columns by time -- so "lower right" is exactly where the worst feature's latest, largest
    numbers live. Seen directly: the legend sat on the two biggest cells of the two worst rows.

    Ranking quadrants by mean magnitude puts it over the calmest one instead. It cannot avoid covering
    something on a grid that is interesting everywhere, but it can avoid covering the part a reader came
    for. All-NaN quadrants sort as calm, which is right: a blank corner is the best possible place for it.
    """
    quadrants = {
        "upper left": np.s_[: matrix.shape[0] // 2, : matrix.shape[1] // 2],
        "upper right": np.s_[: matrix.shape[0] // 2, matrix.shape[1] // 2 :],
        "lower left": np.s_[matrix.shape[0] // 2 :, : matrix.shape[1] // 2],
        "lower right": np.s_[matrix.shape[0] // 2 :, matrix.shape[1] // 2 :],
    }
    best, best_score = "lower right", np.inf
    for loc, sl in quadrants.items():
        block = np.abs(np.asarray(matrix, dtype=float)[sl])
        score = -1.0 if block.size == 0 or not np.isfinite(block).any() else float(np.nanmean(block))
        if score < best_score:
            best, best_score = loc, score
    return best


def _heatmap(self, ax, p: HeatmapPanelSpec, fig) -> None:
    """Render a matrix heatmap: cell text (auto-flipped color by luminance) when the grid is small enough, iso-value threshold contours, and an optional trend/y=x line mapped from value-space into bin-index space via the panel's own binning range."""
    # Imported at call time, not at module scope: ``matplotlib.py`` binds these two functions onto its
    # renderer at the BOTTOM of its own module, so a top-level import back into it would be circular.
    from .matplotlib import _HEATMAP_TICK_FONTSIZE, _place_legend, _set_panel_title

    import matplotlib
    cmap_name = resolve_heatmap_cmap(p.colormap)
    cm = matplotlib.colormaps[cmap_name]
    # A density panel carrying ``trend_xy`` (the regression pred-vs-true heatmap) reads "bottom-up"
    # (row 0 = lowest value), so it needs origin="lower"; other heatmaps (confusion / drift) keep the
    # default top-down matrix orientation.
    _heatmap_origin = "lower" if getattr(p, "trend_xy", None) is not None else "upper"
    # Explicit colour bounds when the builder pinned them: a diverging colormap's midpoint means nothing
    # unless the scale is anchored, and autoscale anchors it to whatever the data happens to span.
    _clim = {}
    if p.color_vmin is not None:
        _clim["vmin"] = p.color_vmin
    if p.color_vmax is not None:
        _clim["vmax"] = p.color_vmax
    im = ax.imshow(p.matrix, cmap=cm, aspect="auto", origin=_heatmap_origin, **_clim)
    # How many names the axis can actually hold, from its real size rather than a fixed 8. A drift heatmap
    # grows its figure with the feature count (40 rows over ~14 inches) and then named 8 of them, so the
    # height it bought went to unlabelled rows. Labels are truncated as well, which the bar branches have
    # always done and this one never did -- a generated feature name runs off the left edge otherwise.
    # Deferred until after the colorbar, which shrinks the axes it is attached to: budgeting against the
    # pre-colorbar width bought a fifth more labels than the axes ends up able to hold.
    def _apply_tick_budget() -> None:
        """Set both axes' ticks to what the axes can hold, measured at its post-colorbar size."""
        try:
            _pos = ax.get_position()
            _fig_w, _fig_h = (float(v) for v in ax.figure.get_size_inches())
            _w_in: Optional[float] = float(_pos.width) * _fig_w
            _h_in: Optional[float] = float(_pos.height) * _fig_h
        except Exception:
            logger.debug("could not measure heatmap axes for tick budgeting; falling back to the fixed cap", exc_info=True)
            _w_in = _h_in = None
        _xt = _thin_tick_positions(len(p.col_labels), ticks_that_fit(_w_in, len(p.col_labels), pitch_in=rotated_tick_pitch_in(_HEATMAP_TICK_FONTSIZE, 45)))
        ax.set_xticks(_xt)
        ax.set_xticklabels([truncate_bar_label(p.col_labels[i]) for i in _xt], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
        _yt = _thin_tick_positions(len(p.row_labels), ticks_that_fit(_h_in, len(p.row_labels), pitch_in=rotated_tick_pitch_in(_HEATMAP_TICK_FONTSIZE, 0)))
        ax.set_yticks(_yt)
        ax.set_yticklabels([truncate_bar_label(p.row_labels[i]) for i in _yt], fontsize=_HEATMAP_TICK_FONTSIZE)
    rng = _finite_range(p.matrix)
    drew_cell_text = p.cell_text is not None and rng is not None and p.matrix.size <= _HEATMAP_CELL_TEXT_MAX
    if drew_cell_text and p.cell_text is not None and rng is not None:
        from mlframe.reporting.colors import auto_text_colors_batch
        # Compute global vmin / vmax so each cell's text color reflects
        # its position in the actual color range — naive
        # ``< 0.5`` threshold fails when the matrix range is e.g.
        # [0.3, 0.85] (all values map to the high-luminance end of
        # the colormap and white text becomes invisible).
        mat = p.matrix
        vmin, vmax = rng
        # The bounds the CELLS were drawn with, when the builder pinned them. The text colour is chosen by
        # sampling the colormap at the cell's position in the scale, so it has to use the same scale the
        # image did -- otherwise a pinned range paints pale cells while the text colour still believes the
        # data range, and white labels land on a near-white fill.
        if p.color_vmin is not None:
            vmin = float(p.color_vmin)
        if p.color_vmax is not None:
            vmax = float(p.color_vmax)
        # One vectorized colormap sample for the whole grid instead of one matplotlib call per cell
        # (bit-identical to the per-cell auto_text_color -- same pattern PlotlyRenderer._heatmap uses).
        text_colors = auto_text_colors_batch(np.where(np.isfinite(mat), mat, vmin), cmap_name, vmin=vmin, vmax=vmax)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(p.cell_text[i, j], p.text_format), ha="center", va="center", fontsize=7, color=text_colors[i, j])
    # Iso-value contour overlays at named matrix levels (PSI 0.10 / 0.25 triage lines on the drift heatmap).
    # Contour coords are the imshow cell-center grid (0..ncols-1, 0..nrows-1) so lines land between cells.
    if p.threshold_contours:
        mat = np.asarray(p.matrix, dtype=float)
        contour_legend: list = []
        if mat.ndim == 2 and mat.shape[0] >= 2 and mat.shape[1] >= 2:
            gx, gy = np.meshgrid(np.arange(mat.shape[1]), np.arange(mat.shape[0]))
            # Hoisted out of the loop: both are full-matrix reductions over the same unchanged matrix,
            # so recomputing them per contour level was O(levels * cells) for an O(cells) answer. The
            # plotly twin already hoists them.
            lo, hi = float(np.nanmin(mat)), float(np.nanmax(mat))
            for _entry in p.threshold_contours:
                level, color = _entry[0], _entry[1]
                dash = _entry[2] if len(_entry) > 2 else "solid"
                label = _entry[3] if len(_entry) > 3 else ""
                if lo < level < hi:  # contour only exists when the level is crossed
                    cs = ax.contour(gx, gy, mat, levels=[level], colors=[color], linewidths=1.4, linestyles=_DASH_STYLES.get(dash, "-"))
                    if label and not drew_cell_text:
                        ax.clabel(cs, fmt={level: label}, fontsize=7)
                    elif label:
                        # An inline contour label follows the contour, so on a grid that also carries
                        # per-cell numbers it runs diagonally across them: seen writing "significant 0.25"
                        # over two cells of a drift heatmap. The cell values are the more precise reading,
                        # so the wording moves to the legend rather than competing for the same pixels.
                        # The plotly twin suppresses its own inline label in the same case and keeps the
                        # trace name, so both backends still name the threshold.
                        from matplotlib.lines import Line2D as _CLine

                        contour_legend.append(_CLine([], [], color=color, linewidth=1.4, label=label, linestyle=_DASH_STYLES.get(dash, "-")))
            if contour_legend:
                ax.legend(handles=contour_legend, fontsize=6, loc=_calmest_corner(mat), framealpha=0.85)
    if p.trend_line is not None and p.trend_xy is not None:
        from mlframe.reporting.renderers._trend import robust_fit_endpoints
        # The imshow axes live in BIN-INDEX space (0..nbins-1); robust_fit_endpoints + the y=x
        # diagonal are in VALUE space. Map value -> index using the SAME (lo, hi) the panel binned on
        # (lo = min over both arrays, hi = max), else the line is plotted at value coords (~1e4) on a
        # 0..79 axis, auto-expanding the axis and squishing the density into a corner.
        _xv = np.asarray(p.trend_xy[0], dtype=np.float64).ravel()
        _yv = np.asarray(p.trend_xy[1], dtype=np.float64).ravel()
        _fin = np.isfinite(_xv) & np.isfinite(_yv)
        _nb = len(p.col_labels)
        if int(_fin.sum()) >= 2 and _nb >= 2:
            _lo = float(min(_xv[_fin].min(), _yv[_fin].min()))
            _hi = float(max(_xv[_fin].max(), _yv[_fin].max()))
            if _hi > _lo:
                # Shared with the plotly renderer so the two backends cannot drift on this map again.
                _to_idx = heatmap_value_to_index(_lo, _hi, _nb)
                # y=x reference in index space (origin="lower" -> bottom-left to top-right).
                ax.plot([0, _nb - 1], [0, _nb - 1], color="0.4", linestyle=":", linewidth=1.0, label="y=x")
                ends = robust_fit_endpoints(_xv, _yv, p.trend_line)
                if ends is not None:
                    (tx0, ty0), (tx1, ty1) = ends
                    ax.plot(
                        [_to_idx(tx0), _to_idx(tx1)], [_to_idx(ty0), _to_idx(ty1)],
                        color=TREND_LINE, linestyle="-", linewidth=1.6,
                        label=f"robust fit ({p.trend_line})",
                    )
                ax.set_xlim(-0.5, _nb - 0.5)
                ax.set_ylim(-0.5, _nb - 0.5)
                _place_legend(ax)
    cbar = fig.colorbar(im, ax=ax)
    if p.colorbar_label:
        cbar.set_label(p.colorbar_label)
    _apply_tick_budget()
    ax.set_xlabel(p.xlabel)
    ax.set_ylabel(p.ylabel)
    _set_panel_title(ax, p.title)

def _confusion_margins(self, ax, p: ConfusionMarginsPanelSpec, fig) -> None:
    """Render a confusion matrix as a 2x2 small-multiple (predicted-volume bar on top, true-support bar on the right) by subdividing the panel's own subplotspec into sub-axes, replacing the placeholder ``ax``."""
    # Imported at call time, not at module scope: ``matplotlib.py`` binds these two functions onto its
    # renderer at the BOTTOM of its own module, so a top-level import back into it would be circular.
    from .matplotlib import _set_panel_title

    import matplotlib
    from mlframe.reporting.colors import resolve_heatmap_cmap
    # The single panel cell hosts a 2x2 small-multiple: top bar (predicted volume), heatmap + right bar (true
    # support). Subdividing the cell's own subplotspec keeps the layout grid-driven and aligned with siblings;
    # the passed ``ax`` is the placeholder we replace with the sub-axes.
    cmap_name = resolve_heatmap_cmap(p.colormap)
    cm = matplotlib.colormaps[cmap_name]
    K = p.matrix.shape[0]
    ax.set_axis_off()
    gs = ax.get_subplotspec().subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0.05, hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    im = ax_hm.imshow(p.matrix, cmap=cm, aspect="auto")
    # Thin to the shared ceiling on BOTH axes. One tick per class smears past ~30 classes, and the plotly
    # twin already thins to _HEATMAP_MAX_TICKS -- so a large-K confusion matrix rendered with a readable
    # axis on one backend and an unreadable band on the other, from the same spec.
    _xt = _thin_tick_positions(len(p.col_labels))
    _yt = _thin_tick_positions(len(p.row_labels))
    ax_hm.set_xticks(_xt)
    ax_hm.set_xticklabels([p.col_labels[i] for i in _xt], rotation=45, ha="right", fontsize=8)
    ax_hm.set_yticks(_yt)
    ax_hm.set_yticklabels([p.row_labels[i] for i in _yt], fontsize=8)
    ax_hm.set_xlabel(p.xlabel)
    ax_hm.set_ylabel(p.ylabel)
    rng = _finite_range(p.matrix)
    if p.cell_text is not None and rng is not None and p.matrix.size <= _HEATMAP_CELL_TEXT_MAX:
        from mlframe.reporting.colors import auto_text_colors_batch
        vmin, vmax = rng
        # One vectorized colormap sample for the whole grid instead of one matplotlib call per cell.
        text_colors = auto_text_colors_batch(np.where(np.isfinite(p.matrix), p.matrix, vmin), cmap_name, vmin=vmin, vmax=vmax)
        for i in range(K):
            for j in range(p.matrix.shape[1]):
                ax_hm.text(j, i, format(p.cell_text[i, j], p.text_format), ha="center", va="center", fontsize=7, color=text_colors[i, j])

    pos = np.arange(K)
    # Top bar: predicted-class volume, aligned to the heatmap columns (shared x, ticks hidden -- the heatmap owns them).
    ax_top.bar(pos, np.asarray(p.col_margin, dtype=float), color=CONFUSION_COL_MARGIN, width=0.8)
    ax_top.set_xlim(-0.5, K - 0.5)
    ax_top.set_xticks([])
    ax_top.tick_params(axis="y", labelsize=7)
    ax_top.set_ylabel(p.col_margin_label, fontsize=7)
    # Right bar: per-true-class support, aligned to the heatmap rows (imshow y runs top->bottom, so invert).
    ax_right.barh(pos, np.asarray(p.row_margin, dtype=float), color=CONFUSION_ROW_MARGIN, height=0.8)
    ax_right.set_ylim(-0.5, K - 0.5)
    ax_right.invert_yaxis()
    ax_right.set_yticks([])
    ax_right.tick_params(axis="x", labelsize=7, rotation=45)
    ax_right.set_xlabel(p.row_margin_label, fontsize=7)

    cbar = fig.colorbar(im, ax=ax_right, fraction=0.25, pad=0.35)
    if p.colorbar_label:
        cbar.set_label(p.colorbar_label, fontsize=8)
    title = p.title if not p.note else f"{p.title}\n{p.note}"
    _set_panel_title(ax_top, title)
