"""``MatplotlibRenderer._scatter``, carved out to keep ``matplotlib.py`` under the 1000-LOC house limit.

Same pattern as the plotly backend's ``_plotly_scatter``: the method is defined here as a plain function and
bound onto the class at the bottom of ``matplotlib.py``, so ``MatplotlibRenderer._scatter`` and the
``_render_panel`` dispatch keep resolving unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.reporting.colors import OVERLAY_LINE, TREND_LINE
from mlframe.reporting.spec import ScatterPanelSpec

from ._shared_helpers import _SCATTER_MAX_POINTS, low_evidence_mask, select_per_point

logger = logging.getLogger(__name__)

def _scatter(self, ax, p: ScatterPanelSpec, fig, cbar_axes=None) -> None:
    """Render a scatter panel: subsamples above ``_SCATTER_MAX_POINTS`` (preserving extremes, rasterized), then layers optional error bars, highlighted worst-K points, trend line, overlay band/line, y=x reference and inline labels/colorbar/legend on top."""
    # Lazy, function-local, matching the plotly sibling: the parent module imports this one at its own bottom,
    # so a module-level ``from .matplotlib import ...`` would be a hard cycle. By call time the parent is loaded.
    from .matplotlib import _EDGE_LABEL_FLIP_FRACTION, _err_to_mpl, _set_panel_title

    import matplotlib
    x = np.asarray(p.x)
    y = np.asarray(p.y)
    n = len(x)
    size_arr = p.point_size if isinstance(p.point_size, np.ndarray) else None
    color_arr = p.point_color if isinstance(p.point_color, np.ndarray) else None

    rasterized = False
    if n > _SCATTER_MAX_POINTS:
        from mlframe.reporting.charts import subsample_preserving_extremes
        idx = subsample_preserving_extremes(x, y, sample_size=_SCATTER_MAX_POINTS)
        x, y = x[idx], y[idx]
        if size_arr is not None and len(size_arr) == n:
            size_arr = size_arr[idx]
        if color_arr is not None and len(color_arr) == n:
            color_arr = color_arr[idx]
        rasterized = True  # capped scatter still rasterized so a vector export stays small.

    # Per-point error bars (e.g. Wilson CIs on reliability bins). Drawn before the scatter so the markers
    # sit on top. Subsample never reorders for these CI panels (n is bin-count, well under the cap), so the
    # error arrays align with x/y as-passed.
    # Points whose value rests on too little data to be an observation. Their interval is drawn muted and
    # their marker hollow, so the pair keeps its honest width without carrying an observation's weight.
    weak = low_evidence_mask(p.low_evidence_indices if len(x) == len(np.asarray(p.x)) else None, len(x))

    if p.y_err is not None or p.x_err is not None:
        yerr = _err_to_mpl(p.y_err)
        xerr = _err_to_mpl(p.x_err)

        def _slice_err(err, mask):
            """Select the masked columns of an error spec, keeping the (2, N) asymmetric shape intact."""
            if err is None:
                return None
            arr = np.asarray(err)
            return arr[:, mask] if arr.ndim == 2 else arr[mask]

        if weak.any():
            strong = ~weak
            if strong.any():
                ax.errorbar(x[strong], y[strong], yerr=_slice_err(yerr, strong), xerr=_slice_err(xerr, strong), fmt="none", ecolor="gray", elinewidth=1.0, capsize=3, alpha=0.7, zorder=1)
            ax.errorbar(x[weak], y[weak], yerr=_slice_err(yerr, weak), xerr=_slice_err(xerr, weak), fmt="none", ecolor="0.75", elinewidth=0.6, capsize=0, alpha=0.5, linestyle=":", zorder=1)
        else:
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="none", ecolor="gray", elinewidth=1.0, capsize=3, alpha=0.7, zorder=1)

    # plotly names the scatter trace from `legend_label`; matplotlib never passed it to `ax.scatter`, so
    # the label was dropped AND the legend call below drew an empty box (matplotlib warns about it) on
    # every scatter that set one. Passing it makes the two backends agree and gives the legend content.
    kw: dict[str, Any] = {"alpha": p.point_alpha, "rasterized": rasterized}
    if p.legend_label:
        kw["label"] = p.legend_label
    kw["s"] = size_arr if size_arr is not None else float(p.point_size)
    if color_arr is not None:
        kw["c"] = color_arr
        kw["cmap"] = matplotlib.colormaps[p.colormap]
        if p.color_vmin is not None:
            kw["vmin"] = p.color_vmin
        if p.color_vmax is not None:
            kw["vmax"] = p.color_vmax
    elif p.point_color is not None:
        kw["color"] = p.point_color
    if weak.any() and (~weak).any():
        # Two calls so the weak points can be hollow: matplotlib takes ``facecolors`` per call, not per point,
        # and the colorbar is built from the FILLED call so it still describes the observations.
        strong = ~weak
        _sub = {k: select_per_point(v, strong, len(x)) for k, v in kw.items()}
        sc = ax.scatter(x[strong], y[strong], **_sub)
        _weak_kw = {k: select_per_point(v, weak, len(x)) for k, v in kw.items()}
        _weak_kw.pop("label", None)
        # The colour-mapping keys go with the fill: a hollow marker has nothing to map, and leaving them
        # set makes matplotlib warn that it is ignoring them.
        for _cmap_key in ("c", "cmap", "vmin", "vmax"):
            _weak_kw.pop(_cmap_key, None)
        _weak_kw["facecolors"] = "none"
        _weak_kw["edgecolors"] = "0.55"
        _weak_kw["linewidths"] = 0.9
        _weak_kw["alpha"] = min(1.0, float(p.point_alpha) + 0.2)
        _weak_kw["label"] = "too few rows to read"
        ax.scatter(x[weak], y[weak], **_weak_kw)
    else:
        sc = ax.scatter(x, y, **kw)

    # Emphasised subset (worst-K errors): drawn on top, larger + colored. Indices are positions into the
    # ORIGINAL arrays, so resolve against the pre-subsample data (``p.x`` / ``p.y``), not the capped ``x``/``y``.
    if p.highlight_indices is not None:
        hi_idx = np.asarray(p.highlight_indices, dtype=np.int64)
        ox, oy = np.asarray(p.x), np.asarray(p.y)
        hi_idx = hi_idx[(hi_idx >= 0) & (hi_idx < len(ox))]
        if hi_idx.size:
            base_s = float(p.point_size) if size_arr is None else float(np.median(np.asarray(p.point_size)))
            ax.scatter(ox[hi_idx], oy[hi_idx], s=base_s * 4.0, facecolors="none", edgecolors=p.highlight_color, linewidths=1.5, zorder=5, label="worst-K")

    if p.trend_line is not None and n > 1:
        from mlframe.reporting.renderers._trend import robust_fit_endpoints
        ends = robust_fit_endpoints(np.asarray(p.x), np.asarray(p.y), p.trend_line)
        if ends is not None:
            (tx0, ty0), (tx1, ty1) = ends
            ax.plot([tx0, tx1], [ty0, ty1], color=TREND_LINE, linestyle="-", linewidth=1.6, zorder=4, label=f"robust fit ({p.trend_line})")

    if p.overlay_band is not None:
        bx, blo, bhi = (np.asarray(a) for a in p.overlay_band)
        ax.fill_between(bx, blo, bhi, color=OVERLAY_LINE, alpha=0.18, zorder=3, linewidth=0, label="curve 95% band")

    if p.overlay_line is not None:
        ox_grid, oy_grid, olabel = p.overlay_line
        ax.plot(np.asarray(ox_grid), np.asarray(oy_grid), color=OVERLAY_LINE, linestyle="-", linewidth=1.8, zorder=4, label=olabel)

    if p.perfect_fit_line and n > 0:
        # Span y=x over the UNION of both axes (so it stays the diagonal even when prediction collapse makes
        # y constant) and square the panel so y=x is a true 45-degree line.
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        ax.plot([lo, hi], [lo, hi], "g--", label="Perfect fit")
        if not p.equal_aspect:
            # Probability-vs-probability (calibration): the diagonal spans corner-to-corner at any aspect, so let
            # the panel fill its cell width and align with the histogram below; xlim/ylim are applied just after.
            pass
        elif p.xlim is not None or p.ylim is not None:
            # Explicit limits given: "datalim" would discard set_xlim to satisfy equal aspect (large bubble
            # markers then drive x far past the data); "box" keeps the fixed limits and squares via the box.
            ax.set_aspect("equal", "box")
        else:
            # Equal lo..hi limits on both axes already make the diagonal a true 45-degree line; square via the box
            # ("box" respects the fixed limits). "datalim" would instead adjust the limits to satisfy the aspect and
            # log "Ignoring fixed x limits to fulfill fixed data aspect" on every scatter panel.
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", "box")
    elif p.equal_aspect:
        # The flag used to be read only inside the perfect-fit branch, so asking for a square panel
        # without that diagonal silently did nothing.
        ax.set_aspect("equal", "box")
    if p.xlim is not None:
        ax.set_xlim(*p.xlim)
    if p.ylim is not None:
        ax.set_ylim(*p.ylim)

    if p.inline_labels:
        _lab_colors = p.inline_label_colors if p.inline_label_colors is not None else ()
        # Alignment flips near an edge. A fixed ``ha="right"`` puts the text to the LEFT of its point, so a
        # point in the left margin -- the busiest corner of a reliability diagram -- had its label run off
        # the axis and get clipped mid-word; the same happens vertically for a point at the top.
        _xlo, _xhi = ax.get_xlim()
        _ylo, _yhi = ax.get_ylim()
        # A zero span means a collapsed axis, not a missing measurement; 1.0 just keeps the ratio below
        # finite so every label lands on the 'not near an edge' side rather than dividing by zero.
        _raw_xspan = _xhi - _xlo
        _raw_yspan = _yhi - _ylo
        _xspan = _raw_xspan if _raw_xspan != 0 else 1.0
        _yspan = _raw_yspan if _raw_yspan != 0 else 1.0
        from matplotlib import patheffects as _pe

        for _i, (lx, ly, txt) in enumerate(p.inline_labels):
            _ha = "left" if (lx - _xlo) / _xspan < _EDGE_LABEL_FLIP_FRACTION else "right"
            _va = "top" if (_yhi - ly) / _yspan < _EDGE_LABEL_FLIP_FRACTION else "bottom"
            _colour = _lab_colors[_i] if _i < len(_lab_colors) else "black"
            # A halo in the opposite tone, because one colour cannot serve a label that STRADDLES its
            # marker: white text chosen for a dark bubble turns invisible the moment it runs off the fill
            # onto the panel, and black text chosen for a pale one disappears the other way.
            ax.text(
                lx, ly, txt, fontsize=8, ha=_ha, va=_va, color=_colour, zorder=6,
                path_effects=[_pe.withStroke(linewidth=1.6, foreground=("black" if _colour == "white" else "white"))],
            )

    if p.colorbar_label and color_arr is not None:
        cbar = fig.colorbar(sc, ax=(cbar_axes if cbar_axes is not None else ax))
        cbar.set_label(p.colorbar_label)

    ax.set_xlabel(p.xlabel)
    ax.set_ylabel(p.ylabel)
    _set_panel_title(ax, p.title)
    if p.legend_label or p.perfect_fit_line or p.trend_line or p.overlay_line is not None or p.overlay_band is not None or p.highlight_indices is not None:
        if p.legend_outside:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, framealpha=0.7, borderaxespad=0.0)
        else:
            ax.legend(loc="best", fontsize=8, framealpha=0.7)
    if p.grid:
        ax.grid(True, alpha=0.3)


__all__ = ["_scatter"]
