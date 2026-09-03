"""Interactivity enrichment for the plotly renderer's HTML output.

HTML reports are meant to be explored, not read like a static PNG. These helpers set per-panel-type
interactivity (unified hover for line panels, rangesliders for temporal panels, rich hovertemplates,
clickable legends) and a cleaned-up modebar, gated so each property lands only where it is correct
(e.g. unified hover is wrong for scatter/heatmap; a rangeslider wastes vertical space on non-temporal charts).
"""

from __future__ import annotations

from typing import Any

from ._shared_helpers import plotly_axis_suffix

from mlframe.reporting.spec import (
    BarPanelSpec, HeatmapPanelSpec, HistogramPanelSpec, LinePanelSpec, ScatterPanelSpec, ViolinPanelSpec,
)

# Rarely-used buttons dropped from the modebar; zoom/pan/reset/download stay. lasso/select only make sense
# for point selection on scatter and confuse on line/heatmap panels, so they go for the whole figure.
_MODEBAR_REMOVE = ("lasso2d", "select2d", "autoScale2d")

# Hovertemplate axis-name hints keyed by xlabel/ylabel substrings the chart builders use for the key
# panel types (ROC/PR/calibration). Falls back to a generic x/y template when nothing matches.
_KEY_PANEL_TEMPLATES = (
    (("fpr", "tpr"), "FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra>%{fullData.name}</extra>"),
    (("recall", "precision"), "Recall=%{x:.3f}<br>Precision=%{y:.3f}<extra>%{fullData.name}</extra>"),
    (("predicted", "observed"), "Predicted=%{x:.3f}<br>Observed=%{y:.3f}<extra>%{fullData.name}</extra>"),
    (("predicted", "fraction"), "Predicted=%{x:.3f}<br>Observed=%{y:.3f}<extra>%{fullData.name}</extra>"),
)


def _trace_axis(tr) -> str:
    """The x-axis id a plotly trace is bound to; the FIRST subplot leaves it unset, which means ``"x"``.

    Written out rather than ``getattr(tr, "xaxis", None) or "x"``: that idiom also swallows a legitimately falsy
    value, and it was repeated at three call sites here.
    """
    axis = getattr(tr, "xaxis", None)
    return "x" if axis is None else str(axis)


def _line_is_temporal(p: LinePanelSpec) -> bool:
    """Whether the line panel's x-axis is a time axis (triggers the rangeslider)."""
    return bool(getattr(p, "x_is_time", False))


def _line_is_multiseries(p: LinePanelSpec) -> bool:
    """Whether the line panel plots more than one y-series (affects hovertemplate/legend handling)."""
    return isinstance(p.y, tuple) and len(p.y) > 1


def _key_template(p: LinePanelSpec) -> str | None:
    """Look up a nicer hovertemplate for known metric-pair panels (ROC, PR, calibration) by matching axis labels; returns None when no known template matches (generic x/y hover is then used)."""
    xl = (p.xlabel or "").lower()
    yl = (p.ylabel or "").lower()
    for (xkey, ykey), tmpl in _KEY_PANEL_TEMPLATES:
        if xkey in xl and ykey in yl:
            return tmpl
    return None


def html_config() -> dict:
    """``write_html`` config: cleaner modebar (no lasso/select), no plotly logo, responsive sizing."""
    return dict(displaylogo=False, modeBarButtonsToRemove=list(_MODEBAR_REMOVE), responsive=True)


def apply_interactivity(fig: Any, spec, *, static_legend: bool = False) -> None:
    """Set per-panel-type interactivity props on an already-rendered figure.

    Gating: unified hover + rich line templates only on LinePanelSpec; rangeslider only on temporal line
    panels; clickable-legend toggles only when the figure carries a legend (>1 trace and legend shown).
    Scatter/heatmap keep plotly's default closest-point hover (unified hover misreads them).
    """
    panels = [p for row in spec.panels for p in row if p is not None]
    line_panels = [p for p in panels if isinstance(p, LinePanelSpec)]
    has_line = bool(line_panels)
    has_temporal = any(_line_is_temporal(p) for p in line_panels)
    has_heatmap = any(isinstance(p, HeatmapPanelSpec) for p in panels)

    # Unified hover (all series at the hovered x) only when the figure is line-dominated and carries no heatmap;
    # on a mixed line+heatmap figure unified hover spills wrong readouts onto the heatmap cells.
    if has_line and not has_heatmap:
        fig.update_layout(hovermode="x unified")
    elif has_line and has_heatmap:
        cols = max((len(r) for r in spec.panels), default=0)
        for r, row in enumerate(spec.panels):
            for c in range(cols):
                if not isinstance(row[c] if c < len(row) else None, LinePanelSpec):
                    continue
                _sfx = plotly_axis_suffix(fig, r + 1, c + 1, cols)
                fig.update_layout(**{f"xaxis{_sfx}": dict(showspikes=True, spikemode="across")})

    # Clickable legend: single-click hides a series, double-click isolates it. Only meaningful when a legend
    # is actually drawn (static export, or any multi-trace legend); harmless no-op otherwise.
    fig.update_layout(legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"))

    # Cleaner modebar baked into the layout too (so a figure shown via fig.show() / embedded without our
    # html_config still drops the logo); write_html additionally strips lasso/select via html_config().
    fig.update_layout(modebar=dict(remove=list(_MODEBAR_REMOVE)))

    _apply_line_traces(fig, spec)
    _apply_nonline_traces(fig, spec)

    if has_temporal:
        _apply_rangeslider(fig, spec)


def _apply_line_traces(fig, spec) -> None:
    """Rich hovertemplate on the key line panels (ROC/PR/calibration); generic x/y template elsewhere on lines."""
    # Map each subplot (row,col) carrying a LinePanelSpec to its axis suffix, so we template only line traces.
    cols = max((len(r) for r in spec.panels), default=0)
    line_axes: dict[str, str | None] = {}
    for r, row in enumerate(spec.panels):
        for c in range(cols):
            panel = row[c] if c < len(row) else None
            if not isinstance(panel, LinePanelSpec):
                continue
            suffix = plotly_axis_suffix(fig, r + 1, c + 1, cols)
            tmpl = _key_template(panel)
            if tmpl is None:
                x_fmt = "|%Y-%m-%d %H:%M" if _line_is_temporal(panel) else ":.6g"
                tmpl = "%{xaxis.title.text}=%{x" + x_fmt + "}<br>%{yaxis.title.text}=%{y:.6g}" "<extra>%{fullData.name}</extra>"
            line_axes[f"x{suffix}"] = tmpl

    for tr in fig.data:
        if tr.type not in ("scatter", "scattergl"):
            continue
        if "lines" not in (tr.mode or ""):
            continue
        xax = _trace_axis(tr)
        tmpl = line_axes.get(xax)
        if tmpl is not None and tr.hovertemplate is None and tr.hoverinfo != "skip":
            tr.hovertemplate = tmpl


_NONLINE_TEMPLATES = {
    ScatterPanelSpec: "{x}=%{{x}}<br>{y}=%{{y}}<extra>%{{fullData.name}}</extra>",
    HistogramPanelSpec: "{x}=%{{x}}<br>{y}=%{{y}}<extra></extra>",
    BarPanelSpec: "{x}=%{{x}}<br>{y}=%{{y}}<extra>%{{fullData.name}}</extra>",
    ViolinPanelSpec: "{x}=%{{x}}<br>{y}=%{{y}}<extra></extra>",
}


def _axis_names(panel) -> tuple:
    """Human axis names for a hover readout, falling back to plain x/y when the panel set no label."""
    return (getattr(panel, "xlabel", "") or "x", getattr(panel, "ylabel", "") or "y")


def _apply_nonline_traces(fig, spec) -> None:
    """Hovertemplate for the scatter / histogram / bar / violin panels, which otherwise keep plotly's raw default."""
    cols = max((len(r) for r in spec.panels), default=0)
    axis_templates: dict[str, str] = {}
    for r, row in enumerate(spec.panels):
        for c in range(cols):
            panel = row[c] if c < len(row) else None
            tmpl = _NONLINE_TEMPLATES.get(type(panel))
            if tmpl is None:
                continue
            suffix = plotly_axis_suffix(fig, r + 1, c + 1, cols)
            xn, yn = _axis_names(panel)
            axis_templates[f"x{suffix}"] = tmpl.format(x=xn, y=yn)

    # A builder that attached per-point support text wins over the generic axis-name template: the denominator is
    # the thing a reader most needs and least often has.
    supports: dict[str, tuple] = {}
    for r, row in enumerate(spec.panels):
        for c in range(cols):
            panel = row[c] if c < len(row) else None
            ht = getattr(panel, "hovertext", None)
            if ht:
                supports["x" + plotly_axis_suffix(fig, r + 1, c + 1, cols)] = tuple(ht)

    for tr in fig.data:
        if tr.type not in ("scatter", "scattergl", "bar", "histogram", "violin"):
            continue
        sup = supports.get(_trace_axis(tr))
        if sup and tr.hovertext is None and tr.hoverinfo != "skip":
            # ``arr or ()`` evaluates an ndarray's truth value, which raises for size > 1 -- and the trace x IS an
            # ndarray now that the renderers pass arrays to plotly natively.
            _tx = getattr(tr, "x", None)
            n_pts = 0 if _tx is None else len(_tx)
            if n_pts == len(sup):
                tr.hovertext = list(sup)
                tr.hoverinfo = "text"
                continue
        if tr.type in ("scatter", "scattergl") and "lines" in (tr.mode or ""):
            continue  # handled by _apply_line_traces, which knows the metric-pair templates
        # The old gate skipped every marker-only trace, which is exactly the chosen-threshold operating point on a
        # ROC/PR panel -- the single most-hovered mark on the figure.
        if tr.hovertemplate is not None or tr.hoverinfo == "skip":
            continue
        tmpl = axis_templates.get(_trace_axis(tr))
        if tmpl is not None:
            tr.hovertemplate = tmpl


def _apply_rangeslider(fig, spec) -> None:
    """Rangeslider + range-selector zoom buttons on temporal line panels only (never on non-temporal charts)."""
    cols = max((len(r) for r in spec.panels), default=0)
    for r, row in enumerate(spec.panels):
        for c in range(cols):
            panel = row[c] if c < len(row) else None
            if not (isinstance(panel, LinePanelSpec) and _line_is_temporal(panel)):
                continue
            fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06), row=r + 1, col=c + 1)
