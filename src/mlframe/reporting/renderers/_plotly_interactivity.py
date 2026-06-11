"""Interactivity enrichment for the plotly renderer's HTML output.

HTML reports are meant to be explored, not read like a static PNG. These helpers set per-panel-type
interactivity (unified hover for line panels, rangesliders for temporal panels, rich hovertemplates,
clickable legends) and a cleaned-up modebar, gated so each property lands only where it is correct
(e.g. unified hover is wrong for scatter/heatmap; a rangeslider wastes vertical space on non-temporal charts).
"""

from __future__ import annotations

from typing import Any

from mlframe.reporting.spec import HeatmapPanelSpec, LinePanelSpec

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


def _line_is_temporal(p: LinePanelSpec) -> bool:
    return bool(getattr(p, "x_is_time", False))


def _line_is_multiseries(p: LinePanelSpec) -> bool:
    return isinstance(p.y, tuple) and len(p.y) > 1


def _key_template(p: LinePanelSpec) -> str | None:
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

    # Clickable legend: single-click hides a series, double-click isolates it. Only meaningful when a legend
    # is actually drawn (static export, or any multi-trace legend); harmless no-op otherwise.
    fig.update_layout(legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"))

    # Cleaner modebar baked into the layout too (so a figure shown via fig.show() / embedded without our
    # html_config still drops the logo); write_html additionally strips lasso/select via html_config().
    fig.update_layout(modebar=dict(remove=list(_MODEBAR_REMOVE)))

    _apply_line_traces(fig, spec)

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
            idx = r * cols + c + 1
            suffix = "" if idx == 1 else str(idx)
            tmpl = _key_template(panel)
            if tmpl is None:
                tmpl = "%{xaxis.title.text}=%{x}<br>%{yaxis.title.text}=%{y}<extra>%{fullData.name}</extra>"
            line_axes[f"x{suffix}"] = tmpl

    for tr in fig.data:
        if tr.type not in ("scatter", "scattergl"):
            continue
        if "lines" not in (tr.mode or ""):
            continue
        xax = getattr(tr, "xaxis", None) or "x"
        tmpl = line_axes.get(xax)
        if tmpl is not None and tr.hovertemplate is None and tr.hoverinfo != "skip":
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
