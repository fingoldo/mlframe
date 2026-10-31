"""Legend placement for matplotlib panels: an in-axes legend, promoted beside the panel only where that space exists."""

from __future__ import annotations

_LEGEND_MAX_PANEL_FRACTION = 0.62  # widest entry may claim at most this share of the panel
_LEGEND_MAX_INSIDE_ENTRIES = 8  # beyond this the stack is taller than most panels regardless of width
_LEGEND_CHAR_WIDTH_RATIO = 0.6  # mean glyph advance as a fraction of font size, DejaVu Sans at these sizes
_LEGEND_HANDLE_INCHES = 0.45  # marker/line sample plus padding, left of the text


def _legend_overflows(ax, labels, *, fontsize: int = 8) -> bool:
    """True when this legend cannot sit inside ``ax`` without covering a meaningful part of it."""
    texts = [str(lbl) for lbl in labels if lbl]
    if not texts:
        return False
    if len(texts) > _LEGEND_MAX_INSIDE_ENTRIES:
        return True
    fig_w_in = float(ax.figure.get_size_inches()[0])
    panel_w_in = max(0.5, float(ax.get_position().width) * fig_w_in)
    widest_in = (max(len(t) for t in texts) * _LEGEND_CHAR_WIDTH_RATIO * fontsize) / 72.0 + _LEGEND_HANDLE_INCHES
    return widest_in > _LEGEND_MAX_PANEL_FRACTION * panel_w_in


def _panel_is_rightmost(ax) -> bool:
    """True when nothing in the figure is drawn to the right of ``ax``."""
    x1 = float(ax.get_position().x1)
    return all(float(other.get_position().x1) <= x1 + 1e-3 for other in ax.figure.axes)


def _place_legend(ax, handles=None, labels=None, *, force_outside: bool = False, ncol: int = 1, has_secondary_y: bool = False):
    """Draw the legend inside the panel, or beside it when it would not fit and there is room beside it.

    "Outside" is only an improvement where the space actually exists. Anchoring past the right edge of an INNER
    grid column parks the legend in the narrow gutter between panels, where it overlaps its neighbour; doing it
    on a panel with a right-hand axis lands it straight on that axis's label. In both cases an in-axes legend at
    a smaller size, over a more opaque box, is the lesser harm -- so the automatic promotion is gated on the
    panel being the rightmost one and having no secondary axis. An explicit ``legend_outside`` from the builder
    is still honoured unconditionally: the builder knows its own layout.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    overflows = _legend_overflows(ax, labels)
    if force_outside or (overflows and _panel_is_rightmost(ax) and not has_secondary_y):
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.7, ncol=max(1, int(ncol)))
    else:
        ax.legend(handles, labels, loc="best", fontsize=7 if overflows else 8, framealpha=0.85 if overflows else 0.7, ncol=max(1, int(ncol)))
