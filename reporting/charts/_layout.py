"""Shared layout helpers for multi-panel target-type figures.

Each multi-* compose function returns a FigureSpec whose panels grid is
built by ``pack_panels``: takes a flat list of PanelSpec instances and
arranges them into a 2-D grid with at most ``max_cols`` per row,
padding the last row with ``None`` cells when needed.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from mlframe.reporting.spec import PanelSpec


def pack_panels(
    panels: Sequence[PanelSpec],
    *,
    max_cols: int = 2,
) -> Tuple[Tuple[PanelSpec, ...], ...]:
    """Pack a flat list of panels into a row-major grid.

    Returns a tuple-of-tuples (rows × cols). Last row is padded with
    ``None`` so all rows are equal-width (the renderers skip ``None``
    cells).
    """
    if not panels:
        return ()
    rows: List[Tuple[PanelSpec, ...]] = []
    for i in range(0, len(panels), max_cols):
        chunk = list(panels[i : i + max_cols])
        # Pad last partial row with None to keep grid rectangular.
        while len(chunk) < max_cols:
            chunk.append(None)
        rows.append(tuple(chunk))
    return tuple(rows)


def figsize_for_grid(
    n_rows: int,
    n_cols: int,
    *,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
) -> Tuple[float, float]:
    """Compute a sensible figure size for the packed grid."""
    return (n_cols * cell_width, n_rows * cell_height)


def parse_panel_template(template: str) -> List[str]:
    """Parse a space-separated panel template into a list of upper-case tokens."""
    return [t.strip().upper() for t in template.split() if t.strip()]


__all__ = ["pack_panels", "figsize_for_grid", "parse_panel_template"]
