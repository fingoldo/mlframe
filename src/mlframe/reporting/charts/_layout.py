"""Shared layout helpers for multi-panel target-type figures.

Each multi-* compose function returns a FigureSpec whose panels grid is
built by ``pack_panels``: takes a flat list of PanelSpec instances and
arranges them into a 2-D grid with at most ``max_cols`` per row,
padding the last row with ``None`` cells when needed.
"""

from __future__ import annotations

import os
from typing import Optional, List, Sequence, Tuple

from mlframe.reporting.spec import PanelSpec


def pack_panels(
    panels: Sequence[PanelSpec],
    *,
    max_cols: int = 2,
) -> Tuple[Tuple[Optional[PanelSpec], ...], ...]:
    """Pack a flat list of panels into a row-major grid.

    Returns a tuple-of-tuples (rows x cols). The last row is padded with ``None`` so every row is equal-width, which
    is why the element type is Optional: the renderers skip ``None`` cells. The annotation used to claim
    non-optional panels while the function padded with ``None``, so mypy could not see the very case every renderer
    guards against.
    """
    if not panels:
        return ()
    rows: List[Tuple[Optional[PanelSpec], ...]] = []
    for i in range(0, len(panels), max_cols):
        chunk: List[Optional[PanelSpec]] = list(panels[i : i + max_cols])
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


def base_for(plot_file: str, suffix: str) -> str:
    """Compose a per-panel sibling path: insert ``_<suffix>`` before ``plot_file``'s extension so each panel writes a distinct file."""
    root, ext = os.path.splitext(plot_file)
    return f"{root}_{suffix}{ext}"


__all__ = ["pack_panels", "figsize_for_grid", "parse_panel_template", "base_for"]
