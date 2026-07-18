"""Tests for the shared reporting palette (mlframe.reporting.colors)."""

from __future__ import annotations

from mlframe.reporting.colors import LINE_PALETTE, line_color

_TAB10 = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def test_palette_extended_to_tab20():
    """INV-29: at K>=11 classes collide on a 10-color palette; the palette must hold >=20 distinct colors so a default
    20-class line plot (per-class ROC etc.) never reuses a color before cycling."""
    assert len(LINE_PALETTE) >= 20
    assert len(set(LINE_PALETTE)) == len(LINE_PALETTE), "palette colors must be distinct"


def test_first_ten_unchanged_for_snapshot_backcompat():
    """The original tab10 prefix must stay byte-stable so existing snapshots of <=10-class charts don't shift."""
    assert LINE_PALETTE[:10] == _TAB10


def test_line_color_does_not_collide_until_palette_exhausted():
    """Classes 0..len-1 each get a unique color (no early collision the way a 10-color palette had at idx 10)."""
    colors = [line_color(i) for i in range(len(LINE_PALETTE))]
    assert len(set(colors)) == len(LINE_PALETTE)
    # 11th class differs from the 1st (the exact defect a tab10-only palette had).
    assert line_color(10) != line_color(0)


def test_line_color_cycles_after_palette():
    """Line color cycles after palette."""
    assert line_color(len(LINE_PALETTE)) == line_color(0)
