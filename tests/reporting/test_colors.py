"""Tests for the shared reporting palette (mlframe.reporting.colors)."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.colors import LINE_PALETTE, auto_text_color, auto_text_colors_batch, line_color, line_style

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


def test_twenty_series_stay_distinguishable():
    """INV-29's real contract: 20 classes must be tellable apart. The tab20 extension satisfied it on paper only --
    a tab10 hue and its tab20 lightness twin separate by 2.8 under a deuteranopia simulation, against 14.6 for the
    worst tab10 pair, so half those "distinct" colors were indistinguishable to a red-green-deficient reader.
    The separation now comes from the (color, dash) PAIR, which survives the simulation."""
    keys = [(line_color(i), line_style(i)) for i in range(20)]
    assert len(set(keys)) == 20
    assert len(set(LINE_PALETTE)) == len(LINE_PALETTE), "palette colors must be distinct"


def test_first_ten_unchanged_for_snapshot_backcompat():
    """The original tab10 prefix must stay byte-stable so existing snapshots of <=10-class charts don't shift."""
    assert LINE_PALETTE[:10] == _TAB10


def test_line_color_does_not_collide_until_palette_exhausted():
    """Classes 0..len-1 each get a unique color, and the wrap past the palette changes the dash instead."""
    colors = [line_color(i) for i in range(len(LINE_PALETTE))]
    assert len(set(colors)) == len(LINE_PALETTE)
    # The 11th class reuses the 1st color by design now; the dash is what separates them.
    assert line_color(len(LINE_PALETTE)) == line_color(0)
    assert line_style(len(LINE_PALETTE)) != line_style(0)


def test_auto_text_colors_batch_matches_per_cell_auto_text_color():
    """Regression: ``auto_text_colors_batch`` (one vectorized colormap sample for a whole grid) must be
    bit-identical to calling ``auto_text_color`` once per cell -- the invariant the plotly heatmap renderer's
    per-cell-annotation loop now relies on after switching from a per-cell to a batched call."""
    rng = np.random.default_rng(0)
    for colormap in ("viridis", "RdYlBu", "RdBu_r"):
        mat = rng.uniform(-2, 2, size=(20, 15))
        vmin, vmax = -1.5, 1.5
        filled = np.where(np.isfinite(mat), mat, vmin)
        scalar = np.array([[auto_text_color(filled[i, j], colormap, vmin=vmin, vmax=vmax) for j in range(mat.shape[1])] for i in range(mat.shape[0])])
        batch = auto_text_colors_batch(filled, colormap, vmin=vmin, vmax=vmax)
        assert np.array_equal(scalar, batch), f"batched vs per-cell mismatch for colormap={colormap!r}"


def test_auto_text_colors_batch_unknown_colormap_falls_back_to_black():
    """Mirrors ``auto_text_color``'s fallback: an unresolvable colormap name returns 'black' for every cell,
    never raising into the caller's render loop."""
    mat = np.array([[0.1, 0.9], [0.5, 0.3]])
    out = auto_text_colors_batch(mat, "not_a_real_colormap_xyz", vmin=0.0, vmax=1.0)
    assert (out == "black").all()


def test_line_color_cycles_after_palette():
    """Line color cycles after palette."""
    assert line_color(len(LINE_PALETTE)) == line_color(0)


def test_auto_text_color_logs_on_colormap_lookup_failure(caplog):
    """A matplotlib colormap lookup failure must be logged (not a silent except-and-'black')."""
    import logging

    from mlframe.reporting.colors import auto_text_color

    with caplog.at_level(logging.DEBUG, logger="mlframe.reporting.colors"):
        out = auto_text_color(0.5, "not_a_real_colormap_name")
    assert out == "black"
    assert any("colormap lookup" in rec.message for rec in caplog.records)
