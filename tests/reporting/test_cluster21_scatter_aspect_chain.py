"""Cluster 21 regression: the equal-aspect flag is honoured with and without the perfect-fit diagonal (audit 2026-11-20, finding 21-F23).

The flag used to be read only inside the perfect-fit branch, so asking for a square panel without that diagonal did nothing. With the flag on,
the panel is square whether or not the diagonal is drawn. With it off, the aspect is left alone.
"""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mlframe.reporting.renderers import _matplotlib_scatter as ms


def _draw(*, perfect_fit_line, equal_aspect, xlim=None, ylim=None):
    """Draw the perfect-fit overlay for a small cloud and return the axes it was drawn on."""
    fig, ax = plt.subplots()
    p = SimpleNamespace(perfect_fit_line=perfect_fit_line, equal_aspect=equal_aspect, xlim=xlim, ylim=ylim)
    x = np.linspace(0.0, 1.0, 20)
    y = x * 0.9
    try:
        ms._draw_perfect_fit_line(p, x.size, x, y, ax)
        return ax, ax.get_aspect()
    finally:
        plt.close(fig)


def test_equal_aspect_is_applied_without_the_perfect_fit_diagonal():
    """With the flag on and no diagonal, the panel is still squared (21-F23)."""
    _ax, aspect = _draw(perfect_fit_line=False, equal_aspect=True)
    assert aspect == 1.0


def test_equal_aspect_with_diagonal_squares_the_panel_and_fixes_its_limits():
    """With the flag on and the diagonal drawn, the panel is squared and spans the union of both axes (21-F23)."""
    ax, aspect = _draw(perfect_fit_line=True, equal_aspect=True)
    assert aspect == 1.0
    lo, hi = ax.get_xlim()
    assert lo <= 0.0 and hi >= 1.0


def test_flag_off_leaves_the_aspect_alone():
    """Control: with equal_aspect off the aspect is not changed, with or without the diagonal (21-F23)."""
    _ax, aspect = _draw(perfect_fit_line=True, equal_aspect=False)
    assert aspect == "auto"
    _ax, aspect = _draw(perfect_fit_line=False, equal_aspect=False)
    assert aspect == "auto"
