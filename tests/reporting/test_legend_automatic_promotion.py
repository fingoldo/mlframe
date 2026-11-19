"""The automatic legend placement of the matplotlib panels is pinned (reporting audit 2026-09-06, WALK-03).

A legend that cannot fit inside its panel is moved beside it, but only where that space exists: on the rightmost panel and not on one with a right-hand axis. Everywhere else the legend stays inside at a
smaller size over a more opaque box. An explicit ``force_outside`` is honoured unconditionally.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlframe.reporting.renderers._matplotlib_legend import _legend_overflows, _panel_is_rightmost, _place_legend

_LONG = [f"label_{i}_long_descriptive_name (ECE=0.203, n=3,000)" for i in range(6)]
_SHORT = ["a", "b"]


def _axes(n_cols: int = 1, width: float = 6.0):
    """A one-row figure with ``n_cols`` panels and one line per label added by the caller."""
    fig, axes = plt.subplots(1, n_cols, figsize=(width, 3.0))
    return fig, np.atleast_1d(axes)


def _draw(ax, labels):
    """One line per label so the axes carry legend handles."""
    for i, lbl in enumerate(labels):
        ax.plot([0, 1], [i, i + 1], label=lbl)


def _outside(ax) -> bool:
    """True when the legend of ``ax`` is anchored past the right edge of the axes."""
    leg = ax.get_legend()
    return leg is not None and leg._bbox_to_anchor is not None and leg._bbox_to_anchor.bounds[0] > 1.0


def test_short_entries_do_not_overflow_and_stay_inside():
    """Two short entries fit: not an overflow, the legend is inside at the normal size."""
    fig, (ax,) = _axes()
    _draw(ax, _SHORT)
    assert not _legend_overflows(ax, _SHORT)
    _place_legend(ax)
    assert not _outside(ax) and ax.get_legend().get_texts()[0].get_fontsize() == 8
    plt.close(fig)


def test_long_entries_on_the_rightmost_panel_move_outside():
    """Six long entries overflow a narrow panel; the single (rightmost) panel gets the legend beside it."""
    fig, (ax,) = _axes(width=4.0)
    _draw(ax, _LONG)
    assert _legend_overflows(ax, _LONG) and _panel_is_rightmost(ax)
    _place_legend(ax)
    assert _outside(ax)
    plt.close(fig)


def test_long_entries_on_an_inner_panel_stay_inside_and_shrink():
    """On the left panel of two, outside would land in the gutter and overlap the neighbour: inside, smaller font, more opaque box."""
    fig, axes = _axes(n_cols=2, width=5.0)
    _draw(axes[0], _LONG)
    assert _legend_overflows(axes[0], _LONG) and not _panel_is_rightmost(axes[0])
    _place_legend(axes[0])
    leg = axes[0].get_legend()
    assert not _outside(axes[0]) and leg.get_texts()[0].get_fontsize() == 7 and leg.get_frame().get_alpha() == pytest.approx(0.85)
    plt.close(fig)


def test_a_panel_with_a_right_hand_axis_keeps_the_legend_inside():
    """The promotion is off when the panel has a secondary y axis, whose label would sit under the legend."""
    fig, (ax,) = _axes(width=4.0)
    _draw(ax, _LONG)
    _place_legend(ax, has_secondary_y=True)
    assert not _outside(ax)
    plt.close(fig)


def test_more_than_eight_entries_overflow_whatever_the_width():
    """The entry-count rule: a stack of nine short entries is taller than most panels."""
    fig, (ax,) = _axes(width=12.0)
    labels = [f"m{i}" for i in range(9)]
    _draw(ax, labels)
    assert _legend_overflows(ax, labels)
    plt.close(fig)


def test_force_outside_is_honoured_on_an_inner_panel():
    """A builder's explicit ``legend_outside`` wins over the rightmost-panel gate."""
    fig, axes = _axes(n_cols=2, width=8.0)
    _draw(axes[0], _SHORT)
    _place_legend(axes[0], force_outside=True)
    assert _outside(axes[0])
    plt.close(fig)
