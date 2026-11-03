"""The decile table caption names the width of the top bin that was actually built."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts._binary_decile_table import binary_decile_table_figure


def _caption(fig) -> str:
    """The figure's caption text."""
    return " ".join(t.get_text() for t in fig.texts)


def test_caption_says_top_ten_percent_for_ten_deciles():
    """With the default ten deciles the caption keeps the familiar 10%."""
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.4).astype(int)
    assert "top-scored 10%" in _caption(binary_decile_table_figure(y, rng.random(200)))


def test_caption_follows_fewer_bins_than_ten():
    """Six finite rows build six bins, so the top group is a sixth (16.7%) of the rows, not 10%."""
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2])
    cap = _caption(binary_decile_table_figure(y, s))
    assert "top-scored 16.6667%" in cap and "10%" not in cap
