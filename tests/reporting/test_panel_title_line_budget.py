"""A statistic-bearing panel title has a line budget (reporting audit 2026-09-06, WALK-05).

The two titles the walkthrough named were cut by hand; nothing stopped the next one from wrapping onto four lines and into the data. The renderer now fits a title to at most ``TITLE_MAX_LINES`` lines by
stepping the font down to ``TITLE_MIN_FONTSIZE`` (no word is cut, every statistic stays), and logs a warning when even the smallest font needs more lines.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlframe.reporting.renderers._shared_helpers import TITLE_MAX_LINES, TITLE_MIN_FONTSIZE, fit_title_lines
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec

_SHORT = "Reliability (ECE=0.031)"
# a title of the length the audit complained about: five statistics, each with a name and a number
_STATS = "Interval width by decile: mean=0.412, median=0.388, p90=0.771, coverage=0.903 (target 0.900), n=12,480, ECE=0.0312, MCE=0.0871, Brier=0.1243, AUC=0.8127"


def test_a_short_title_keeps_its_size_and_one_line():
    """A short title is not touched by the budget."""
    lines, size = fit_title_lines(_SHORT, fontsize=10, width_in=6.0)
    assert len(lines) == 1 and size == 10


def test_a_long_title_is_brought_within_the_budget_by_the_font_without_losing_a_word():
    """On a narrow panel the long statistic title needs more than the budget at 10 pt; it fits after stepping the font down, with every statistic still in the text."""
    width_in = 4.4
    from mlframe.reporting.renderers._shared_helpers import wrap_text_to_width

    assert len(wrap_text_to_width(_STATS, fontsize=10, width_in=width_in)) > TITLE_MAX_LINES, "premise: at 10 pt this title is over the budget"
    lines, size = fit_title_lines(_STATS, fontsize=10, width_in=width_in)
    assert len(lines) <= TITLE_MAX_LINES or size == TITLE_MIN_FONTSIZE
    assert size <= 10 and size >= TITLE_MIN_FONTSIZE
    joined = " ".join(lines)
    for token in ("mean=0.412", "p90=0.771", "coverage=0.903", "n=12,480", "ECE=0.0312", "AUC=0.8127"):
        assert token in joined, f"{token} was lost: {lines}"


def test_a_title_that_cannot_fit_is_returned_whole_at_the_smallest_font():
    """Past the floor the text is wrapped as is (no cut) and the caller sees a line count above the budget."""
    absurd = " ".join(f"statistic_{i}=0.{i:03d}" for i in range(80))
    lines, size = fit_title_lines(absurd, fontsize=10, width_in=3.0)
    assert size == TITLE_MIN_FONTSIZE and len(lines) > TITLE_MAX_LINES
    assert all(f"statistic_{i}=" in " ".join(lines) for i in (0, 40, 79))


def test_the_renderer_applies_the_budget_and_warns_when_it_cannot_be_met(caplog):
    """A figure whose title cannot fit logs one warning that names it; a fitting title logs none."""
    x = np.linspace(0, 1, 20)
    long_title = " ".join(f"statistic_{i}=0.{i:03d}" for i in range(120))
    spec = FigureSpec(panels=((ScatterPanelSpec(x=x, y=x, title=long_title),),), figsize=(4.0, 3.0))
    with caplog.at_level(logging.WARNING, logger="mlframe.reporting.renderers.matplotlib"):
        fig = MatplotlibRenderer().render(spec)
    assert any("panel title needs" in r.getMessage() for r in caplog.records)
    plt.close(fig)
    caplog.clear()
    ok = FigureSpec(panels=((ScatterPanelSpec(x=x, y=x, title=_SHORT),),), figsize=(6.0, 3.0))
    with caplog.at_level(logging.WARNING, logger="mlframe.reporting.renderers.matplotlib"):
        fig = MatplotlibRenderer().render(ok)
    assert not any("panel title needs" in r.getMessage() for r in caplog.records)
    plt.close(fig)
