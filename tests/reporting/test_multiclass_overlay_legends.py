"""A per-class overlay legend must not need most of the panel.

Each entry carries a class name AND its metric with a confidence interval; at twelve real class names that
is fourteen rows of ~45 characters, wide enough to cover the region where the curves separate. The width is
attacked at the source -- the class name is shortened, keeping the tail that distinguishes generated names
-- rather than by pushing the legend outside the axes: measured on a twelve-class figure, moving these three
legends out shrank every panel to about a third of its row, which trades one unreadable chart for another.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.multiclass import _LEGEND_CLASS_NAME_MAXLEN, compose_multiclass_figure

K = 12
CLASS_STEM = "electronics_accessories_tier"
MAX_ENTRY_CHARS = 46


def _figure(panels: str = "ROC PR_CURVES CALIB_GRID"):
    """Twelve classes with realistic string names, the shape the finding names."""
    rng = np.random.default_rng(0)
    n = 4000
    classes = [f"{CLASS_STEM}_{i}" for i in range(K)]
    idx = rng.integers(0, K, n)
    logit = rng.normal(size=(n, K))
    logit[np.arange(n), idx] += 1.5
    proba = np.exp(logit)
    proba /= proba.sum(1, keepdims=True)
    return compose_multiclass_figure(np.array([classes[i] for i in idx]), proba, classes, panels_template=panels)


def _labelled_panels(spec):
    """Panels carrying a legend."""
    return [p for row in spec.panels for p in row if p is not None and getattr(p, "series_labels", None)]


def test_the_fixture_would_overflow_without_shortening():
    """Guard: with names this short the test proves nothing."""
    assert len(f"{CLASS_STEM}_0") > _LEGEND_CLASS_NAME_MAXLEN, "class names no longer exceed the cap"


def test_no_legend_entry_is_longer_than_a_panel_can_hold():
    """The whole point: entry width, not legend placement."""
    for panel in _labelled_panels(_figure()):
        longest = max(len(lbl) for lbl in panel.series_labels)
        assert longest <= MAX_ENTRY_CHARS, f"{panel.title!r} has a {longest}-char legend entry: {max(panel.series_labels, key=len)!r}"


def test_the_shortened_name_keeps_the_part_that_distinguishes_it():
    """`tier_0` vs `tier_11` is the whole difference; a head-preserving cut would erase it."""
    labels = [lbl for panel in _labelled_panels(_figure()) for lbl in panel.series_labels]
    class_entries = [lbl for lbl in labels if "..." in lbl]
    assert class_entries, "nothing was shortened, so this test is vacuous"
    tails = {lbl.split("...")[1].split(" ")[0] for lbl in class_entries}
    assert len(tails) > 1, f"every shortened entry ends the same way, so the classes are indistinguishable: {sorted(tails)}"


def test_a_crowded_legend_is_split_into_columns():
    """Fourteen rows at 8pt is taller than most panels; two columns halves that."""
    crowded = [p for p in _labelled_panels(_figure()) if len(p.series_labels) > 10]
    # The precondition is asserted, not assumed: a fixture that stops producing a crowded panel would
    # otherwise leave this test passing while checking nothing at all.
    assert crowded, "the fixture no longer builds a panel with more than ten legend entries"
    for panel in crowded:
        assert getattr(panel, "legend_ncol", 1) == 2, f"{panel.title!r} has {len(panel.series_labels)} entries in one column"


@pytest.mark.parametrize("panels", ["ROC", "PR_CURVES", "CALIB_GRID"])
def test_each_overlay_panel_individually(panels):
    """Per panel, so a regression names the panel that broke."""
    panel = _labelled_panels(_figure(panels))[0]
    longest = max(len(lbl) for lbl in panel.series_labels)
    assert longest <= MAX_ENTRY_CHARS, f"{panels}: {longest}-char entry"
