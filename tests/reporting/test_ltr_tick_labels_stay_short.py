"""The category tick labels of the LTR panels stay short, so a rotated label cannot run into the panel below (reporting audit 2026-09-06, WALK-01).

The walkthrough found ``8-15 (n=300, CI[0.92,0.94])`` (24 characters, rotated) on NDCG_BY_QSIZE and ``all queries (n=300)`` on NDCG_DIST overprinting the axis title below them. The tick text now carries
only the bin or the support; the interval is in the bar's error bar and the panel title.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlframe.reporting.charts.ltr import compose_ltr_figure
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

_MAX_TICK_CHARS = 14


def _figure(n_queries: int = 300, docs: int = 10):
    """The audit's scenario: one query-size bin, ``n_queries`` queries of ``docs`` documents."""
    rng = np.random.default_rng(0)
    gid = np.repeat(np.arange(n_queries), docs)
    rel = rng.integers(0, 4, n_queries * docs)
    score = rel + rng.normal(0.0, 1.0, n_queries * docs)
    spec = compose_ltr_figure(rel, score, gid, panels_template="NDCG_DIST NDCG_BY_QSIZE", max_cols=1)
    return MatplotlibRenderer().render(spec)


def test_the_category_tick_labels_are_short_and_carry_no_interval():
    """Every x tick label of the two panels is at most ``_MAX_TICK_CHARS`` characters and none of them holds a confidence interval."""
    fig = _figure()
    labels = [t.get_text() for ax in fig.axes for t in ax.get_xticklabels() if t.get_text()]
    assert labels, "premise: the panels have category tick labels"
    too_long = [lbl for lbl in labels if len(lbl) > _MAX_TICK_CHARS]
    assert not too_long, f"tick labels over {_MAX_TICK_CHARS} characters: {too_long}"
    assert not any("CI[" in lbl for lbl in labels), labels
    plt.close(fig)


def test_the_support_is_still_readable_from_the_labels():
    """The shortening did not drop the information: the support of the distribution panel and the size bin of the second panel are present."""
    fig = _figure()
    labels = [t.get_text() for ax in fig.axes for t in ax.get_xticklabels() if t.get_text()]
    assert any("300" in lbl for lbl in labels), labels
    assert any("-" in lbl and lbl[0].isdigit() for lbl in labels), labels
    plt.close(fig)
