"""An all-non-finite histogram must say so on both backends, not render an empty framed panel.

Matplotlib drops non-finite values before binning and writes "no finite values" when nothing survives.
Plotly handed the raw array straight to ``go.Histogram``, which draws nothing and reports nothing, so the
same spec read as "this diagnostic was not computed" on one backend and "this column has no usable data"
on the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, HistogramPanelSpec

NOTICE = "no finite values"


def _figure(values):
    """One-panel histogram spec over ``values``."""
    return FigureSpec(panels=((HistogramPanelSpec(values=np.asarray(values, dtype=float), bins=10, title="col"),),), figsize=(6.0, 4.0))


def _plotly_notes(fig):
    """Every non-empty annotation text on a plotly figure."""
    return [a.text for a in fig.layout.annotations if a.text]


def test_plotly_says_no_finite_values_when_nothing_survives():
    """The note matplotlib has always drawn, on the backend that drew an empty frame instead."""
    fig = PlotlyRenderer().render(_figure([np.nan, np.nan, np.inf, -np.inf]))
    assert any(NOTICE in t for t in _plotly_notes(fig)), f"plotly drew no notice; annotations were {_plotly_notes(fig)}"
    assert not [t for t in fig.data if t.type == "histogram"], "plotly still binned an all-non-finite column"


def test_the_notice_is_anchored_to_its_own_panel():
    """A subplot cell with no trace is never laid out, so its annotation lands on a neighbour instead.

    Two panels side by side, only the second all-non-finite: the notice must reference the second panel's
    axes AND that panel must actually exist in the layout, or the reader sees "no finite values" written
    across a perfectly good histogram.
    """
    good = np.asarray([1.0, 2.0, 3.0, 4.0])
    spec = FigureSpec(
        panels=((HistogramPanelSpec(values=good, bins=4, title="finite"), HistogramPanelSpec(values=np.asarray([np.nan, np.nan]), bins=4, title="empty")),),
        figsize=(10.0, 4.0),
    )
    fig = PlotlyRenderer().render(spec)
    notice = [a for a in fig.layout.annotations if a.text and NOTICE in a.text]
    assert len(notice) == 1, f"expected one notice, got {[a.text for a in fig.layout.annotations]}"
    assert notice[0].xref.startswith("x2"), f"the notice is anchored to {notice[0].xref!r}, not the empty panel's own axis"
    on_second = [t for t in fig.data if getattr(t, "xaxis", None) == "x2"]
    assert on_second, "the empty panel holds no trace at all, so plotly never lays its axes out and the notice drifts"


def test_matplotlib_says_the_same_thing():
    """The reference behaviour the plotly path had to match."""
    fig = MatplotlibRenderer().render(_figure([np.nan, np.nan, np.inf, -np.inf]))
    try:
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any(NOTICE in t for t in texts), f"matplotlib drew no notice; texts were {texts}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.parametrize("renderer", [MatplotlibRenderer, PlotlyRenderer])
def test_a_partly_finite_column_still_bins_the_finite_part(renderer):
    """Guard: the filter must drop the non-finite rows, not the whole panel."""
    import matplotlib.pyplot as plt

    spec = _figure([1.0, 2.0, np.nan, 3.0, np.inf])
    fig = renderer().render(spec)
    try:
        # Collected per backend, asserted for both: behind an ``if`` the stronger failure -- a panel that was
        # never drawn at all -- would skip the check instead of failing it.
        notes = _plotly_notes(fig) if renderer is PlotlyRenderer else [t.get_text() for t in fig.axes[0].texts]
        assert not any(NOTICE in t for t in notes), f"{renderer.__name__} claimed no finite values on a column with three of them"
        if renderer is PlotlyRenderer:
            assert len(fig.data) == 1 and len(fig.data[0].x) == 3, f"plotly binned {getattr(fig.data[0], 'x', None)!r} instead of the three finite values"
    finally:
        if renderer is not PlotlyRenderer:
            plt.close(fig)
