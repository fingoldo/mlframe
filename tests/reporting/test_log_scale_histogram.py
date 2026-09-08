"""A log-scaled histogram labelled one decade and silently dropped its empty bins.

``HistogramPanelSpec.yscale`` is public API. matplotlib installs a ``LogLocator`` with the 1/2/5
subdivisions because the stock locator labels only decades -- a histogram spanning a decade and a bit then
carries a scale name and no readable values. Plotly set ``type="log"`` and took the defaults.

Worse on both: ``log(0)`` is undefined, so an empty bin is not drawn small, it is not drawn at all. On a
tight cluster plus one far outlier that is 24 of 30 bars gone, with nothing saying the gaps are empty
rather than unplotted.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlframe.reporting.renderers._shared_helpers import log_axis_dropped_note
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, HistogramPanelSpec

NOTE = "log scale"


def _spec(yscale: str = "log", bins: int = 30) -> FigureSpec:
    """A tight cluster plus one far outlier: most bins are empty, which is the case that breaks."""
    values = np.concatenate([np.random.default_rng(0).normal(0.0, 0.2, 400), np.array([8.0])])
    panel = HistogramPanelSpec(values=values, bins=bins, title="latency", xlabel="ms", ylabel="count", yscale=yscale)
    return FigureSpec(panels=((panel,),), figsize=(7.0, 4.0))


def _mpl_texts(spec) -> list:
    """Free-standing texts matplotlib drew on the panel."""
    fig = MatplotlibRenderer().render(spec)
    try:
        return [t.get_text() for t in fig.axes[0].texts]
    finally:
        plt.close(fig)


def _plotly_texts(spec) -> list:
    """Annotation texts plotly drew on the panel."""
    return [a.text for a in PlotlyRenderer().render(spec).layout.annotations if a.text]


def test_the_helper_counts_only_the_undrawable_bins():
    """Only a log axis hides empty bins, and only the empty ones are hidden."""
    assert log_axis_dropped_note([1, 0, 3, 0], "log") == "log scale: 2 of 4 bins are empty and cannot be drawn"
    assert log_axis_dropped_note([1, 2, 3], "log") is None, "a histogram with no empty bins must not be annotated"
    assert log_axis_dropped_note([1, 0, 3], "linear") is None, "a linear axis draws the empty bins; there is nothing to report"


@pytest.mark.parametrize("reader", [_mpl_texts, _plotly_texts], ids=["matplotlib", "plotly"])
def test_both_backends_say_how_many_bars_the_log_axis_cannot_show(reader):
    """The defect: six bars drawn, twenty-four gone, nothing on the panel saying so."""
    notes = [t for t in reader(_spec()) if NOTE in t]
    assert len(notes) == 1, f"expected one notice, got {reader(_spec())}"
    assert "24 of 30" in notes[0], f"the notice does not name the counts: {notes[0]}"


@pytest.mark.parametrize("reader", [_mpl_texts, _plotly_texts], ids=["matplotlib", "plotly"])
def test_a_linear_axis_is_not_annotated(reader):
    """Guard: on a linear axis the empty bins ARE drawn, so the notice would be noise."""
    assert not [t for t in reader(_spec(yscale="linear")) if NOTE in t]


def test_the_fixture_really_does_lose_most_of_its_bars():
    """Without this the notice tests could pass against a histogram that has nothing to hide."""
    values = np.concatenate([np.random.default_rng(0).normal(0.0, 0.2, 400), np.array([8.0])])
    counts, _ = np.histogram(values, bins=30)
    assert int(np.count_nonzero(counts == 0)) == 24


def test_plotly_subdivides_the_log_decades():
    """ "D2" is plotly's name for the 1/2/5 subdivisions the matplotlib LogLocator asks for by subs=(1, 2, 5)."""
    axis = PlotlyRenderer().render(_spec()).layout.yaxis
    assert axis.type == "log", "the axis is not log-scaled at all"
    assert axis.dtick == "D2", f"the log axis takes plotly's decade-only default (dtick={axis.dtick!r})"


def test_plotly_asks_for_a_readable_linear_tick_count():
    """The same complaint on the other branch: a short panel gets two or three labels by default."""
    axis = PlotlyRenderer().render(_spec(yscale="linear")).layout.yaxis
    assert axis.type == "linear" and axis.nticks == 6, f"linear ticks are unconfigured (nticks={axis.nticks!r})"


def test_matplotlib_labels_more_than_one_decade():
    """The defect in its original form: a scale name and no readable values."""
    fig = MatplotlibRenderer().render(_spec())
    try:
        fig.canvas.draw()
        labelled = [t.get_text() for t in fig.axes[0].get_yticklabels() if t.get_text()]
    finally:
        plt.close(fig)
    assert len(labelled) >= 3, f"only {len(labelled)} y labels on a log axis: {labelled}"
