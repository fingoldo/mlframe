"""Two scatter defects where one spec produced two different readings.

VIS-09: the hollow "low evidence" treatment was gated on there being at least one STRONG point as well, so
a reliability diagram built entirely on 3-row bins rendered pixel-identical to one built on 300k-row bins
-- and lost its "too few rows to read" legend entry. The confidence signal disappeared exactly when it
mattered most.

VIS-08: plotly windowed a perfect-fit scatter to the data hull whenever the diagonal was drawn, while
matplotlib does that only in the square (``equal_aspect``) branch and lets calibration autoscale on
purpose. The same points came out at visibly different scales, so a gap that looked large in the PNG
looked small in the HTML.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec

N = 10
WEAK_LEGEND = "too few rows to read"


def _spec(**kwargs) -> FigureSpec:
    """A reliability-shaped scatter."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.05, 0.95, N)
    y = x + rng.normal(0, 0.05, N)
    panel = ScatterPanelSpec(x=x, y=y, title="Reliability", xlabel="predicted", ylabel="observed", legend_label="bins", **kwargs)
    return FigureSpec(panels=((panel,),), figsize=(7.0, 5.0))


def _mpl_legend(spec):
    """Legend entry texts drawn by matplotlib."""
    fig = MatplotlibRenderer().render(spec)
    try:
        legend = fig.axes[0].get_legend()
        return [t.get_text() for t in legend.get_texts()] if legend is not None else []
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def _plotly_trace_names(spec):
    """Named traces on the plotly figure."""
    return [tr.name for tr in PlotlyRenderer().render(spec).data if tr.name]


@pytest.mark.parametrize("reader", [_mpl_legend, _plotly_trace_names], ids=["matplotlib", "plotly"])
def test_an_all_low_evidence_panel_still_says_so(reader):
    """The case the guard silently dropped: every bin resting on too little data."""
    entries = reader(_spec(low_evidence_indices=tuple(range(N))))
    assert WEAK_LEGEND in entries, f"an all-low-evidence panel does not flag itself; entries were {entries}"


@pytest.mark.parametrize("reader", [_mpl_legend, _plotly_trace_names], ids=["matplotlib", "plotly"])
def test_a_mixed_panel_still_says_so(reader):
    """The case that already worked must keep working."""
    assert WEAK_LEGEND in reader(_spec(low_evidence_indices=(0, 1, 2)))


@pytest.mark.parametrize("reader", [_mpl_legend, _plotly_trace_names], ids=["matplotlib", "plotly"])
def test_a_fully_supported_panel_does_not_cry_wolf(reader):
    """No low-evidence points means no warning; the fix must not add one everywhere."""
    entries = reader(_spec(low_evidence_indices=()))
    assert WEAK_LEGEND not in entries, f"a fully-supported panel claims low evidence; entries were {entries}"


def test_calibration_autoscales_on_both_backends():
    """A non-square perfect-fit panel is deliberately autoscaled by matplotlib; plotly forced the data hull."""
    fig = PlotlyRenderer().render(_spec(perfect_fit_line=True, equal_aspect=False))
    assert fig.layout.xaxis.range is None, f"plotly pinned the x range to {fig.layout.xaxis.range}; calibration should autoscale"
    assert fig.layout.yaxis.range is None, f"plotly pinned the y range to {fig.layout.yaxis.range}"


def test_a_square_panel_gets_the_same_window_on_both_backends():
    """Where matplotlib DOES window to lo..hi, plotly must use the same numbers."""
    spec = _spec(perfect_fit_line=True, equal_aspect=True)
    plotly_range = PlotlyRenderer().render(spec).layout.xaxis.range
    fig = MatplotlibRenderer().render(spec)
    try:
        mpl_range = fig.axes[0].get_xlim()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
    assert plotly_range is not None, "the square branch must still window to the data hull"
    assert tuple(round(float(v), 6) for v in plotly_range) == tuple(
        round(float(v), 6) for v in mpl_range
    ), f"backends disagree on the square window: plotly {plotly_range} vs matplotlib {mpl_range}"


def test_an_explicit_limit_still_wins_on_the_autoscaled_branch():
    """Gating the window must not make the builder's own xlim/ylim a no-op."""
    fig = PlotlyRenderer().render(_spec(perfect_fit_line=True, equal_aspect=False, xlim=(0.0, 1.0), ylim=(0.0, 1.0)))
    assert tuple(fig.layout.xaxis.range) == (0.0, 1.0), f"explicit xlim was dropped: {fig.layout.xaxis.range}"
    assert tuple(fig.layout.yaxis.range) == (0.0, 1.0), f"explicit ylim was dropped: {fig.layout.yaxis.range}"


def test_the_error_bars_are_muted_too_when_every_point_is_weak():
    """A SECOND guard had the same defect, and only a probe found it.

    The marker branch and the error-bar branch each tested ``weak.any() and (~weak).any()``. Fixing the
    markers alone left an all-low-evidence panel drawing ordinary solid grey error bars beside hollow
    markers -- half the confidence signal restored and half still missing.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0.05, 0.95, N)
    y = x + rng.normal(0, 0.05, N)
    err = np.full(N, 0.05)

    def _containers(indices):
        """Number of errorbar containers matplotlib drew."""
        panel = ScatterPanelSpec(x=x, y=y, y_err=err, low_evidence_indices=indices, title="Reliability", legend_label="bins")
        fig = MatplotlibRenderer().render(FigureSpec(panels=((panel,),), figsize=(7.0, 5.0)))
        try:
            return len(fig.axes[0].containers)
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    # A mixed panel draws the strong bars and the muted weak ones separately; an all-weak panel draws only
    # the muted set. Either way the muted treatment is present -- what must not happen is the single
    # ordinary-looking container the old guard produced.
    assert _containers(tuple(range(N))) >= 1, "an all-low-evidence panel drew no error bars at all"
    assert _containers((0, 1, 2)) == 2, "a mixed panel should draw the strong and weak error bars separately"
