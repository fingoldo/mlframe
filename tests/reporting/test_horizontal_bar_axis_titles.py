"""A horizontal bar chart must name the same axis the same way on both backends.

``BarPanelSpec.xlabel`` names the VALUE and ``ylabel`` the CATEGORY regardless of orientation -- that is what
every horizontal-bar builder in ``charts/`` passes and what matplotlib draws. The plotly branch swapped them,
so six chart types labelled their value axis "subgroup" / "model" / "metric" in the HTML report while the PNG
built from the same spec read correctly. A report whose meaning depends on the backend is a defect on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import BarPanelSpec, FigureSpec

VALUE_LABEL = "ECE (lower = better calibrated)"
CATEGORY_LABEL = "subgroup"


def _horizontal_spec() -> FigureSpec:
    """The shape every horizontal-bar builder produces: value on xlabel, category on ylabel."""
    panel = BarPanelSpec(
        categories=("group_a", "group_b", "group_c"),
        values=np.array([0.12, 0.08, 0.19]),
        title="Calibration error by subgroup",
        xlabel=VALUE_LABEL,
        ylabel=CATEGORY_LABEL,
        orientation="horizontal",
    )
    return FigureSpec(panels=((panel,),), figsize=(8.0, 4.0))


def test_plotly_puts_the_value_label_on_the_value_axis():
    """The bars run along x, so x is the value axis and must carry the value label."""
    fig = PlotlyRenderer().render(_horizontal_spec())
    assert fig.layout.xaxis.title.text == VALUE_LABEL, f"value axis is labelled {fig.layout.xaxis.title.text!r}"
    assert fig.layout.yaxis.title.text == CATEGORY_LABEL, f"category axis is labelled {fig.layout.yaxis.title.text!r}"


def test_both_backends_label_the_value_axis_identically():
    """One spec, one reading. This is the assertion the swap actually broke."""
    spec = _horizontal_spec()
    plotly_fig = PlotlyRenderer().render(spec)
    mpl_fig = MatplotlibRenderer().render(spec)
    try:
        ax = mpl_fig.axes[0]
        assert (
            ax.get_xlabel() == plotly_fig.layout.xaxis.title.text
        ), f"value axis differs between backends: matplotlib {ax.get_xlabel()!r} vs plotly {plotly_fig.layout.xaxis.title.text!r}"
        assert (
            ax.get_ylabel() == plotly_fig.layout.yaxis.title.text
        ), f"category axis differs between backends: matplotlib {ax.get_ylabel()!r} vs plotly {plotly_fig.layout.yaxis.title.text!r}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(mpl_fig)


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param("fairness_calibration", id="fairness"),
        pytest.param("category_discriminability", id="category_discriminability"),
    ],
)
def test_a_real_chart_keeps_the_value_on_the_value_axis(builder):
    """Not just a hand-built spec: the charts the finding named, through their own builders."""
    rng = np.random.default_rng(0)
    n = 1500
    y = (rng.random(n) < 0.3).astype(int)

    if builder == "fairness_calibration":
        from mlframe.reporting.charts.fairness_calibration import compose_fairness_calibration_figure

        spec = compose_fairness_calibration_figure(y, np.clip(0.25 * y + rng.random(n) * 0.7, 0, 1), rng.integers(0, 4, n).astype(str))
    else:
        from mlframe.reporting.charts.category_discriminability import compose_category_discriminability_figure

        X = pd.DataFrame({f"cat_{i}": rng.integers(0, 8, n).astype(str) for i in range(4)})
        spec = compose_category_discriminability_figure(X, y, list(X.columns))

    bars = [pnl for row in spec.panels for pnl in row if isinstance(pnl, BarPanelSpec) and pnl.orientation == "horizontal"]
    if not bars:
        pytest.skip(f"{builder} produced no horizontal bar panel on this fixture")

    fig = PlotlyRenderer().render(spec)
    x_titles = [fig.layout[a].title.text for a in fig.layout if a.startswith("xaxis")]
    for panel in bars:
        assert panel.xlabel in x_titles, f"{builder}: value label {panel.xlabel!r} is not on any x axis; got {x_titles}"
