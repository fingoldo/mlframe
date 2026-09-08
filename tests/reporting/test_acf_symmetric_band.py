"""A band labelled "+-x" must be drawn on both sides of zero.

The ACF and PACF panels count significant lags with ``|acf| > band`` and label the reference line
``+-1.96/sqrt(n)``, but drew only the POSITIVE bound. On a series whose structure is in the negative lags --
an AR(1) with a negative coefficient, where the strongest lag is around -0.6 -- the title claimed twelve
significant lags while the reader had no line to judge any of them against.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.charts.temporal import compose_target_acf_figure
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import BarPanelSpec


def _negative_ar1(n: int = 2000, phi: float = -0.6, seed: int = 0) -> np.ndarray:
    """AR(1) with a negative coefficient: odd lags are strongly NEGATIVE."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = phi * y[i - 1] + rng.normal()
    return y


def _acf_panels(spec):
    """The bar panels carrying a reference band."""
    return [p for row in spec.panels for p in row if isinstance(p, BarPanelSpec) and p.hline is not None]


def test_the_acf_band_is_declared_symmetric():
    """The spec has to say so; a renderer cannot infer it from a label that merely reads "+-"."""
    panels = _acf_panels(compose_target_acf_figure(_negative_ar1()))
    assert panels, "no ACF/PACF panel carries a reference line"
    for panel in panels:
        assert panel.hline_symmetric, f"{panel.title!r} labels its band '+-' but declares only one bound"


def test_the_fixture_actually_needs_the_lower_bound():
    """Guard: on a series with no negative lags past the band, this whole test proves nothing."""
    panels = _acf_panels(compose_target_acf_figure(_negative_ar1()))
    band = abs(panels[0].hline[0])
    values = np.asarray(panels[0].values, dtype=float)
    assert (values < -band).any(), "fixture has no lag below the negative bound, so the missing line would be invisible here"


def test_matplotlib_draws_both_bounds_but_lists_the_band_once():
    """Two lines, one legend entry -- a key naming the same threshold twice is its own noise."""
    spec = compose_target_acf_figure(_negative_ar1())
    fig = MatplotlibRenderer().render(spec)
    try:
        panel = _acf_panels(spec)[0]
        band = abs(panel.hline[0])
        ax = fig.axes[0]
        ys = sorted(round(float(ln.get_ydata()[0]), 6) for ln in ax.get_lines() if len(set(np.asarray(ln.get_ydata(), dtype=float))) == 1)
        assert round(-band, 6) in ys, f"the lower bound {-band:.6f} was not drawn; horizontal lines at {ys}"
        assert round(band, 6) in ys, f"the upper bound {band:.6f} was not drawn; horizontal lines at {ys}"
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()] if legend is not None else []
        band_entries = [lbl for lbl in labels if "1.96" in lbl]
        assert len(band_entries) == 1, f"the band is listed {len(band_entries)} times in the legend: {labels}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plotly_draws_both_bounds_too():
    """Same spec, same reading: a bound present on one backend only is a defect of its own."""
    spec = compose_target_acf_figure(_negative_ar1())
    panel = _acf_panels(spec)[0]
    band = abs(panel.hline[0])
    fig = PlotlyRenderer().render(spec)
    ys = sorted(round(float(sh.y0), 6) for sh in fig.layout.shapes if getattr(sh, "y0", None) is not None and sh.y0 == sh.y1)
    assert round(band, 6) in ys and round(-band, 6) in ys, f"plotly drew bounds at {ys}, expected both {band:.6f} and {-band:.6f}"


@pytest.mark.parametrize("symmetric", [False, True])
def test_the_flag_is_what_decides_it(symmetric):
    """Directly on the renderer: the flag, not the label text, controls the second line."""
    panel = BarPanelSpec(
        categories=("1", "2", "3"),
        values=np.array([0.5, -0.4, 0.1]),
        hline=(0.25, "red", "+-0.25"),
        hline_symmetric=symmetric,
    )
    from mlframe.reporting.spec import FigureSpec

    fig = MatplotlibRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0)))
    try:
        ax = fig.axes[0]
        ys = {round(float(ln.get_ydata()[0]), 6) for ln in ax.get_lines() if len(set(np.asarray(ln.get_ydata(), dtype=float))) == 1}
        assert (round(-0.25, 6) in ys) is symmetric, f"symmetric={symmetric} but the lower bound presence is {round(-0.25, 6) in ys}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
