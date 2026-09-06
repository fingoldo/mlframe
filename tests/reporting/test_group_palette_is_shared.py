"""Two charts kept private byte-identical copies of the shared line palette.

A local copy means repainting ``colors.LINE_PALETTE`` never reaches these charts -- the drift ``colors.py``
exists to prevent. And because the copies were cycled with a bare modulo, a chart with more groups than the
palette has entries drew two of them in the same colour as two solid marker lines: the legend named them
differently and nothing on the chart told them apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.colors import LINE_PALETTE, line_color
from mlframe.reporting.charts.fairness_calibration import compose_fairness_calibration_figure

N = 24_000


def _fairness(n_groups: int, max_groups: int | None = None):
    """The reliability overlay panel of a fairness figure with ``n_groups`` subgroups."""
    rng = np.random.default_rng(0)
    y = (rng.random(N) < 0.3).astype(int)
    score = np.clip(0.25 * y + rng.random(N) * 0.7, 0, 1)
    groups = np.array([f"subgroup_{i}" for i in rng.integers(0, n_groups, N)])
    kwargs = {"max_groups": max_groups} if max_groups is not None else {}
    spec = compose_fairness_calibration_figure(y, score, groups, **kwargs)
    return spec.panels[0][0]


def test_neither_chart_keeps_a_private_palette_copy():
    """A local tuple silently pins the chart to whatever the palette was when it was copied."""
    import mlframe.reporting.charts.calibration_by_feature as cbf
    import mlframe.reporting.charts.fairness_calibration as fc

    assert not hasattr(fc, "_GROUP_COLORS"), "fairness_calibration still holds a private palette copy"
    assert not hasattr(cbf, "_BIN_COLORS"), "calibration_by_feature still holds a private palette copy"


def test_the_colours_come_from_the_shared_palette():
    """Whatever the palette says today is what the chart draws."""
    panel = _fairness(5)
    for colour in panel.colors or ():
        assert colour in LINE_PALETTE or colour == "#888888", f"{colour!r} is not from the shared palette"


def test_a_repeated_colour_is_given_a_different_line_style():
    """Past the wrap, colour alone stops identifying a group; the dash pattern has to take over."""
    panel = _fairness(13, max_groups=13)
    colours = list(panel.colors or ())
    styles = list(panel.line_styles or ())
    repeated = {c for c in colours if colours.count(c) > 1}
    assert repeated, f"fixture did not wrap the palette ({len(colours)} curves, palette has {len(LINE_PALETTE)})"
    for colour in repeated:
        idx = [i for i, c in enumerate(colours) if c == colour]
        assert len({styles[i] for i in idx}) == len(idx), (
            f"colour {colour} is used by curves {idx} with styles {[styles[i] for i in idx]} -- two of them are " "indistinguishable on the chart"
        )


def test_below_the_wrap_every_curve_keeps_its_markers():
    """The dash fallback must not cost the common case its markers."""
    panel = _fairness(4)
    styles = [s for s in (panel.line_styles or ()) if s]
    assert styles, "no line styles were set"
    assert all(s == "lines+markers" for s in styles[1:]), f"a small-group chart lost its markers: {styles}"


@pytest.mark.parametrize("idx", [0, 5, 9, 10, 19, 20])
def test_line_color_is_what_the_charts_now_call(idx):
    """Guard on the helper the charts delegate to."""
    assert line_color(idx) == LINE_PALETTE[idx % len(LINE_PALETTE)]
