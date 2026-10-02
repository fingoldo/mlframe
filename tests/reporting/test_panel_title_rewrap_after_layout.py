"""A panel title is wrapped against the width layout actually gave the panel, not the pre-layout width."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.calibration import build_calibration_spec
from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.renderers import matplotlib as matplotlib_renderer  # noqa: F401 -- the module whose layout call this file pins

_IDENTITY = "TEST 2026-08-06/2026-09-19 CatBoostClassifier alljobs jobsdetails_shuffled target_total_hired_gte_1 BTTR/BTTS=29%/34% w=recency"


def test_a_model_identity_line_that_fits_the_final_width_stays_on_one_line():
    """With a colorbar beside the axes the pre-layout width is 7.4in; the identity line needs ~9.4in and the real room is ~11in."""
    x = np.linspace(0.07, 0.95, 10)
    spec = build_calibration_spec(x, x + 0.02, np.full(10, 1000), plot_title=_IDENTITY + "\nsecond line", figsize=(12, 6))
    fig = get_renderer("matplotlib").render(spec)
    lines = fig.axes[0].get_title().splitlines()
    assert lines[0] == _IDENTITY
    assert lines[1] == "second line"


def test_layout_is_executed_before_the_titles_are_rewrapped():
    """The post-layout pass runs the figure's layout engine itself, so the measured axes width is the final one."""
    from unittest import mock

    from matplotlib.layout_engine import ConstrainedLayoutEngine

    x = np.linspace(0.07, 0.95, 10)
    spec = build_calibration_spec(x, x + 0.02, np.full(10, 1000), plot_title=_IDENTITY, figsize=(12, 6))
    with mock.patch.object(ConstrainedLayoutEngine, "execute", autospec=True, side_effect=ConstrainedLayoutEngine.execute) as execute:
        fig = get_renderer("matplotlib").render(spec)
    execute.assert_called()
    assert execute.call_args.args[1] is fig
