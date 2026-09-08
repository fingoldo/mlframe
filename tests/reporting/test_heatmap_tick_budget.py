"""A heatmap that grows its figure must label the rows the extra height bought.

The tick budget was the constant 8, so the drift heatmap -- which deliberately sizes its figure from the
feature count, 40 rows over ~14 inches -- named eight of them and left the reader unable to tell which
feature a row belonged to. The budget now comes from the axis's real extent, and the labels are truncated
the way both bar branches have always truncated theirs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts.drift import psi_heatmap
from mlframe.reporting.renderers._shared_helpers import _BAR_LABEL_MAXLEN, _HEATMAP_MAX_TICKS, ticks_that_fit
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer

N_FEATURES = 40


def _drift_spec(name_len: str = "short"):
    """A 40-feature drift heatmap, the shape whose figure grows with the feature count."""
    rng = np.random.default_rng(0)
    n = 8000
    stem = "engineered_feature" if name_len == "short" else "engineered_feature_with_a_generated_name_far_past_any_sane_length"
    X = pd.DataFrame({f"{stem}_{i}_ratio_log": rng.normal(i * 0.01, 1, n) for i in range(N_FEATURES)})
    ts = np.sort(rng.uniform(0, 1, n))
    return psi_heatmap(X, ts, baseline_mask=ts < 0.2)


def test_the_tick_budget_follows_the_axis_extent():
    """The helper itself: more room means more names, and an unknown extent keeps the old floor."""
    assert ticks_that_fit(None, 40) == _HEATMAP_MAX_TICKS, "an unmeasurable axis must fall back to the fixed cap"
    assert ticks_that_fit(0.0, 40) == _HEATMAP_MAX_TICKS, "a degenerate extent must fall back too"
    assert ticks_that_fit(14.0, 40) > _HEATMAP_MAX_TICKS, "14 inches holds far more than 8 labels"
    assert ticks_that_fit(14.0, 40) <= 40, "never more ticks than there are rows"
    assert ticks_that_fit(1.0, 40) == _HEATMAP_MAX_TICKS, "a tiny panel must not be given more labels than the floor"


def test_matplotlib_names_every_row_of_a_tall_drift_heatmap():
    """The figure is ~14 inches tall for 40 rows; naming 8 of them wastes the height it just bought."""
    spec = _drift_spec()
    assert spec.figsize[1] > 10.0, "fixture no longer produces a tall figure, so this test cannot detect the waste"
    fig = MatplotlibRenderer().render(spec)
    try:
        heat = [ax for ax in fig.axes if ax.get_images()]
        assert heat, "no heatmap axes"
        n_ticks = len(heat[0].get_yticks())
        assert n_ticks > _HEATMAP_MAX_TICKS, f"only {n_ticks} of {N_FEATURES} rows are named on a {spec.figsize[1]:.1f}in figure"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plotly_names_the_same_rows_as_matplotlib():
    """Same spec, same reading: a backend-dependent row count is a defect in its own right."""
    spec = _drift_spec()
    fig = PlotlyRenderer().render(spec)
    y_axes = [a for a in fig.layout if a.startswith("yaxis")]
    assert y_axes, "no y axis on the plotly figure"
    tickvals = fig.layout[y_axes[0]].tickvals or []
    assert len(tickvals) > _HEATMAP_MAX_TICKS, f"plotly names only {len(tickvals)} of {N_FEATURES} rows"


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_a_pathological_row_name_is_truncated_not_run_off_the_edge(backend):
    """Bars truncate their category labels; the heatmap branch never did, on either backend."""
    spec = _drift_spec(name_len="long")
    raw = spec.panels[0][0].row_labels
    assert max(len(r) for r in raw) > _BAR_LABEL_MAXLEN, "fixture names are not long enough to need truncation"

    if backend == "plotly":
        fig = PlotlyRenderer().render(spec)
        drawn = list(fig.layout[next(a for a in fig.layout if a.startswith("yaxis"))].ticktext or [])
    else:
        fig = MatplotlibRenderer().render(spec)
        heat = [ax for ax in fig.axes if ax.get_images()]
        drawn = [t.get_text() for t in heat[0].get_yticklabels()]
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert drawn, "no row labels were drawn"
    # ``truncate_bar_label`` keeps ``maxlen - 1`` characters and appends an ellipsis, so a truncated label is
    # two chars over the nominal cap. That allowance is the existing documented contract (see
    # test_renderer_audit_regressions.py), not something this test gets to redefine.
    longest = max(len(d) for d in drawn)
    assert longest <= _BAR_LABEL_MAXLEN + 2, f"a {longest}-char row label was drawn untruncated"
    assert any(d.endswith("...") for d in drawn), "nothing was actually truncated, so the cap is not being applied"
