"""Charts fed a spec without a gain, or an error group with no rows, draw without numpy warnings."""

from __future__ import annotations

import warnings

import numpy as np


def test_a_spec_without_a_gain_gets_no_band_and_no_warning():
    """A spec whose mi_gain is NaN has no jitter band; percentiles of its all-NaN column used to warn."""
    from mlframe.training.composite.diagnostics import plot_mi_gain_with_jitter

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fig = plot_mi_gain_with_jitter([{"name": "a", "mi_gain": 0.2}, {"name": "b", "mi_gain": float("nan")}], n_jitter=50)
    (ax,) = fig.axes
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b"]
    assert len(ax.patches) == 2


def test_an_error_group_with_no_rows_draws_a_flat_line_without_a_warning():
    """No row is UNDER-predicted: that group's density is zero, not a 0/0 division."""
    from mlframe.reporting.charts._error_bias import _error_bias_panel

    rng = np.random.default_rng(0)
    col = rng.normal(size=300)
    over = np.zeros(300, dtype=bool)
    over[:30] = True
    masks = {"OVER": over, "UNDER": np.zeros(300, dtype=bool), "MAJORITY": ~over}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = _error_bias_panel(col, "x", masks=masks, resid_signed=rng.normal(size=300), nbins=10,
                                group_colors={"OVER": "red", "UNDER": "blue", "MAJORITY": "grey"})
    panel = out[0]
    assert panel.series_labels == ("OVER", "UNDER", "MAJORITY")
    assert np.all(np.asarray(panel.y[1]) == 0.0)
    assert np.all(np.isfinite(np.asarray(panel.y[0]))) and np.asarray(panel.y[0]).max() > 0.0
