"""One non-finite residual no longer removes its bin from the worst-bias segment search."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts._error_bias import _error_bias_panel


def test_nan_residual_in_the_worst_bin_does_not_drop_that_bin():
    """Residual +5 for x > 0.9, one NaN inside that bin: the worst |bias| must still be ~5 (the NaN made the bin's mean NaN and the search named another bin)."""
    rng = np.random.default_rng(0)
    n = 2000
    col = rng.random(n)
    resid = rng.normal(scale=0.1, size=n)
    resid[col > 0.9] += 5.0
    resid[np.flatnonzero(col > 0.9)[0]] = np.nan
    over = resid < -0.2
    under = resid > 0.2
    masks = {"OVER": over, "UNDER": under, "MAJORITY": ~(over | under)}
    out = _error_bias_panel(col, "x", masks=masks, resid_signed=resid, nbins=10, group_colors={"OVER": "red", "UNDER": "blue", "MAJORITY": "grey"})
    assert out is not None
    worst_abs = out[2]
    assert worst_abs > 4.0
