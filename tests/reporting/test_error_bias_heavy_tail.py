"""Error-bias panels on a heavy-tailed feature: readable axis and a worst segment backed by enough rows."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from mlframe.reporting.charts._error_bias import error_bias_per_feature


def test_heavy_feature_uses_asinh_and_worst_segment_has_support():
    rng = np.random.default_rng(0)
    n = 30000
    X = pd.DataFrame({"budget": rng.lognormal(6, 2, n)})
    yt = rng.normal(size=n)
    yp = yt + rng.normal(size=n)
    fig = error_bias_per_feature(X, yt, yp).figure
    panel = fig.panels[0][0]
    assert panel.xscale == "asinh"
    lo, hi = (float(v) for v in re.search(r"in \[([^,]+), ([^\]]+)\]", panel.title).groups())
    assert ((X.budget >= lo) & (X.budget <= hi)).sum() >= 30
