"""The host binned-aggregate path bins group and aggregate values at float64 even under the relaxed float32 crit dtype.

The device path already bins at float64. The host path cast the group column to ``crit_np_dtype()`` (float32 by default) before
quantile binning. Group values near ``1e8`` then collapse onto one float32 value, so every row lands in one cell and no
``binagg_*`` column is emitted, while the float64 path finds the decile structure and emits them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._binned_numeric_agg_fe import binned_numeric_agg_with_recipes


def test_host_binning_keeps_float64_group_resolution_under_relaxed_crit_dtype(monkeypatch):
    """A group column whose signal lives below float32 resolution still yields binned aggregate columns under relaxed crit dtype."""
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", "1")
    rng = np.random.default_rng(11)
    n = 3000
    u = rng.uniform(0.0, 1.0, n)
    group = 1e8 + u
    y = (u > 0.5).astype(float)
    agg = 3.0 * y + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"g": group, "a": agg})

    _, appended, _ = binned_numeric_agg_with_recipes(
        X, y, group_num_cols=["g"], agg_num_cols=["a"], mi_gate=False, redundancy_gate=False,
    )

    assert any(name.startswith("binagg_mean(a|qbin(g))") for name in appended), appended
