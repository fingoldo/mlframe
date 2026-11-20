"""A feature with no defined full-sample correlation sign must not be reported stable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._monotonic_stability import monotonic_deviation_stability_filter


def _frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Groups of 6 rows with one informative column and one column whose deviation is constant."""
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(60), 6)
    n = groups.size
    signal = rng.normal(size=n)
    df = pd.DataFrame({"signal": signal, "flat": 7.0, "group": groups})
    y = signal + 0.1 * rng.normal(size=n)
    return df, y


def test_constant_deviation_feature_is_not_stable() -> None:
    """The constant column has an undefined correlation, so it cannot be sign-stable; the real signal still is."""
    df, y = _frame()
    res = monotonic_deviation_stability_filter(df, y, group_col="group", feature_cols=["signal", "flat"], random_state=0).set_index("feature")
    assert bool(res.loc["signal", "stable"]) is True
    assert bool(res.loc["flat", "stable"]) is False
    assert res.loc["flat", "sign_agreement_fraction"] == 0.0
