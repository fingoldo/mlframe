"""NaNs in a scanned column are reported as missingness, not ranked last into the value correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.evaluation.leak_scan import scan_temporal_leak


def _frame():
    """A noise column whose values are missing only in the second split."""
    rng = np.random.default_rng(0)
    n = 600
    split = np.repeat([0, 1], n // 2).astype(float)
    x = rng.normal(size=n)
    x[n // 2 :] = np.where(rng.random(n // 2) < 0.9, np.nan, x[n // 2 :])
    return pd.DataFrame({"gappy": x}), split


def test_value_correlation_ignores_missing_rows_and_missingness_is_reported():
    """The value correlation of a noise column stays near zero while the missingness correlation carries the split signal."""
    X, split = _frame()
    row = scan_temporal_leak(X, split).iloc[0]
    assert abs(row["correlation"]) < 0.3
    assert row["missing_correlation"] > 0.7
    assert bool(row["flagged"])


def test_complete_column_has_zero_missing_correlation():
    """A column without gaps reports zero missingness correlation."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=100)})
    out = scan_temporal_leak(X, np.arange(100) // 50)
    assert out["missing_correlation"].iloc[0] == 0.0
