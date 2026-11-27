"""Cluster 23 regression: a missing or zero denominator keeps the derived ratio NaN instead of turning it into 0.0 (audit 2026-11-20, 23-F6)."""

import numpy as np
import pandas as pd

from mlframe.evaluation.leak_scan import _build_derived_matrix


def test_missing_and_zero_denominators_stay_nan():
    """Rows with a NaN or zero denominator are NaN in the ratio column; ordinary rows keep their quotient (23-F6)."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, np.nan, 0.0, 4.0]})
    derived, names = _build_derived_matrix(X, ["a", "b"], ("ratio",), None)
    assert names == ["a / b"]
    col = derived[:, 0]
    assert col[0] == 0.5 and col[3] == 1.0
    assert np.isnan(col[1]) and np.isnan(col[2])
