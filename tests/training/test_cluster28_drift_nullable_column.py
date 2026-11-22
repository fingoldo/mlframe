"""Audit 2026-11-20 finding 28-F35: a nullable pandas column with pd.NA converts to float with NaN, not to an object array.

The float cast on the object array raised, and the caller dropped the whole drift sensor for that target. The helper now converts the
nullable column with NaN for the missing cells.
"""

import numpy as np
import pandas as pd

from mlframe.training.feature_drift_report import _col_to_numpy


def test_nullable_int_column_with_na_becomes_float_with_nan():
    """Int64 with pd.NA gives a float64 array whose missing cell is NaN."""
    df = pd.DataFrame({"x": pd.array([1, None, 3], dtype="Int64")})
    out = _col_to_numpy(df, "x")
    assert out.dtype == np.float64
    assert np.isnan(out[1]) and out[0] == 1.0 and out[2] == 3.0
