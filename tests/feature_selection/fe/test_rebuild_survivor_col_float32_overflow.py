"""A survivor column whose float64 value overflows float32 is scrubbed to a finite value, not left as inf."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.feature_engineering import _rebuild_full_survivor_col


def test_rebuild_survivor_col_float32_overflow_is_finite():
    """mul of 1e30 * 1e30 is finite in float64 but inf in float32; the returned column must be finite."""
    X = pd.DataFrame({"a": [1e30, 2.0, 3.0], "b": [1e30, 1.0, 1.0]})
    config = (((0, "identity"), (1, "identity")), "mul", 0)
    out = _rebuild_full_survivor_col(
        config, X, {0: 0, 1: 1}, {"identity": lambda x: x}, {"mul": np.multiply},
    )
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert out[1] == 2.0
