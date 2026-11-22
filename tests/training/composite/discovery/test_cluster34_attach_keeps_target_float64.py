"""Audit 2026-11-20 finding 34-F30: the attached target keeps float64 precision; engineered bases stay float32.

With |y| near 1e8 a float32 target differs from the float64 one by more than the base-to-target gap, so the val-side bases did not
match the train-side ones.
"""

import numpy as np
import pandas as pd

from mlframe.training.composite.discovery._grouped_causal_bases import _attach_columns


def test_target_column_kept_float64_bases_float32():
    """The target keeps full precision (1e8 + 0.25 survives); a base column is still stored as float32."""
    df = pd.DataFrame({"x": np.arange(3, dtype=np.float64)})
    y = np.array([1e8 + 0.25, 1e8 + 0.5, 1e8 + 0.75])
    out = _attach_columns(df, {"y": y, "b": np.arange(3, dtype=np.float64)}, keep_float64=frozenset({"y"}))
    assert out["y"].dtype == np.float64
    np.testing.assert_array_equal(out["y"].to_numpy(), y)
    assert out["b"].dtype == np.float32
