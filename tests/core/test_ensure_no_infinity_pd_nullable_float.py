"""ensure_no_infinity_pd handles nullable Float64 columns holding both pd.NA and inf, and keeps their dtype."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.core.helpers import ensure_no_infinity_pd


def test_nullable_float_with_na_and_inf_is_sanitised_and_keeps_its_dtype():
    """[1.0, <NA>, inf]: the replacement loop crashed on the object array; now inf becomes the filler and NA stays NA."""
    df = pd.DataFrame({"a": pd.array([1.0, None, np.inf], dtype="Float64")})
    out = ensure_no_infinity_pd(df, nans_filler=-1.0)
    assert str(out["a"].dtype) == "Float64"
    assert out["a"].iloc[0] == 1.0 and out["a"].iloc[2] == -1.0 and pd.isna(out["a"].iloc[1])


def test_plain_float_column_is_unchanged_in_behaviour():
    """A numpy float64 column: inf replaced, NaN kept (negative control)."""
    df = pd.DataFrame({"a": [1.0, np.nan, -np.inf]})
    out = ensure_no_infinity_pd(df, nans_filler=0.0)
    assert out["a"].iloc[2] == 0.0 and np.isnan(out["a"].iloc[1])
