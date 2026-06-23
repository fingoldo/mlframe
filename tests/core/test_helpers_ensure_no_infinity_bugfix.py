"""Regression test for TYPE4: ensure_no_infinity honest types + unknown-type guard."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.core.helpers import ensure_no_infinity


def test_ensure_no_infinity_pd_returns_frame():
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
    out = ensure_no_infinity(df)
    assert isinstance(out, pd.DataFrame)


def test_ensure_no_infinity_pl_returns_frame():
    df = pl.DataFrame({"a": [1.0, float("inf"), 3.0]})
    out = ensure_no_infinity(df)
    assert isinstance(out, pl.DataFrame)


def test_ensure_no_infinity_unknown_type_raises():
    # Previously the implicit-None branch silently returned None on an unknown type.
    with pytest.raises(TypeError):
        ensure_no_infinity([1.0, 2.0, 3.0])
