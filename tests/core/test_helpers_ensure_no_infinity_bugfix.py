"""Regression test for TYPE4: ensure_no_infinity honest types + unknown-type guard."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.core.helpers import ensure_no_infinity


def test_ensure_no_infinity_pd_returns_frame():
    """Ensure no infinity pd returns frame."""
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
    out = ensure_no_infinity(df)
    assert isinstance(out, pd.DataFrame)


def test_ensure_no_infinity_pl_returns_frame():
    """Ensure no infinity pl returns frame."""
    df = pl.DataFrame({"a": [1.0, float("inf"), 3.0]})
    out = ensure_no_infinity(df)
    assert isinstance(out, pl.DataFrame)


def test_ensure_no_infinity_unknown_type_raises():
    # Previously the implicit-None branch silently returned None on an unknown type.
    """Ensure no infinity unknown type raises."""
    with pytest.raises(TypeError):
        ensure_no_infinity([1.0, 2.0, 3.0])


def test_ensure_no_infinity_ndarray_replaces_inf_in_place():
    """Some model pre-pipelines (e.g. PytorchLightning's eager numpy conversion) hand a raw ndarray to
    the generic pre-fit infinity check instead of a DataFrame -- surfaced by fuzzing (2026-07-06,
    models=[cb,hgb,mlp,xgb]) as ``TypeError: ensure_no_infinity expects a pandas or polars DataFrame;
    got ndarray``. Mirrors ensure_no_infinity_pd's in-place mutate-and-return contract.
    """
    arr = np.array([[1.0, np.inf], [2.0, -np.inf]], dtype=np.float32)
    out = ensure_no_infinity(arr)
    assert out is arr
    assert not np.isinf(out).any()
    np.testing.assert_array_equal(out, [[1.0, 0.0], [2.0, 0.0]])


def test_ensure_no_infinity_int_ndarray_is_noop():
    """Ensure no infinity int ndarray is noop."""
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)
    out = ensure_no_infinity(arr)
    assert out is arr
    np.testing.assert_array_equal(out, arr)
