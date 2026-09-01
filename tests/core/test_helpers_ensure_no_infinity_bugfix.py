"""Regression test for TYPE4: ensure_no_infinity honest types + unknown-type guard."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.core.helpers import ensure_no_infinity


def test_ensure_no_infinity_pd_actually_removes_the_infinities():
    """The type assertion this replaced would pass on a function that did nothing at all.

    ``assert isinstance(out, pd.DataFrame)`` is true of the input, so it cannot distinguish a working
    implementation from a no-op or from one that corrupts every value -- which is exactly what happened: the
    NaN-rewriting defect below lived under a green test for as long as this file existed.
    """
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
    out = ensure_no_infinity(df)
    assert isinstance(out, pd.DataFrame)
    assert not np.isinf(out["a"].to_numpy()).any()
    assert list(out["a"]) == [1.0, 0.0, 3.0]


def test_ensure_no_infinity_pl_actually_removes_the_infinities():
    """Same contract on the polars carrier, asserted on the VALUES rather than on the type."""
    df = pl.DataFrame({"a": [1.0, float("inf"), 3.0]})
    out = ensure_no_infinity(df)
    assert isinstance(out, pl.DataFrame)
    assert not np.isinf(out["a"].to_numpy()).any()
    assert out["a"].to_list() == [1.0, 0.0, 3.0]


def test_ensure_no_infinity_pd_keeps_missing_values_missing():
    """A NaN is a signal, not a value to be helpfully filled in.

    ``np.nan_to_num``'s ``nan`` argument defaults to 0.0, so passing only ``posinf``/``neginf`` rewrote every
    NaN in the column to zero as well. It happened only in columns that also contained an infinity, so two
    otherwise identical frames could disagree about their missing values depending on one stray inf -- and a
    tree that would have split on missingness silently saw a plausible number instead.
    """
    df = pd.DataFrame({"has_inf": [1.0, np.nan, np.inf, -np.inf], "no_inf": [1.0, np.nan, 2.0, 3.0]})
    out = ensure_no_infinity(df.copy())
    assert out["has_inf"].isna().sum() == 1, "the NaN in the inf-bearing column must survive"
    assert list(out["has_inf"].fillna(-1)) == [1.0, -1.0, 0.0, 0.0]
    assert out["no_inf"].isna().sum() == 1, "a column with no infinity must be untouched"


def test_ensure_no_infinity_pd_honours_a_custom_filler():
    """The filler is a parameter; a test that only ever exercises the default cannot catch it being ignored."""
    from mlframe.core.helpers import ensure_no_infinity_pd

    out = ensure_no_infinity_pd(pd.DataFrame({"a": [1.0, np.inf, -np.inf]}), nans_filler=-999.0)
    assert list(out["a"]) == [1.0, -999.0, -999.0]


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
