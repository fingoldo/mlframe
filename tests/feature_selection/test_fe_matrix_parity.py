"""P0 parity tests for the matrix-native FE adapter (``_fe_matrix_io``).

The adapter is GATED OFF + UN-WIRED, so these only exercise the conversion contract:
float32 value parity (against a float32-cast baseline, NOT the raw float64 frame) + exact
round-trip of numeric / nullable / categorical columns across pandas, polars, numpy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_matrix_io import (
    NON_PURE_FE_FAMILIES,
    fe_matrix_p0_enabled,
    from_feature_matrix,
    to_feature_matrix,
)


def test_gate_default_off(monkeypatch):
    monkeypatch.delenv("MLFRAME_FE_MATRIX_P0", raising=False)
    assert fe_matrix_p0_enabled() is False
    monkeypatch.setenv("MLFRAME_FE_MATRIX_P0", "1")
    assert fe_matrix_p0_enabled() is True


def test_non_pure_family_contract_enumerated():
    # The streaming phase relies on this list to know which families need a full-column anchor.
    for fam in ("smart_log", "safe_div", "grad1", "grad2", "prewarp"):
        assert fam in NON_PURE_FE_FAMILIES


@pytest.mark.parametrize("dist", ["uniform", "normal", "lognormal", "heavytail"])
def test_numeric_pandas_float32_roundtrip(dist):
    rng = np.random.default_rng(0)
    n = 2000
    if dist == "uniform":
        a = rng.uniform(1, 5, n)
    elif dist == "normal":
        a = rng.normal(0, 1, n)
    elif dist == "lognormal":
        a = rng.lognormal(0, 1, n)
    else:
        a = rng.standard_cauchy(n)
    df = pd.DataFrame({"a": a, "b": rng.normal(size=n)})
    fm = to_feature_matrix(df, dtype=np.float32)
    # values match the FLOAT32-CAST baseline (the intended behaviour change), not float64.
    np.testing.assert_array_equal(fm.numeric_column("a"), df["a"].to_numpy().astype(np.float32))
    back = from_feature_matrix(fm)
    np.testing.assert_allclose(back["a"].to_numpy(), a.astype(np.float32), rtol=0, atol=0)


def test_nullable_numeric_roundtrip():
    n = 500
    a = np.arange(n, dtype=np.float64)
    a[::7] = np.nan
    df = pd.DataFrame({"a": a})
    fm = to_feature_matrix(df)
    assert fm.null_mask[:, 0].sum() == np.isnan(a).sum()
    back = from_feature_matrix(fm)
    assert np.array_equal(np.isnan(back["a"].to_numpy()), np.isnan(a))
    finite = ~np.isnan(a)
    np.testing.assert_array_equal(back["a"].to_numpy()[finite], a[finite].astype(np.float32))


def test_categorical_pandas_kept_in_int_plane_not_float():
    n = 1000
    cats = pd.Categorical(np.array(["x", "y", "z"])[np.arange(n) % 3])
    df = pd.DataFrame({"g": cats, "v": np.arange(n, dtype=float)})
    fm = to_feature_matrix(df)
    gj = fm.columns.index("g")
    assert fm.col_kind[gj] == "categorical"
    # codes live in the int plane; the float plane only holds the numeric column.
    assert fm.numeric.shape[1] == 1
    back = from_feature_matrix(fm)
    assert list(back["g"].astype(str)) == list(df["g"].astype(str))


def test_high_cardinality_categorical_no_float_aliasing():
    # > 2**24 distinct codes would alias in a float32 mantissa; the int plane must keep them exact.
    n = 5000
    codes = np.arange(n) + 20_000_000  # large integer labels
    df = pd.DataFrame({"id": pd.Categorical(codes)})
    fm = to_feature_matrix(df)
    back = from_feature_matrix(fm)
    assert list(back["id"].astype(np.int64)) == list(codes)


def test_pandas_string_object_column_factorized_not_crashed():
    """A plain string/object column must NOT crash astype(float32); it factorizes to categorical codes
    and round-trips its labels (regression for the unhandled-string-column bug)."""
    n = 600
    vals = np.array(["red", "green", "blue"])[np.arange(n) % 3]
    df = pd.DataFrame({"color": pd.Series(vals, dtype=object), "v": np.arange(n, dtype=float)})
    fm = to_feature_matrix(df)
    assert fm.col_kind[fm.columns.index("color")] == "categorical"
    back = from_feature_matrix(fm)
    assert list(back["color"].astype(str)) == list(vals)


def test_numpy_roundtrip():
    rng = np.random.default_rng(3)
    arr = rng.normal(size=(300, 4))
    fm = to_feature_matrix(arr, dtype=np.float32)
    assert fm.framework == "numpy"
    np.testing.assert_array_equal(from_feature_matrix(fm), arr.astype(np.float32))


def test_polars_numeric_and_categorical_roundtrip():
    pl = pytest.importorskip("polars")
    n = 800
    rng = np.random.default_rng(5)
    df = pl.DataFrame({
        "x": rng.normal(size=n),
        "g": pl.Series(np.array(["a", "b", "c"])[np.arange(n) % 3]).cast(pl.Categorical),
    })
    fm = to_feature_matrix(df, dtype=np.float32)
    assert fm.framework == "polars"
    np.testing.assert_array_equal(fm.numeric_column("x"), df["x"].to_numpy().astype(np.float32))
    back = from_feature_matrix(fm)
    assert back["g"].to_list() == df["g"].to_list()
    np.testing.assert_array_equal(back["x"].to_numpy().astype(np.float32), df["x"].to_numpy().astype(np.float32))
