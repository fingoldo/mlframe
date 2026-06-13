"""Regression: LeakageSafeEncoder must survive int<->float category dtype drift.

``_categorical_to_string_array`` produces the per-category key used at BOTH fit
and transform. A bare ``str`` made the integer ``1`` (``'1'``) and the float
``1.0`` (``'1.0'``) DIFFERENT keys, so fitting on an integer-coded categorical
then transforming the SAME column arriving as float (a routine polars int->float
promotion / pandas join upcast) missed every per-category entry and returned the
prior for every row -- a silently wrong target encoding.

These tests FAIL on the pre-fix bare-``str`` coercion (every transform row hits
the global prior) and PASS once integral int/float values collapse to one key.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import pytest

from mlframe.training.feature_handling.target_encoders import (
    _NULL_SENTINEL,
    LeakageSafeEncoder,
    _categorical_to_string_array,
)


def test_polars_float_nan_maps_to_null_sentinel() -> None:
    """Polars float NaN must map to the null sentinel, like pandas/numpy.

    Pre-fix the polars float branch keyed NaN as ``repr(nan) == 'nan'`` (the
    ``is_null()`` mask does not flag NaN and the ``"NaN"`` string-rebrand never
    matched the lowercase token), so a polars float categorical with NaN cells
    produced a spurious ``'nan'`` category instead of the unified NULL bucket --
    diverging from the pandas / numpy backends on the same data.
    """
    pl = pytest.importorskip("polars")
    ps = pl.Series("a", [1.0, float("nan"), None, 2.0], dtype=pl.Float64)
    out = _categorical_to_string_array(ps)
    assert list(out) == ["1", _NULL_SENTINEL, _NULL_SENTINEL, "2"]

    # Cross-backend parity: pandas float NaN already collapses to the sentinel.
    pd_out = _categorical_to_string_array(pd.Series([1.0, float("nan"), 2.0]))
    assert pd_out[1] == _NULL_SENTINEL
    np_out = _categorical_to_string_array(np.array([1.0, np.nan, 2.0]))
    assert np_out[1] == _NULL_SENTINEL


def test_polars_bool_token_matches_pandas_numpy() -> None:
    """Polars bool must yield canonical ``"True"``/``"False"`` tokens, like pandas/numpy/list.

    Pre-fix the polars Boolean branch went through ``cast(pl.Utf8)``, which emits lowercase
    ``"true"``/``"false"`` -- diverging from every other backend's ``_canonical_cat_token`` form. A bool
    categorical fit as polars then transformed as pandas (strategy-layer frame swap) missed every key and
    returned the prior for every row.
    """
    pl = pytest.importorskip("polars")
    pl_out = _categorical_to_string_array(pl.Series("b", [True, False, True, None], dtype=pl.Boolean))
    assert list(pl_out) == ["True", "False", "True", _NULL_SENTINEL]
    # Cross-backend parity (no nulls): pandas / numpy / list all canonicalise to "True"/"False".
    pd_out = _categorical_to_string_array(pd.Series([True, False, True]))
    np_out = _categorical_to_string_array(np.array([True, False, True]))
    list_out = _categorical_to_string_array([True, False, True])
    assert list(pd_out) == list(np_out) == list(list_out) == ["True", "False", "True"]
    assert list(pl_out)[:3] == list(pd_out)


def test_leakage_safe_encoder_transform_robust_to_bool_frame_drift() -> None:
    """Fit on polars bool categories, transform the SAME categories as pandas bool.

    Pre-fix: polars keyed "true"/"false", pandas transform produced "True"/"False" -> every row missed its
    key and collapsed to the global prior (a single distinct value). Post-fix: both levels recovered.
    """
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(7)
    n = 800
    flags = rng.integers(0, 2, n).astype(bool)
    y = (flags * 0.4 + rng.standard_normal(n) * 0.05).astype(np.float64)

    enc = LeakageSafeEncoder(method="target_mean", smoothing=1.0, cv=5)
    enc.fit(pl.Series("b", flags, dtype=pl.Boolean), y)
    out = enc.transform(pd.Series(flags))
    # Two bool levels recovered, not collapsed to the single global prior.
    assert len(np.unique(np.round(out, 6))) == 2
    assert not np.allclose(out, float(enc._global_prior))


def test_categorical_to_string_array_int_float_agree() -> None:
    a = _categorical_to_string_array(pd.Series([1, 2, 1], dtype="int64"))
    b = _categorical_to_string_array(pd.Series([1.0, 2.0, 1.0], dtype="float64"))
    c = _categorical_to_string_array(np.array([1, 2, 1], dtype=np.int64))
    d = _categorical_to_string_array(np.array([1.0, 2.0, 1.0], dtype=np.float64))
    assert list(a) == list(b) == list(c) == list(d) == ["1", "2", "1"]
    # NaN still maps to the null sentinel; non-integral floats keep precision.
    nan_out = _categorical_to_string_array(np.array([1.0, np.nan]))
    assert list(nan_out) == ["1", "__NULL__"]
    assert list(_categorical_to_string_array(pd.Series([1.5, 2.0]))) == [repr(1.5), "2"]


def test_leakage_safe_encoder_transform_robust_to_int_float_drift() -> None:
    """Fit on int categories, transform the SAME categories as float.

    Pre-fix: every float row missed its '1'-style key (transform produced
    '1.0') and got the global prior -> a SINGLE distinct encoded value.
    Post-fix: the per-category means are recovered (5 distinct levels).
    """
    rng = np.random.default_rng(3)
    n = 1000
    cats = rng.integers(0, 5, n)
    y = (cats * 0.15 + rng.standard_normal(n) * 0.05).astype(np.float64)

    enc = LeakageSafeEncoder(method="target_mean", smoothing=1.0, cv=5)
    enc.fit(pd.Series(cats, dtype="int64"), y)

    out_int = enc.transform(pd.Series(cats, dtype="int64"))
    out_float = enc.transform(pd.Series(cats.astype("float64"), dtype="float64"))

    assert np.allclose(out_int, out_float)
    # Per-category levels recovered, not collapsed to the single global prior.
    assert len(np.unique(np.round(out_float, 6))) == 5
    assert not np.allclose(out_float, float(enc._global_prior))
