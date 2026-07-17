"""Regression test for the homogeneous-numeric fast path in ``create_shadow_features``.

``create_shadow_features`` used ``X.apply(lambda col: _rng.permutation(col.values))`` -- one
``_rng.permutation`` per column in column order, each wrapped in a pandas Series. When every
column shares one numpy numeric dtype it now permutes a no-upcast 2-D ``to_numpy()`` buffer
column-by-column instead (~1.9x), and falls back to a dtype-preserving dict loop otherwise.

Both paths must be BIT-IDENTICAL to the original ``.apply``: identical shadow values, identical
per-column dtype, identical column order -- so the RNG stream (and every downstream hit) is
unchanged. This test asserts that against the ``.apply`` reference for the float (fast path),
int32 (fast path), and mixed/categorical (fallback path) regimes, all seeded identically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.boruta_shap import BorutaShap


def _apply_reference(X: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xs = X.apply(lambda col: rng.permutation(col.values))
    Xs.columns = ["shadow_" + str(c) for c in X.columns]
    return Xs


def _run_create_shadow(X: pd.DataFrame, seed: int) -> pd.DataFrame:
    bs = BorutaShap.__new__(BorutaShap)
    bs.X = X.copy()
    bs._rng = np.random.default_rng(seed)
    bs.create_shadow_features()
    return bs.X_shadow


def _assert_identical(ref: pd.DataFrame, got: pd.DataFrame, label: str):
    assert list(ref.columns) == list(got.columns), f"[{label}] column order diverged"
    assert list(ref.dtypes.astype(str)) == list(got.dtypes.astype(str)), (
        f"[{label}] dtypes diverged: ref={list(ref.dtypes.astype(str))} got={list(got.dtypes.astype(str))}"
    )
    for c in ref.columns:
        if str(ref[c].dtype) == "category":
            assert np.array_equal(ref[c].cat.codes.values, got[c].cat.codes.values), f"[{label}] {c} codes diverged"
        else:
            assert np.array_equal(np.asarray(ref[c]), np.asarray(got[c])), f"[{label}] {c} values diverged"


def test_shadow_fast_path_float_bit_identical():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((400, 30)), columns=[f"f{i}" for i in range(30)])
    _assert_identical(_apply_reference(X, 7), _run_create_shadow(X, 7), "float")


def test_shadow_fast_path_int32_bit_identical():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.integers(0, 7, (400, 20)).astype("int32"), columns=[f"i{i}" for i in range(20)])
    _assert_identical(_apply_reference(X, 11), _run_create_shadow(X, 11), "int32")


def test_shadow_fallback_mixed_dtypes_bit_identical():
    rng = np.random.default_rng(2)
    data = {f"f{i}": rng.random(300) for i in range(10)}
    for i in range(10, 14):
        data[f"i{i}"] = rng.integers(0, 5, 300).astype("int32")
    data["c0"] = pd.Categorical(rng.integers(0, 3, 300))
    X = pd.DataFrame(data)
    _assert_identical(_apply_reference(X, 5), _run_create_shadow(X, 5), "mixed")


def test_shadow_fallback_bool_bit_identical():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({f"b{i}": rng.integers(0, 2, 300).astype(bool) for i in range(6)})
    _assert_identical(_apply_reference(X, 9), _run_create_shadow(X, 9), "bool")
