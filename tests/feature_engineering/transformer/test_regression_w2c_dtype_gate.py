"""Regression sensors for w2c-fe-dtype-gate findings: #13, #19, #23, #25.

Each test asserts behaviour (shape, dtype, value identity) is preserved by the perf-driven changes; a refactor regression that drops dtype= passthrough or
reverts the ndarray-buffer optimization will trip these tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Cache-key invalidation + dtype-passthrough sensors on n<=20 frames; pure helper logic, no fits, wall <0.3s total.
pytestmark = [pytest.mark.fast]


# ---------- #13: _coerce_features_to_float32 cache key invalidates on column mutation ----------


def test_w2c_13_coerce_cache_key_invalidates_on_column_mutation():
    """The cache key must include a cols-signature component so an in-place column add/remove on the same frame id is not served a stale cached ndarray."""
    from mlframe.training.core._phase_recurrent import _coerce_features_to_float32

    df = pd.DataFrame({"a": np.arange(5, dtype=np.float32), "b": np.arange(5, 10, dtype=np.float32)})
    cache = {}

    # First call: 2 columns
    cols_sig_1 = tuple(df.columns)
    arr1 = _coerce_features_to_float32(df, cache=cache, cache_key=("train", id(df), cols_sig_1))
    assert arr1.shape == (5, 2)
    assert arr1.dtype == np.float32

    # Mutate in place: add a column. id(df) unchanged, but cols_signature differs.
    df["c"] = np.arange(10, 15, dtype=np.float32)
    cols_sig_2 = tuple(df.columns)
    assert cols_sig_2 != cols_sig_1
    arr2 = _coerce_features_to_float32(df, cache=cache, cache_key=("train", id(df), cols_sig_2))
    # With the new cache key, arr2 must reflect 3 columns (not the stale 2-col cache hit).
    assert arr2.shape == (5, 3), f"cache key did not invalidate on column mutation; got shape {arr2.shape}"
    # Distinct cache entries
    assert len(cache) == 2


def test_w2c_13_coerce_cache_key_hits_on_repeated_call_same_cols():
    """Repeated call on the same frame and same cols-signature must reuse the cached ndarray (no extra copy)."""
    from mlframe.training.core._phase_recurrent import _coerce_features_to_float32

    df = pd.DataFrame({"a": np.arange(8, dtype=np.float32)})
    cache = {}
    key = ("val", id(df), tuple(df.columns))
    a = _coerce_features_to_float32(df, cache=cache, cache_key=key)
    b = _coerce_features_to_float32(df, cache=cache, cache_key=key)
    assert a is b, "second call must hit cache and return the same ndarray object"


# ---------- #19: bruteforce _kfold_target_encode shape / dtype / value identity ----------


def test_w2c_19_kfold_encode_returns_dataframe_with_correct_shape_dtype_index():
    """Pre-allocated ndarray + wrap-once path must yield the same DataFrame shape, columns, index, and dtype as the prior DataFrame.iloc[]= pattern."""
    try:
        from category_encoders import CatBoostEncoder  # noqa: F401
    except ImportError:
        pytest.skip("category_encoders not installed")
    from mlframe.feature_engineering.bruteforce import _kfold_target_encode

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "cat_a": pd.Categorical(rng.choice(["x", "y", "z"], size=n)),
            "cat_b": pd.Categorical(rng.choice(["p", "q"], size=n)),
            "noise": rng.standard_normal(n),
        },
        index=pd.RangeIndex(n),
    )
    target = pd.Series(rng.integers(0, 2, size=n).astype(float), index=df.index)
    # category_encoders 2.6 / sklearn < 1.6 combos (Python 3.9 CI) break the
    # ``__sklearn_tags__`` super() chain inside CatBoostEncoder.fit; skip
    # on the upstream-incompat path (same guard as sibling tests).
    try:
        out = _kfold_target_encode(df, cols=["cat_a", "cat_b"], target=target, n_splits=5, random_state=0)
    except AttributeError as exc:
        if "__sklearn_tags__" in str(exc):
            pytest.skip(f"category_encoders / sklearn version mismatch on this runner: {exc}.")
        raise
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (n, 2)
    assert list(out.columns) == ["cat_a", "cat_b"]
    # Index identity preserved
    assert (out.index == df.index).all()
    # Every row filled (no NaN slots from missing folds)
    assert out.notna().all().all(), "every row must be encoded by exactly one OOF fold"
    # Dtype is float (pd.DataFrame from float ndarray gives float64)
    assert all(pd.api.types.is_float_dtype(out[c]) for c in out.columns)


def test_w2c_19_kfold_encode_deterministic_under_random_state():
    """Same random_state -> same OOF encoded values, byte-for-byte."""
    try:
        from category_encoders import CatBoostEncoder  # noqa: F401
    except ImportError:
        pytest.skip("category_encoders not installed")
    from mlframe.feature_engineering.bruteforce import _kfold_target_encode

    rng = np.random.default_rng(42)
    n = 150
    df = pd.DataFrame(
        {
            "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
        },
        index=pd.RangeIndex(n),
    )
    target = pd.Series(rng.integers(0, 2, size=n).astype(float), index=df.index)
    # Same ``__sklearn_tags__`` super() chain guard as the sibling test;
    # category_encoders >= 2.6 + sklearn < 1.6 (Python 3.9 CI) breaks the
    # chain inside CatBoostEncoder.fit -> _check_fit_inputs -> _get_tags.
    try:
        out1 = _kfold_target_encode(df, cols=["cat"], target=target, n_splits=5, random_state=123)
        out2 = _kfold_target_encode(df, cols=["cat"], target=target, n_splits=5, random_state=123)
    except AttributeError as exc:
        if "__sklearn_tags__" in str(exc):
            pytest.skip(f"category_encoders / sklearn version mismatch on this runner: {exc}.")
        raise
    np.testing.assert_allclose(out1.values, out2.values)


# ---------- #23: stacked_attention dtype passthrough ----------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_w2c_23_stacked_attention_dtype_passthrough(dtype):
    """compute_stacked_row_attention must emit a frame whose backing ndarray dtype matches the caller-supplied dtype (no silent float32 default leakage)."""
    from sklearn.model_selection import KFold

    from mlframe.feature_engineering.transformer.stacked_attention import compute_stacked_row_attention

    rng = np.random.default_rng(0)
    n, d = 200, 6
    X = rng.standard_normal((n, d)).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)

    out = compute_stacked_row_attention(
        X_train=X,
        y_train=y,
        X_query=None,
        splitter=splitter,
        seed=0,
        n_layers=2,
        n_heads=2,
        head_dim=3,
        k=4,
        dtype=dtype,
    )
    # Output is polars DataFrame; each column should be the requested dtype family.
    import polars as pl

    assert isinstance(out, pl.DataFrame)
    arr = out.to_numpy()
    # polars stores float32/float64 as Float32/Float64 -> to_numpy preserves
    assert arr.dtype == dtype, f"expected dtype {dtype}, got {arr.dtype}"


# ---------- #25: boosted_attention dtype passthrough ----------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_w2c_25_boosted_attention_dtype_passthrough(dtype):
    """compute_boosted_attention must emit a frame whose backing ndarray dtype matches the caller-supplied dtype."""
    from sklearn.model_selection import KFold

    from mlframe.feature_engineering.transformer.boosted_attention import compute_boosted_attention

    rng = np.random.default_rng(0)
    n, d = 200, 6
    X = rng.standard_normal((n, d)).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)

    out = compute_boosted_attention(
        X_train=X,
        y_train=y,
        X_query=None,
        splitter=splitter,
        seed=0,
        n_boost_layers=2,
        n_heads=2,
        head_dim=3,
        k=4,
        dtype=dtype,
    )
    import polars as pl

    assert isinstance(out, pl.DataFrame)
    arr = out.to_numpy()
    assert arr.dtype == dtype, f"expected dtype {dtype}, got {arr.dtype}"
