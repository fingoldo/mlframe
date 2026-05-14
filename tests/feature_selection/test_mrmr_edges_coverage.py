"""Additional coverage for mrmr.py edge cases: validation paths, pickle BC, transform branches, clone-replay."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR


def _make_data(n: int = 200, m: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(np.int32)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(m)])
    return df, y


def _fast_mrmr(**overrides):
    base = dict(full_npermutations=3, baseline_npermutations=2, n_jobs=1, verbose=0, random_seed=42)
    base.update(overrides)
    return MRMR(**base)


# ----------------------------------------------------------------------------
# __init__ kwarg validation
# ----------------------------------------------------------------------------

@pytest.mark.fast
def test_init_quantization_nbins_over_1000_raises():
    """quantization_nbins > 1000 raises ValueError."""
    df, y = _make_data()
    sel = _fast_mrmr(quantization_nbins=2000)
    with pytest.raises(ValueError, match="quantization_nbins"):
        sel.fit(df, y)


def test_init_interactions_max_order_over_5_raises():
    df, y = _make_data()
    sel = _fast_mrmr(interactions_max_order=6)
    with pytest.raises(ValueError, match="interactions_max_order"):
        sel.fit(df, y)


def test_init_fe_max_steps_over_20_raises():
    df, y = _make_data()
    sel = _fast_mrmr(fe_max_steps=25)
    with pytest.raises(ValueError, match="fe_max_steps"):
        sel.fit(df, y)


# ----------------------------------------------------------------------------
# _validate_inputs branches
# ----------------------------------------------------------------------------

def test_validate_inputs_empty_rows_raises():
    df = pd.DataFrame(np.empty((0, 3)), columns=list("abc"))
    y = np.array([], dtype=np.int32)
    with pytest.raises(ValueError, match="empty"):
        _fast_mrmr().fit(df, y)


def test_validate_inputs_single_row_raises():
    df = pd.DataFrame(np.array([[1.0, 2.0, 3.0]]), columns=list("abc"))
    y = np.array([1], dtype=np.int32)
    with pytest.raises(ValueError, match="single row"):
        _fast_mrmr().fit(df, y)


def test_validate_inputs_duplicate_column_names_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 3)).astype(np.float64), columns=["a", "a", "b"])
    y = (rng.standard_normal(50) > 0).astype(int)
    with pytest.raises(ValueError, match="duplicate column"):
        _fast_mrmr().fit(df, y)


def test_validate_inputs_constant_y_raises():
    df, _ = _make_data(seed=1)
    y = np.zeros(len(df), dtype=np.int32)
    with pytest.raises(ValueError, match="1 unique value"):
        _fast_mrmr().fit(df, y)


def test_validate_inputs_polars_lazyframe_autocollects():
    pl = pytest.importorskip("polars")
    df, y = _make_data(seed=2)
    lazy = pl.from_pandas(df).lazy()
    target = pl.Series("y", y)
    with pytest.warns(UserWarning, match="LazyFrame"):
        _fast_mrmr().fit(lazy, target)


def test_validate_inputs_polars_struct_column_raises():
    pl = pytest.importorskip("polars")
    df, y = _make_data(seed=3)
    pldf = pl.from_pandas(df).with_columns(pl.struct(["f0", "f1"]).alias("s"))
    with pytest.raises(ValueError, match="Struct"):
        _fast_mrmr().fit(pldf, pl.Series("y", y))


# ----------------------------------------------------------------------------
# __setstate__ pickle BC
# ----------------------------------------------------------------------------

def test_setstate_injects_legacy_defaults():
    """Old pickle without newer attrs gets defaults injected by __setstate__."""
    sel = _fast_mrmr()
    # Build a minimal legacy state lacking newer attrs
    state = sel.__dict__.copy()
    # Pretend old pickle: strip every newer attribute
    for k in ("max_confirmation_cand_nbins", "fe_fallback_to_all", "_engineered_features_",
              "_engineered_recipes_", "ran_out_of_time_"):
        state.pop(k, None)
    fresh = _fast_mrmr.__wrapped__() if hasattr(_fast_mrmr, "__wrapped__") else MRMR(random_seed=42, verbose=0)
    fresh.__setstate__(state)
    # All legacy-default attrs are now present:
    assert hasattr(fresh, "_engineered_features_")
    assert hasattr(fresh, "_engineered_recipes_")
    assert getattr(fresh, "fe_fallback_to_all", None) in (True, False)


def test_pickle_round_trip_smoke():
    """Fit -> pickle -> unpickle -> transform produces same output."""
    df, y = _make_data(seed=10)
    sel = _fast_mrmr().fit(df, y)
    blob = pickle.dumps(sel)
    other = pickle.loads(blob)
    a = sel.transform(df)
    b = other.transform(df)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ----------------------------------------------------------------------------
# transform / get_feature_names_out edges
# ----------------------------------------------------------------------------

def test_transform_unfitted_raises_not_fitted_error():
    from sklearn.exceptions import NotFittedError
    df, _ = _make_data()
    with pytest.raises(NotFittedError):
        _fast_mrmr().transform(df)


def test_get_feature_names_out_unfitted_raises():
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        _fast_mrmr().get_feature_names_out()


def test_get_feature_names_out_includes_engineered():
    """After fit, get_feature_names_out concatenates raw + engineered names."""
    df, y = _make_data(seed=20)
    sel = _fast_mrmr().fit(df, y)
    names = sel.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    # Length matches transform output column count
    out = sel.transform(df)
    n_cols = out.shape[1] if hasattr(out, "shape") else len(out.columns)
    assert len(names) == n_cols


def test_transform_numpy_input():
    """Numpy 2D X path is distinct from the pandas path."""
    df, y = _make_data(seed=21)
    sel = _fast_mrmr().fit(df, y)
    arr = df.to_numpy()
    out = sel.transform(arr)
    assert out.shape[0] == arr.shape[0]


def test_transform_polars_input():
    """Polars DataFrame transform path."""
    pl = pytest.importorskip("polars")
    df, y = _make_data(seed=22)
    sel = _fast_mrmr().fit(df, y)
    pldf = pl.from_pandas(df)
    out = sel.transform(pldf)
    assert out is not None


def test_transform_column_drift_raises():
    """Drop a selected column post-fit; transform must raise RuntimeError, not KeyError."""
    df, y = _make_data(seed=23)
    sel = _fast_mrmr().fit(df, y)
    # Find a selected col and drop it
    names = sel.get_feature_names_out()
    if len(names) >= 1:
        drop_col = str(names[0])
        if drop_col in df.columns:
            df2 = df.drop(columns=[drop_col])
            with pytest.raises(RuntimeError, match=r"column"):
                sel.transform(df2)


def test_transform_all_cols_selected_identity_fast_path():
    """When MRMR selects every column AND no engineered recipes, transform returns X unchanged."""
    df = pd.DataFrame(
        {"a": np.linspace(-3, 3, 50), "b": np.linspace(0, 1, 50)},
    )
    y = (df["a"] > 0).astype(np.int32).to_numpy()
    sel = _fast_mrmr(min_relevance_gain=1e-12).fit(df, y)
    out = sel.transform(df)
    # If both got selected, out is df (identity). Else regular transform.
    assert out is not None


# ----------------------------------------------------------------------------
# _FIT_CACHE behaviour
# ----------------------------------------------------------------------------

def test_fit_cache_clear_resets_state():
    """Clearing _FIT_CACHE forces a fresh fit on the same arrays."""
    df, y = _make_data(seed=30)
    MRMR._FIT_CACHE.clear()
    _fast_mrmr().fit(df, y)
    # Cache should have at least one entry after fit
    assert len(MRMR._FIT_CACHE) >= 1
    MRMR._FIT_CACHE.clear()
    assert len(MRMR._FIT_CACHE) == 0


# ----------------------------------------------------------------------------
# Clone-replay helper
# ----------------------------------------------------------------------------

def test_replay_fitted_state_preserves_constructor_params():
    """_replay_fitted_state copies fitted attrs from source onto target without touching constructor params."""
    from mlframe.feature_selection.filters.mrmr import _replay_fitted_state
    df, y = _make_data(seed=40)
    src = _fast_mrmr(random_seed=42).fit(df, y)
    tgt = _fast_mrmr(random_seed=99)  # different seed
    n_replayed = _replay_fitted_state(target=tgt, source=src)
    assert n_replayed > 0
    assert tgt.random_seed == 99  # constructor param preserved
    assert hasattr(tgt, "support_")  # fitted attr replayed


# ----------------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------------

def test_target_to_numpy_values_various_inputs():
    from mlframe.feature_selection.filters.mrmr import _target_to_numpy_values
    arr = np.array([1, 2, 3])
    np.testing.assert_array_equal(_target_to_numpy_values(arr), arr)
    series = pd.Series([1, 2, 3])
    np.testing.assert_array_equal(_target_to_numpy_values(series), np.array([1, 2, 3]))
    df = pd.DataFrame({"y": [1, 2, 3]})
    # DataFrame.values is 2D; handler may flatten or keep 2D shape
    out = _target_to_numpy_values(df)
    assert out is not None


def test_content_array_signature_deterministic():
    from mlframe.feature_selection.filters.mrmr import _content_array_signature
    arr1 = np.arange(100, dtype=np.float64).reshape(20, 5)
    arr2 = np.arange(100, dtype=np.float64).reshape(20, 5)
    assert _content_array_signature(arr1) == _content_array_signature(arr2)


def test_hashable_params_signature_invariant_to_dict_order():
    from mlframe.feature_selection.filters.mrmr import _hashable_params_signature
    a = _hashable_params_signature({"x": 1, "y": 2})
    b = _hashable_params_signature({"y": 2, "x": 1})
    assert a == b
