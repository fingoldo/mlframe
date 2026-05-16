"""Additional coverage for engineered_recipes.py -- exercise each recipe-kind dispatch + helpers."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.engineered_recipes import (
    EngineeredRecipe,
    apply_recipe,
)

# Per-helper presence flags so a single missing name doesn't skip the entire file.
def _try_import(name):
    try:
        mod = __import__("mlframe.feature_selection.filters.engineered_recipes", fromlist=[name])
        return getattr(mod, name, None)
    except ImportError:
        return None


build_unary_binary_recipe = _try_import("build_unary_binary_recipe")
_apply_factorize = _try_import("_apply_factorize")
_apply_factorize_kway = _try_import("_apply_factorize_kway")
_apply_target_encoding = _try_import("_apply_target_encoding")
_apply_unary_binary = _try_import("_apply_unary_binary")
_coerce_to_int_with_nan_handling = _try_import("_coerce_to_int_with_nan_handling")
_extract_column = _try_import("_extract_column")
_extra_equal = _try_import("_extra_equal")
_handle_missing = _try_import("_handle_missing")
_maybe_collect_lazy = _try_import("_maybe_collect_lazy")

_HAVE_PRIVATES = build_unary_binary_recipe is not None  # legacy flag for old tests


def _basic_df():
    return pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})


# ----------------------------------------------------------------------------
# EngineeredRecipe dataclass
# ----------------------------------------------------------------------------

@pytest.mark.fast
def test_engineered_recipe_equality():
    """Two recipes with same fields are equal."""
    r1 = EngineeredRecipe(name="mul(a,b)", kind="unary_binary", src_names=("a", "b"), extra={"binary_name": "mul"})
    r2 = EngineeredRecipe(name="mul(a,b)", kind="unary_binary", src_names=("a", "b"), extra={"binary_name": "mul"})
    assert r1 == r2


def test_engineered_recipe_pickle_round_trip():
    r = EngineeredRecipe(name="mul(a,b)", kind="unary_binary", src_names=("a", "b"), extra={"binary_name": "mul"})
    blob = pickle.dumps(r)
    r2 = pickle.loads(blob)
    assert r == r2


def test_engineered_recipe_hash_consistent():
    r1 = EngineeredRecipe(name="x", kind="factorize", src_names=("a",), extra={})
    r2 = EngineeredRecipe(name="x", kind="factorize", src_names=("a",), extra={})
    assert hash(r1) == hash(r2)


# ----------------------------------------------------------------------------
# build_unary_binary_recipe + apply_recipe(kind="unary_binary")
# ----------------------------------------------------------------------------

def _build_recipe_via_actual_api(binary: str, unary_a: str, unary_b: str, *, unary_preset: str = "minimal"):
    """Build a unary_binary recipe via the canonical production builder. Pre-fix this helper
    fell through to silent dataclass construction on TypeError, masking signature drift; that
    drift then triggered downstream apply_recipe skip-masks. Now we use the production builder
    directly with its actual keyword names (src_a_name / src_b_name etc) and accept a preset
    override so callers can test transforms that only live in maximal."""
    assert build_unary_binary_recipe is not None, (
        "build_unary_binary_recipe missing from engineered_recipes module; refactor regression."
    )
    return build_unary_binary_recipe(
        name=f"{binary}({unary_a}(a),{unary_b}(b))",
        src_a_name="a",
        src_b_name="b",
        unary_a_name=unary_a,
        unary_b_name=unary_b,
        binary_name=binary,
        unary_preset=unary_preset,
        binary_preset="minimal",
        quantization_nbins=None,
        quantization_method=None,
        quantization_dtype=np.int32,
    )


def test_build_unary_binary_recipe_mul():
    r = _build_recipe_via_actual_api("mul", "identity", "identity")
    df = _basic_df()
    out = apply_recipe(r, df)
    np.testing.assert_array_equal(np.asarray(out), df["a"].to_numpy() * df["b"].to_numpy())


def test_build_unary_binary_recipe_add():
    r = _build_recipe_via_actual_api("add", "identity", "identity")
    df = _basic_df()
    out = apply_recipe(r, df)
    np.testing.assert_array_equal(np.asarray(out), df["a"].to_numpy() + df["b"].to_numpy())


def test_build_unary_binary_recipe_max():
    # 'sub' is not a 'minimal' preset binary; 'max' is. We exercise the same code path with a
    # binary that actually lives in the preset so the replay succeeds end-to-end. Pre-fix this
    # test used 'sub' and pytest.skip'd on KeyError, masking that the call never replayed.
    r = _build_recipe_via_actual_api("max", "identity", "identity")
    df = _basic_df()
    out = apply_recipe(r, df)
    np.testing.assert_array_equal(np.asarray(out), np.maximum(df["a"].to_numpy(), df["b"].to_numpy()))


def test_build_unary_binary_recipe_with_log_unary():
    # 'log' only lives in 'standard'/'maximal' presets; build with the matching preset so the
    # apply_recipe replay finds the unary function.
    r = _build_recipe_via_actual_api("mul", "log", "identity", unary_preset="standard")
    df = _basic_df()
    out = apply_recipe(r, df)
    assert out is not None and len(out) == len(df)


# ----------------------------------------------------------------------------
# apply_recipe(kind="factorize")
# ----------------------------------------------------------------------------

def test_apply_factorize_seen_pair_codes():
    """Pair factorize: lookup_table maps (a, b)-pre-prune code -> post-prune class.

    Pre-fix this test used a single-column factorize with ``lookup={value: code}`` and pytest.skip
    if the format differed. Production factorize is strictly pairwise with an integer lookup_table
    keyed by ``a + b * nbins_a``. We construct that shape directly and assert the replay.
    """
    nbins_a, nbins_b = 3, 2
    # Pre-prune code map: (a=0,b=0) -> class 0, (a=1,b=0) -> 1, (a=2,b=0) -> 2,
    #                     (a=0,b=1) -> 3, (a=1,b=1) -> 4, (a=2,b=1) -> 5.
    lookup_table = np.arange(nbins_a * nbins_b, dtype=np.int64)
    r = EngineeredRecipe(
        name="ab_factorized",
        kind="factorize",
        src_names=("a", "b"),
        factorize_nbins=(nbins_a, nbins_b),
        unknown_strategy="clip",
        extra={"lookup_table": lookup_table},
    )
    df = pd.DataFrame({"a": [0, 1, 2, 0], "b": [0, 0, 1, 1]})
    out = apply_recipe(r, df)
    # Codes = a + b * nbins_a = [0+0*3, 1+0*3, 2+1*3, 0+1*3] = [0, 1, 5, 3]
    np.testing.assert_array_equal(np.asarray(out, dtype=np.int64), [0, 1, 5, 3])


def test_apply_factorize_clip_out_of_range():
    """Out-of-range pair value is clipped to nbins-1 per unknown_strategy='clip'."""
    nbins_a, nbins_b = 2, 2
    lookup_table = np.array([10, 20, 30, 40], dtype=np.int64)
    r = EngineeredRecipe(
        name="ab_factorized",
        kind="factorize",
        src_names=("a", "b"),
        factorize_nbins=(nbins_a, nbins_b),
        unknown_strategy="clip",
        extra={"lookup_table": lookup_table},
    )
    df = pd.DataFrame({"a": [0, 5], "b": [0, 1]})  # a=5 must be clipped to 1
    out = apply_recipe(r, df)
    # Row0: code = 0+0*2=0 -> 10; Row1: a clipped to 1, code = 1+1*2=3 -> 40.
    assert list(np.asarray(out, dtype=np.int64)) == [10, 40]


# ----------------------------------------------------------------------------
# Helper coverage (if exported)
# ----------------------------------------------------------------------------

def test_coerce_to_int_with_nan_handling():
    if _coerce_to_int_with_nan_handling is None:
        pytest.skip("_coerce_to_int_with_nan_handling not exported")
    # Real signature: (vals, n_bins, recipe_name, col_name, unknown_strategy)
    arr = np.array([1.0, 2.0, np.nan, 3.0])
    out = _coerce_to_int_with_nan_handling(arr, 4, "test_recipe", "test_col", "clip")
    assert out is not None
    assert out.dtype.kind == "i"


def test_coerce_to_int_unknown_strategy_raise():
    if _coerce_to_int_with_nan_handling is None:
        pytest.skip("_coerce_to_int_with_nan_handling not exported")
    arr = np.array([1.0, np.nan], dtype=np.float64)
    with pytest.raises(ValueError, match="NaN"):
        _coerce_to_int_with_nan_handling(arr, 4, "r", "c", "raise")


def test_coerce_to_int_integer_passthrough():
    if _coerce_to_int_with_nan_handling is None:
        pytest.skip("_coerce_to_int_with_nan_handling not exported")
    arr = np.array([0, 1, 2], dtype=np.int32)
    out = _coerce_to_int_with_nan_handling(arr, 4, "r", "c", "clip")
    assert out.dtype == np.int64


def test_extract_column_pandas():
    if _extract_column is None:
        pytest.skip("_extract_column not exported")
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = _extract_column(df, "a")
    np.testing.assert_array_equal(np.asarray(out), [1, 2, 3])


def test_extract_column_polars():
    pl = pytest.importorskip("polars")
    if _extract_column is None:
        pytest.skip("_extract_column not exported")
    pldf = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = _extract_column(pldf, "a")
    np.testing.assert_array_equal(np.asarray(out), [1, 2, 3])


def test_extra_equal_helper():
    if _extra_equal is None:
        pytest.skip("_extra_equal not exported")
    assert _extra_equal({"a": 1}, {"a": 1}) is True
    assert _extra_equal({"a": 1}, {"a": 2}) is False
    assert _extra_equal({}, {}) is True


def test_apply_target_encoding_missing_extra_raises():
    """When recipe.extra is missing cell_means / factorize_lookup, _apply_target_encoding raises a clear KeyError."""
    if _apply_target_encoding is None:
        pytest.skip("_apply_target_encoding not exported")
    r = EngineeredRecipe(
        name="te_pair",
        kind="target_encoding",
        src_names=("a", "b"),
        factorize_nbins=(4, 4),
    )
    df = _basic_df()
    with pytest.raises(KeyError, match=r"cell_means|factorize_lookup"):
        _apply_target_encoding(r, df)


def test_apply_target_encoding_k_gt_2_raises():
    """Target encoding for k > 2 is not implemented; raises NotImplementedError."""
    if _apply_target_encoding is None:
        pytest.skip("_apply_target_encoding not exported")
    r = EngineeredRecipe(
        name="te_triplet",
        kind="target_encoding",
        src_names=("a", "b", "c"),
    )
    df = _basic_df()
    df["c"] = [5, 6, 7, 8]
    with pytest.raises(NotImplementedError, match="k>2"):
        _apply_target_encoding(r, df)


# ----------------------------------------------------------------------------
# Smoke: every recipe kind survives apply
# ----------------------------------------------------------------------------

def test_apply_recipe_unknown_kind_raises():
    r = EngineeredRecipe(name="bogus", kind="nonexistent_kind_xyz", src_names=("a",), extra={})
    df = _basic_df()
    with pytest.raises((ValueError, KeyError, NotImplementedError, TypeError)):
        apply_recipe(r, df)
