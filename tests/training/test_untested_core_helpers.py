"""Tests for small untested helpers in mlframe.training.core.

Covers:
- _df_shape_str, _elapsed_str, _drop_cols_df
- _validate_feature_type_exclusivity
- _auto_detect_feature_types
- _build_tier_dfs
- _convert_dfs_to_pandas
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core import (
    _auto_detect_feature_types,
    _build_tier_dfs,
    _convert_dfs_to_pandas,
    _df_shape_str,
    _drop_cols_df,
    _elapsed_str,
    _validate_feature_type_exclusivity,
)
from mlframe.training.configs import FeatureTypesConfig


# ----- _df_shape_str / _elapsed_str -----

def test_df_shape_str_pandas():
    df = pd.DataFrame({"a": range(1234), "b": range(1234)})
    out = _df_shape_str(df)
    assert "1_234" in out
    assert "2" in out


def test_df_shape_str_polars():
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert _df_shape_str(df) == "3×1"


def test_df_shape_str_none():
    assert _df_shape_str(None) == "None"


def test_elapsed_str_seconds():
    t0 = time.perf_counter() - 0.5
    out = _elapsed_str(t0)
    assert out.endswith("s")
    assert not out.endswith("min")


def test_elapsed_str_minutes():
    t0 = time.perf_counter() - 125.0
    out = _elapsed_str(t0)
    assert out.endswith("min")


# ----- _drop_cols_df -----

def test_drop_cols_pandas():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    out = _drop_cols_df(df, ["b", "missing"])
    assert list(out.columns) == ["a", "c"]


def test_drop_cols_polars():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = _drop_cols_df(df, ["b"])
    assert out.columns == ["a"]


def test_drop_cols_empty_list_noop():
    df = pd.DataFrame({"a": [1]})
    assert _drop_cols_df(df, []) is df
    assert _drop_cols_df(df, None) is df


def test_drop_cols_all_missing_noop():
    df = pd.DataFrame({"a": [1]})
    out = _drop_cols_df(df, ["x", "y"])
    assert out is df


# ----- _validate_feature_type_exclusivity -----

def test_exclusivity_ok():
    # no overlap -> returns None
    assert _validate_feature_type_exclusivity(["t1"], ["e1"], ["c1"]) is None


def test_exclusivity_text_cat_overlap():
    with pytest.raises(ValueError, match="text_features and cat_features"):
        _validate_feature_type_exclusivity(["a"], [], ["a"])


def test_exclusivity_embedding_cat_overlap():
    with pytest.raises(ValueError, match="embedding_features and cat_features"):
        _validate_feature_type_exclusivity([], ["a"], ["a"])


def test_exclusivity_text_embedding_overlap():
    with pytest.raises(ValueError, match="text_features and embedding_features"):
        _validate_feature_type_exclusivity(["a"], ["a"], [])


# ----- _auto_detect_feature_types -----

def test_auto_detect_disabled_returns_user_lists():
    cfg = FeatureTypesConfig(auto_detect_feature_types=False,
                             text_features=["t"], embedding_features=["e"])
    df = pd.DataFrame({"t": ["x"], "e": [[1.0]]})
    t, e = _auto_detect_feature_types(df, cfg, cat_features=[])
    assert t == ["t"]
    assert e == ["e"]


def test_auto_detect_pandas_high_cardinality_text():
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=3)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "low_card": rng.choice(["A", "B"], size=50),
        "high_card": [f"s_{i}" for i in range(50)],
        "num": rng.standard_normal(50),
    })
    t, e = _auto_detect_feature_types(df, cfg, cat_features=[])
    assert "high_card" in t
    assert "low_card" not in t
    assert "num" not in t


def test_auto_detect_pandas_skips_cat_features():
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=2)
    df = pd.DataFrame({"c": [f"v_{i}" for i in range(20)]})
    t, e = _auto_detect_feature_types(df, cfg, cat_features=["c"])
    assert "c" not in t


def test_auto_detect_polars_embedding():
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=100)
    df = pl.DataFrame({
        "emb": [[1.0, 2.0], [3.0, 4.0]],
        "num": [1.0, 2.0],
    })
    t, e = _auto_detect_feature_types(df, cfg, cat_features=[])
    assert "emb" in e


def test_auto_detect_polars_high_card_text():
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=3)
    df = pl.DataFrame({"s": [f"v_{i}" for i in range(10)]})
    t, e = _auto_detect_feature_types(df, cfg, cat_features=[])
    assert "s" in t


# ----- _build_tier_dfs -----

class _Strategy:
    def __init__(self, text=True, emb=True):
        self.supports_text_features = text
        self.supports_embedding_features = emb

    def feature_tier(self):
        return (self.supports_text_features, self.supports_embedding_features)


def test_build_tier_dfs_full_support_no_copy():
    base = {
        "train_df": pd.DataFrame({"a": [1], "t": ["x"], "e": [[1.0]]}),
        "val_df": None,
        "test_df": None,
    }
    cache = {}
    out = _build_tier_dfs(base, _Strategy(True, True), ["t"], ["e"], cache)
    assert out is base  # no copy performed
    assert cache  # cached


def test_build_tier_dfs_drops_unsupported_pandas():
    base = {
        "train_df": pd.DataFrame({"a": [1], "t": ["x"], "e": [[1.0]]}),
        "val_df": pd.DataFrame({"a": [2], "t": ["y"], "e": [[2.0]]}),
        "test_df": None,
    }
    cache = {}
    out = _build_tier_dfs(base, _Strategy(False, False), ["t"], ["e"], cache)
    assert "t" not in out["train_df"].columns
    assert "e" not in out["train_df"].columns
    assert "a" in out["train_df"].columns
    assert out["test_df"] is None


def test_build_tier_dfs_drops_unsupported_polars():
    base = {
        "train_df": pl.DataFrame({"a": [1], "t": ["x"]}),
        "val_df": None,
        "test_df": None,
    }
    cache = {}
    out = _build_tier_dfs(base, _Strategy(False, True), ["t"], [], cache)
    assert "t" not in out["train_df"].columns


def test_build_tier_dfs_cache_hit():
    base = {"train_df": pd.DataFrame({"a": [1]}), "val_df": None, "test_df": None}
    strat = _Strategy(True, True)
    cache = {}
    out1 = _build_tier_dfs(base, strat, [], [], cache)
    out2 = _build_tier_dfs(base, strat, [], [], cache)
    assert out1 is out2


# ----- _convert_dfs_to_pandas -----

def test_convert_dfs_pandas_passthrough():
    df = pd.DataFrame({"a": [1, 2]})
    tr, va, te = _convert_dfs_to_pandas(df, None, None)
    assert tr is df
    assert va is None and te is None


def test_convert_dfs_polars_to_pandas():
    pdf = pl.DataFrame({"a": [1, 2]})
    tr, va, te = _convert_dfs_to_pandas(pdf, pdf, None)
    assert isinstance(tr, pd.DataFrame)
    assert isinstance(va, pd.DataFrame)


def test_convert_dfs_invalid_type():
    with pytest.raises(TypeError, match="train_df"):
        _convert_dfs_to_pandas(np.array([1, 2, 3]), None, None)
