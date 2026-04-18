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


def test_exclusivity_accepts_none_args():
    """Regression sensor: validator used to crash with
    ``TypeError: 'NoneType' object is not iterable`` when any of its
    three arguments was ``None``. Callers pass None for feature-type
    lists that were skipped (e.g. a model without categorical support
    passed None instead of []).
    """
    _validate_feature_type_exclusivity(None, None, None)
    _validate_feature_type_exclusivity(None, [], ["a"])
    _validate_feature_type_exclusivity(["a"], None, [])
    _validate_feature_type_exclusivity([], ["b"], None)


def test_exclusivity_none_still_catches_real_overlap():
    """None args must not silence a real overlap — if one list is None
    and the other two still overlap, the validator must still raise.
    """
    with pytest.raises(ValueError, match="text_features and cat_features"):
        _validate_feature_type_exclusivity(["a"], None, ["a"])


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


def test_auto_detect_pandas_promotes_high_card_cat_to_text():
    """A column in cat_features with cardinality above the threshold is
    promoted to text_features. Contract change 2026-04-19: the function
    no longer mutates ``cat_features``; the caller does the filtering via
    set-difference (see ``effective_cat_features`` in the suite). Previous
    in-place mutation was dead code (caller already filtered) but a latent
    trap for future reuse of a shared list.
    """
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=2)
    df = pd.DataFrame({"c": [f"v_{i}" for i in range(20)]})
    cat_features = ["c"]
    t, e = _auto_detect_feature_types(df, cfg, cat_features=cat_features)
    assert "c" in t, "high-cardinality column must be promoted to text"
    # Contract: input list is NOT mutated. The caller filters separately.
    assert cat_features == ["c"], (
        "cat_features must not be mutated in place — that was a latent "
        "state-leak trap on repeat calls with a shared list"
    )


def test_auto_detect_pandas_keeps_low_card_cat():
    """A column in cat_features with cardinality BELOW the threshold stays
    in cat (is not promoted)."""
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=100)
    df = pd.DataFrame({"c": ["red", "green", "blue"] * 10})  # n_unique = 3
    cat_features = ["c"]
    t, e = _auto_detect_feature_types(df, cfg, cat_features=cat_features)
    assert "c" not in t
    assert "c" in cat_features


@pytest.mark.parametrize("n_unique, threshold, expected_promoted", [
    (9,  10, False),   # strictly below  -> stays as cat
    (10, 10, False),   # exactly at     -> stays (promotion uses `>` not `>=`)
    (11, 10, True),    # strictly above  -> promoted
])
def test_auto_detect_threshold_boundary(n_unique, threshold, expected_promoted):
    """Exercises the promotion threshold boundary. Catches off-by-one
    regressions (>= vs >) and accidental strict-vs-non-strict flips.
    """
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=threshold)
    df = pd.DataFrame({"c": [f"v_{i:04d}" for i in range(n_unique)]})
    cat_features = ["c"]
    t, _ = _auto_detect_feature_types(df, cfg, cat_features=cat_features)
    if expected_promoted:
        assert "c" in t
    else:
        assert "c" not in t
    # Input list is NEVER mutated regardless of promotion outcome.
    assert cat_features == ["c"]


def test_auto_detect_does_not_mutate_cat_features_across_calls():
    """Sensor for repeat-call state leak: calling the function twice with
    the same list as input must produce identical results on the second call.
    Pre-fix, the first call mutated ``cat_features`` (removing the promoted
    column), so the second call saw an empty cat list and silently skipped
    promotion tracking — producing a different ``promoted`` log message
    and potentially different downstream behavior.
    """
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=5)
    df = pd.DataFrame({"c": [f"v_{i}" for i in range(20)], "low": ["a", "b"] * 10})
    cat_features = ["c", "low"]

    t1, e1 = _auto_detect_feature_types(df, cfg, cat_features=cat_features)
    # Snapshot the list after the first call — must be unchanged.
    assert cat_features == ["c", "low"], "first call must not mutate cat_features"

    t2, e2 = _auto_detect_feature_types(df, cfg, cat_features=cat_features)
    assert t1 == t2, "second call with identical input must produce identical text_features"
    assert e1 == e2
    assert cat_features == ["c", "low"], "second call must also not mutate"


def test_auto_detect_user_text_wins_over_promotion():
    """When the user explicitly listed a column in text_features, no
    promotion logic runs — the user's decision is authoritative.
    """
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=2,
                             text_features=["c"])
    df = pd.DataFrame({"c": [f"v_{i}" for i in range(20)]})
    cat_features = ["c"]
    t, _ = _auto_detect_feature_types(df, cfg, cat_features=cat_features)
    # The column is in text_features regardless (user declared it so), and
    # must also NOT be in cat_features — user's text declaration takes
    # precedence over a pipeline-derived cat classification.
    assert "c" in t
    # cat_features is only mutated by the `promoted` loop, which populates
    # from cardinality scan. User-pre-declared text cols skip the scan via
    # `already_assigned`, so they never enter `promoted`. Current
    # implementation therefore leaves cat_features unchanged — document
    # that fact so a future refactor that silently auto-removes user-text
    # cols would trip this test.
    assert "c" in cat_features


def test_auto_detect_polars_categorical_promoted_by_cardinality():
    """Polars ``pl.Categorical`` columns with high cardinality must be
    promoted to text_features. Production case: ``skills_text`` came in
    from raw data as ``pl.Categorical`` with 80k+ unique values; before
    the 2026-04-19 fix it stayed in cat_features and CatBoost wasted
    gigabytes on nominal encoding.
    """
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=10)
    df = pl.DataFrame({
        "hc": pl.Series("hc", [f"v_{i}" for i in range(50)]).cast(pl.Categorical),
    })
    cat_features = ["hc"]
    t, _ = _auto_detect_feature_types(df, cfg, cat_features=cat_features)
    assert "hc" in t
    # Input list must NOT be mutated — caller filters via set-difference.
    assert cat_features == ["hc"]


def test_auto_detect_polars_enum_promoted_by_cardinality():
    """``pl.Enum`` is a fixed-domain categorical. Before the 2026-04-19
    discovery its instance-level dtype object didn't match the
    class-level ``pl.Categorical`` check, and high-cardinality Enum
    columns silently stayed in ``cat_features``. Fixed by adding an
    explicit ``isinstance(dtype, pl.Enum)`` branch; this test is the
    regression sensor.
    """
    enum_t = pl.Enum([f"v_{i:03d}" for i in range(50)])
    df = pl.DataFrame({
        "hc": pl.Series("hc", [f"v_{i:03d}" for i in range(50)], dtype=enum_t),
    })
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=10)
    cat_features = ["hc"]
    t, _ = _auto_detect_feature_types(df, cfg, cat_features=cat_features)
    assert "hc" in t, "pl.Enum column with high cardinality must be promoted to text_features"
    # Input list must NOT be mutated — caller filters via set-difference.
    assert cat_features == ["hc"]


def test_auto_detect_accepts_cat_features_none():
    """Regression sensor: callers sometimes pass ``cat_features=None``
    (e.g. when a model declared no categoricals). The function must
    treat that as an empty list, not crash with
    ``TypeError: argument of type 'NoneType' is not iterable``.
    """
    df = pl.DataFrame({"s": [f"v_{i}" for i in range(10)]})
    cfg = FeatureTypesConfig(auto_detect_feature_types=True,
                             cat_text_cardinality_threshold=3)
    t, e = _auto_detect_feature_types(df, cfg, cat_features=None)
    assert "s" in t


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
