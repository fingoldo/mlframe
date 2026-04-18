"""Tests for the stale-cache detector in train_eval.py.

Background
----------
The suite caches trained models on disk. Older cached models can become
schema-incompatible with the current preprocessing (e.g. a column that
used to be numeric is now Polars `Categorical`). Loading such a model
and calling `predict_proba` produces a cryptic
``CatBoostError: Unsupported data type Categorical for a numerical feature column``
deep inside CatBoost's pyx layer.

The detector — `_validate_cached_model_schema` — runs as a pre-flight
check right after load and returns a human-readable reason string if
the cached model's feature_names / cat_features don't match the current
DataFrame. A non-None return causes the suite to invalidate the cache
and retrain.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.train_eval import (
    _extract_polars_cat_columns,
    _validate_cached_model_schema,
)


# ---------------------------------------------------------------------------
# Fakes — we don't need a real CatBoost/XGB instance, just objects that
# expose the attributes the validator introspects.
# ---------------------------------------------------------------------------


class _FakeSklearnModel:
    """Minimal sklearn-shaped model: exposes `feature_names_in_`."""

    def __init__(self, feature_names):
        self.feature_names_in_ = np.asarray(list(feature_names))


class _FakeCatBoost:
    """Minimal CatBoost-shaped model: exposes `feature_names_` and
    `_get_cat_feature_indices()`. Enough to drive the validator."""

    def __init__(self, feature_names, cat_feature_indices):
        self.feature_names_ = list(feature_names)
        self._cat_indices = list(cat_feature_indices)

    def _get_cat_feature_indices(self):
        return list(self._cat_indices)


def _wrap(model):
    return SimpleNamespace(model=model, pre_pipeline=None)


# ---------------------------------------------------------------------------
# _extract_polars_cat_columns
# ---------------------------------------------------------------------------

class TestExtractPolarsCatColumns:
    def test_none_df_returns_empty(self):
        assert _extract_polars_cat_columns(None) == []

    def test_pandas_df_returns_empty(self):
        # Pandas DF has no pl.Categorical concept — validator should return []
        df = pd.DataFrame({"a": [1, 2, 3]}).astype({"a": "category"})
        assert _extract_polars_cat_columns(df) == []

    def test_polars_categorical_detected(self):
        df = pl.DataFrame({
            "num": pl.Series("num", [1, 2, 3], dtype=pl.Int32),
            "cat": pl.Series("cat", ["x", "y", "x"], dtype=pl.Categorical),
        })
        assert _extract_polars_cat_columns(df) == ["cat"]

    def test_polars_enum_detected(self):
        enum_t = pl.Enum(["a", "b"])
        df = pl.DataFrame({
            "num": pl.Series("num", [1, 2, 3], dtype=pl.Int32),
            "en":  pl.Series("en", ["a", "b", "a"], dtype=enum_t),
        })
        assert _extract_polars_cat_columns(df) == ["en"]


# ---------------------------------------------------------------------------
# _validate_cached_model_schema — feature-name checks
# ---------------------------------------------------------------------------

class TestFeatureNamesCheck:
    def test_none_model_returns_none(self):
        # Loaded model wrapper where the inner `model` is None: validator has
        # nothing to check → returns None (no mismatch).
        wrap = SimpleNamespace(model=None)
        assert _validate_cached_model_schema(wrap, pl.DataFrame({"a": [1]})) is None

    def test_model_without_feature_names_returns_none(self):
        # Plain object with no feature_names_*: validator cannot conclude
        # mismatch, returns None rather than false-positive invalidation.
        wrap = SimpleNamespace(model=object(), pre_pipeline=None)
        assert _validate_cached_model_schema(wrap, pl.DataFrame({"a": [1]})) is None

    def test_exact_match_returns_none(self):
        m = _FakeSklearnModel(["a", "b", "c"])
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        assert _validate_cached_model_schema(_wrap(m), df) is None

    def test_different_columns_returns_reason(self):
        m = _FakeSklearnModel(["a", "b", "c"])
        df = pd.DataFrame({"a": [1], "b": [2], "d": [3]})  # `c` → `d`
        reason = _validate_cached_model_schema(_wrap(m), df)
        assert reason is not None
        assert "feature-name mismatch" in reason

    def test_reordered_columns_returns_reason(self):
        m = _FakeSklearnModel(["a", "b", "c"])
        df = pd.DataFrame({"b": [1], "a": [2], "c": [3]})  # same set, different order
        reason = _validate_cached_model_schema(_wrap(m), df)
        assert reason is not None
        assert "order" in reason

    def test_extra_column_in_current_df(self):
        m = _FakeSklearnModel(["a", "b"])
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        reason = _validate_cached_model_schema(_wrap(m), df)
        assert reason is not None and "mismatch" in reason


# ---------------------------------------------------------------------------
# _validate_cached_model_schema — CatBoost cat_features cross-check
# ---------------------------------------------------------------------------

class TestCatBoostCatFeaturesCheck:
    """Reproduce the production bug: a Polars Categorical column in the
    current df that the saved CatBoost model never learned as a
    cat_feature (was a numeric column at train time).
    """

    def test_matching_cat_features_returns_none(self):
        # Saved model: cat_features = ['b'] at index 1.
        m = _FakeCatBoost(feature_names=["a", "b", "c"], cat_feature_indices=[1])
        df = pl.DataFrame({
            "a": pl.Series("a", [1.0, 2.0], dtype=pl.Float32),
            "b": pl.Series("b", ["x", "y"], dtype=pl.Categorical),
            "c": pl.Series("c", [10, 20], dtype=pl.Int32),
        })
        assert _validate_cached_model_schema(_wrap(m), df) is None

    def test_new_categorical_not_in_saved_model(self):
        # Saved model had cat_features=['b'] only. Now column 'c' is also
        # Polars Categorical — CatBoost would crash at predict_proba.
        m = _FakeCatBoost(feature_names=["a", "b", "c"], cat_feature_indices=[1])
        df = pl.DataFrame({
            "a": pl.Series("a", [1.0, 2.0], dtype=pl.Float32),
            "b": pl.Series("b", ["x", "y"], dtype=pl.Categorical),
            "c": pl.Series("c", ["p", "q"], dtype=pl.Categorical),  # was Int32 at train
        })
        reason = _validate_cached_model_schema(_wrap(m), df)
        assert reason is not None
        assert "CatBoost cache mismatch" in reason
        assert "'c'" in reason

    def test_saved_cat_features_dropped_from_current(self):
        # Saved model expected 'b' to be categorical. Current df has 'b' as
        # non-Categorical (e.g. numeric). This is the OPPOSITE mismatch —
        # the validator allows it because the feature-name check already
        # catches this scenario if columns differ, and if the columns match
        # but dtype narrowed, the backend can still handle it (numeric
        # where cat was expected tends to be tolerated more gracefully).
        m = _FakeCatBoost(feature_names=["a", "b"], cat_feature_indices=[1])
        df = pl.DataFrame({
            "a": pl.Series("a", [1.0, 2.0], dtype=pl.Float32),
            "b": pl.Series("b", [10, 20], dtype=pl.Int32),
        })
        # Columns match exactly; no *new* Polars Categorical column appears.
        # The validator intentionally does not flag this direction.
        assert _validate_cached_model_schema(_wrap(m), df) is None

    def test_cat_indices_out_of_range_flags_mismatch(self):
        # Pathological saved model where _get_cat_feature_indices yields
        # indices past feature_names_. The index-range guard silently
        # drops those, leaving the resolvable cat set empty — equivalent
        # to "saved model has no cat_features". Since the current df DOES
        # have a Polars Categorical column, predict would still crash,
        # so the validator correctly flags mismatch.
        m = _FakeCatBoost(feature_names=["a", "b"], cat_feature_indices=[99])
        df = pl.DataFrame({
            "a": pl.Series("a", [1.0, 2.0], dtype=pl.Float32),
            "b": pl.Series("b", ["x", "y"], dtype=pl.Categorical),
        })
        reason = _validate_cached_model_schema(_wrap(m), df)
        assert reason is not None
        assert "CatBoost cache mismatch" in reason
        assert "'b'" in reason

    def test_saved_model_without_cat_features_and_no_polars_cats(self):
        # Saved model has no cat_features AND current df has no Polars
        # Categoricals — perfectly compatible, no mismatch.
        m = _FakeCatBoost(feature_names=["a", "b"], cat_feature_indices=[])
        df = pl.DataFrame({
            "a": pl.Series("a", [1.0, 2.0], dtype=pl.Float32),
            "b": pl.Series("b", [10, 20], dtype=pl.Int32),
        })
        assert _validate_cached_model_schema(_wrap(m), df) is None

    def test_pandas_df_does_not_false_positive(self):
        # Current df is pandas with a category-dtype column. CatBoost's
        # Polars fastpath is not engaged for pandas dfs, so the validator
        # must not flag this as a mismatch based on Polars Categoricals.
        m = _FakeCatBoost(feature_names=["a", "b"], cat_feature_indices=[1])
        df = pd.DataFrame({"a": [1.0], "b": pd.Categorical(["x"])})
        assert _validate_cached_model_schema(_wrap(m), df) is None
