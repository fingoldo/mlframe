"""A rejected CatBoost polars predict names the column that changed since fit.

A production predict failed with CatBoost's bare Cython "No matching signature found", fell back to pandas and left the
log unable to say which column or dtype was rejected; no synthetic frame reproduced it. The model now remembers the
dtypes it was fitted on and the fallback warning lists what differs.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from mlframe.training.cb._cb_pool import _polars_schema_drift, _predict_with_fallback
from mlframe.training.cb._cb_pool_build import _stamp_fit_polars_schema


class _Model:
    """A stand-in model that records attributes like a CatBoost estimator does."""


def _frame(i16_dtype=pl.Int16):
    return pl.DataFrame({"a": np.arange(4, dtype=np.float32), "b": pl.Series([1, 2, 3, 4], dtype=i16_dtype)})


def test_fit_schema_is_recorded_as_plain_strings():
    """Plain strings: pickling the model must not depend on polars dtype objects."""
    model = _Model()
    _stamp_fit_polars_schema(model, _frame())
    assert model._mlframe_fit_polars_schema == {"a": "Float32", "b": "Int16"}


def test_a_pandas_fit_frame_records_nothing():
    """Only a polars fit has a polars schema to compare against."""
    import pandas as pd

    model = _Model()
    _stamp_fit_polars_schema(model, pd.DataFrame({"a": [1.0]}))
    assert not hasattr(model, "_mlframe_fit_polars_schema")


def test_drift_names_changed_missing_and_extra_columns():
    model = _Model()
    _stamp_fit_polars_schema(model, _frame())
    now = _frame(pl.Int64).with_columns(pl.lit(1.0).alias("c")).drop("a")
    text = _polars_schema_drift(model, now)
    assert "b: Int16 -> Int64" in text and "missing: a" in text and "not in fit: c" in text


def test_an_unchanged_schema_points_inside_catboost():
    """When nothing differs, say so: the operator then knows not to hunt for a dtype drift."""
    model = _Model()
    _stamp_fit_polars_schema(model, _frame())
    assert "inside CatBoost" in _polars_schema_drift(model, _frame())


def test_without_a_recorded_schema_the_dtype_groups_are_listed():
    """A model loaded from an older artefact has no fit schema; the predict frame's own types still help."""
    assert "Int16" in _polars_schema_drift(_Model(), _frame())


def test_the_fallback_warning_carries_the_drift(caplog):
    """End to end through the real fallback: a rejected polars predict, a pandas retry, and a warning naming the column."""

    class CatBoostClassifier:  # the fallback recognises CatBoost by class name
        classes_ = np.array([0, 1])

        def predict_proba(self, X):
            if isinstance(X, pl.DataFrame):
                raise TypeError("No matching signature found")
            return np.tile([0.5, 0.5], (len(X), 1))

    model = CatBoostClassifier()
    _stamp_fit_polars_schema(model, _frame())
    with caplog.at_level(logging.WARNING):
        out = _predict_with_fallback(model, _frame(pl.Int64), method="predict_proba")
    assert np.asarray(out).shape == (4, 2)
    assert any("b: Int16 -> Int64" in r.getMessage() for r in caplog.records)
