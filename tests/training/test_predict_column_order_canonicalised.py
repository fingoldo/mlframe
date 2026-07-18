"""Regression: the inference boundary must canonicalise serving-frame column ORDER to the trained schema.

A predict frame that carries all trained columns but in a different order must not fail or mis-map: sklearn-API
estimators raise on a same-names-different-order frame, and positional consumers would silently mis-map. The
``_validate_input_columns_against_metadata`` boundary reorders to the schema order for both pandas and polars even
when there are no extra columns to drop. Pre-fix the reorder only ran on the extra-columns path, so a benignly
reordered frame crashed the whole predict call with ``all N supplied model(s) failed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.core._misc_helpers import _validate_input_columns_against_metadata
from mlframe.training.core._predict_main_from_models import predict_from_models

_META = {
    "raw_input_columns": ["a", "b", "c"],
    "cat_features": [],
    "text_features": [],
    "embedding_features": [],
}


@pytest.mark.parametrize("flavour", ["pandas", "polars"])
def test_validate_reorders_to_schema_order_without_extra_cols(flavour):
    """Validate reorders to schema order without extra cols."""
    data = {"c": [1, 2], "b": [3, 4], "a": [5, 6]}
    df = pd.DataFrame(data) if flavour == "pandas" else pl.DataFrame(data)
    out = _validate_input_columns_against_metadata(df, _META, verbose=False)
    assert list(out.columns) == ["a", "b", "c"]


@pytest.mark.parametrize("flavour", ["pandas", "polars"])
def test_validate_leaves_already_ordered_frame_untouched(flavour):
    """Validate leaves already ordered frame untouched."""
    data = {"a": [1], "b": [2], "c": [3]}
    df = pd.DataFrame(data) if flavour == "pandas" else pl.DataFrame(data)
    out = _validate_input_columns_against_metadata(df, _META, verbose=False)
    assert list(out.columns) == ["a", "b", "c"]
    assert out is df  # no-op when order already matches: no reselect


@pytest.mark.parametrize("flavour", ["pandas", "polars"])
def test_validate_missing_col_keeps_present_in_schema_order(flavour):
    """Validate missing col keeps present in schema order."""
    data = {"c": [1], "a": [2]}
    df = pd.DataFrame(data) if flavour == "pandas" else pl.DataFrame(data)
    out = _validate_input_columns_against_metadata(df, _META, verbose=False)
    assert list(out.columns) == ["a", "c"]


class _ModelWrapper:
    """Groups tests covering model wrapper."""
    def __init__(self, model):
        self.model = model
        self.pre_pipeline = None


def _fit_linear():
    """Fit linear."""
    from sklearn.linear_model import LinearRegression

    X = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 2, "c": np.arange(20.0) * 3})
    y = X["a"] + 5.0
    return X, LinearRegression().fit(X, y)


@pytest.mark.parametrize("flavour", ["pandas", "polars"])
def test_predict_from_models_reordered_input_matches_schema_order(flavour):
    """Predict from models reordered input matches schema order."""
    X, model = _fit_linear()
    meta = {
        "raw_input_columns": ["a", "b", "c"],
        "columns": ["a", "b", "c"],
        "cat_features": [],
        "text_features": [],
        "embedding_features": [],
    }
    models = {"regression": {"t": [_ModelWrapper(model)]}}
    base = predict_from_models(X.copy(), models, meta, return_probabilities=False, verbose=0)["predictions"]["regression_t"]

    reordered = X[["c", "b", "a"]]
    if flavour == "polars":
        reordered = pl.from_pandas(reordered)
    out = predict_from_models(reordered, models, meta, return_probabilities=False, verbose=0)
    assert "regression_t" in out["predictions"], "reordered serving frame must not fail the predict"
    np.testing.assert_allclose(out["predictions"]["regression_t"], base)
