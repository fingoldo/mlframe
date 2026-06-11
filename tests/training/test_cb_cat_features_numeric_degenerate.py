"""Regression sensor: injected constant / all-NaN NUMERIC columns must never be
resolved as CatBoost categorical features.

Masked-bug provenance: the fuzz suite once collapsed
``inject_degenerate_cols=True`` to ``False`` on the CB + multilabel path because a
``num_const`` / ``num_null`` float column was suspected of being auto-detected as a
CatBoost categorical feature (crash / mis-train). These tests pin the production
guarantee that fixes it: cat-feature detection is dtype-gated on string/category
dtypes only (pandas: ``PANDAS_CATEGORICAL_DTYPES``; polars:
``is_polars_categorical``), so a numeric float column can never enter
``cat_features``, and the CB Pool is built with that explicit numeric-safe list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.pipeline import fit_and_transform_pipeline
from mlframe.training.configs import PreprocessingBackendConfig


def _frame(n: int = 200, *, with_cat: bool, polars: bool):
    rng = np.random.default_rng(0)
    data = {
        "f0": rng.normal(size=n).astype("float32"),
        "f1": rng.normal(size=n).astype("float32"),
        # The degenerate numeric columns the fuzz axis injects.
        "num_const": np.full(n, 7.5, dtype="float32"),
        "num_null": np.full(n, np.nan, dtype="float32"),
    }
    if with_cat:
        data["cat0"] = rng.choice(["a", "b", "c"], size=n)
    df = pl.DataFrame(data) if polars else pd.DataFrame(data)
    if with_cat and polars:
        df = df.with_columns(pl.col("cat0").cast(pl.Categorical))
    elif with_cat:
        df = df.astype({"cat0": "category"})
    return df


@pytest.mark.parametrize("with_cat", [False, True])
@pytest.mark.parametrize("polars", [False, True])
def test_numeric_degenerate_cols_excluded_from_cat_features(with_cat, polars):
    df = _frame(with_cat=with_cat, polars=polars)
    # skip_categorical_encoding=True keeps cat_features un-encoded through to the
    # CB-native boundary -- the exact path the masked CB+multilabel combo took
    # (CB consumes raw categoricals; ordinal/onehot would empty the list instead).
    config = PreprocessingBackendConfig(prefer_polarsds=polars, skip_categorical_encoding=True)
    _tr, _v, _t, _pipe, cat_features = fit_and_transform_pipeline(
        df, None, None, config=config, ensure_float32=False, verbose=0,
    )
    # Numeric (incl. constant / all-NaN) columns must never be resolved as
    # categorical, regardless of whether a genuine cat column is present or
    # whether the default encoder later empties the list.
    assert "num_const" not in cat_features
    assert "num_null" not in cat_features
    assert "f0" not in cat_features and "f1" not in cat_features


def test_cb_pool_builds_and_fits_with_numeric_degenerate_cols():
    """CB Pool built with the resolved (numeric-safe) cat_features fits without
    mis-typing num_const / num_null as categorical -- the end-to-end guarantee the
    masked fuzz combo exercised."""
    Pool = pytest.importorskip("catboost").Pool
    from catboost import CatBoostClassifier

    df = _frame(n=200, with_cat=True, polars=False)
    config = PreprocessingBackendConfig(prefer_polarsds=False, skip_categorical_encoding=True)
    train_df, _v, _t, _pipe, cat_features = fit_and_transform_pipeline(
        df, None, None, config=config, ensure_float32=False, verbose=0,
    )
    assert "num_const" not in cat_features and "num_null" not in cat_features

    y = (np.asarray(train_df["f0"]) > 0).astype("float32")
    pool = Pool(data=train_df, label=y, cat_features=list(cat_features) or None)
    # The Pool's categorical indices must not include the numeric degenerate cols.
    cols = list(train_df.columns)
    cat_idx = set(pool.get_cat_feature_indices())
    assert cols.index("num_const") not in cat_idx
    assert cols.index("num_null") not in cat_idx
    CatBoostClassifier(iterations=10, verbose=False).fit(pool)
