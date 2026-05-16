"""Regression test for iter#49:

fit_and_transform_pipeline used to include free-text object-dtype columns
in cat_features when ``skip_categorical_encoding=True`` (the CB-native
path enabled by the auto-flip in _phase_fit_pipeline). The downstream
``prepare_dfs_for_catboost_joint`` then converted the text column to
pandas ``Categorical`` in place. CB Pool construction subsequently
crashed with ``CatBoostError: features data: pandas.DataFrame column
'text_col' has dtype 'category' but is not in cat_features list``
because by Pool-construction time text_col was correctly listed under
text_features (not cat_features).

The fix routes high-cardinality (> 300 sample-unique) object/string
columns OUT of cat_features in the pipeline-fit phase, before the
joint-Categorical cast. See ``_looks_text`` predicate in
``pipeline.fit_and_transform_pipeline``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.configs import PreprocessingBackendConfig
from mlframe.training.pipeline import fit_and_transform_pipeline


def test_text_object_column_excluded_from_cat_features():
    """Pandas object-dtype column with many distinct short-string values
    must NOT be passed to the joint-Categorical cast.

    Successful return (no exception) is the contract.
    """
    rng = np.random.default_rng(0)
    n = 1_500
    _vocab = np.array("alpha beta gamma delta epsilon zeta eta theta iota kappa".split(), dtype=object)
    _idx = rng.integers(0, len(_vocab), (n, 4))
    df = pd.DataFrame({
        "x0": rng.normal(size=n).astype("float32"),
        "x1": rng.normal(size=n).astype("float32"),
        "cat_low": np.array(["A", "B", "C"], dtype=object)[rng.integers(0, 3, n)],
        "text_col": np.array([" ".join(_vocab[r]) for r in _idx], dtype=object),
    })

    config = PreprocessingBackendConfig(
        categorical_encoding="ordinal",
        skip_categorical_encoding=True,
    )

    train_out, val_out, test_out, pipeline, cat_features = fit_and_transform_pipeline(
        train_df=df.copy(),
        val_df=None,
        test_df=None,
        config=config,
        ensure_float32=False,
        verbose=False,
        text_features=[],
        embedding_features=[],
    )

    assert "text_col" not in cat_features, (
        f"text_col (high-card) should not be in cat_features; got {cat_features}"
    )
    assert "cat_low" in cat_features, (
        f"cat_low (low-card) should be in cat_features; got {cat_features}"
    )
    assert train_out["text_col"].dtype == object, (
        f"text_col must remain object dtype (not category); got {train_out['text_col'].dtype}"
    )


def test_low_card_object_column_stays_categorical():
    """Sanity: low-cardinality object columns continue to be treated as cats
    (get joint-Categorical cast for CB native handling)."""
    rng = np.random.default_rng(0)
    n = 1_500
    df = pd.DataFrame({
        "x0": rng.normal(size=n).astype("float32"),
        "cat_low": np.array(["A", "B", "C", "D", "E"], dtype=object)[rng.integers(0, 5, n)],
    })

    config = PreprocessingBackendConfig(
        categorical_encoding="ordinal",
        skip_categorical_encoding=True,
    )

    train_out, _, _, _, cat_features = fit_and_transform_pipeline(
        train_df=df.copy(),
        val_df=None,
        test_df=None,
        config=config,
        ensure_float32=False,
        verbose=False,
        text_features=[],
        embedding_features=[],
    )

    assert "cat_low" in cat_features
    assert pd.api.types.is_categorical_dtype(train_out["cat_low"]), (
        f"cat_low should be cast to Categorical; got {train_out['cat_low'].dtype}"
    )
