"""Regression test for the iter#42 finding:

prepare_dfs_for_catboost_joint received an embedding column (pandas
object-dtype Series of ndarrays) listed in cat_features. The
``_stringify`` step called .astype("string") on it, which:
1. spent ~32s per call on 800K rows calling repr() on each ndarray
2. then crashed with ``TypeError: unhashable type: 'numpy.ndarray'`` inside
   ``set(.unique().tolist())`` since ndarrays aren't hashable.

Post-fix: detect the embedding-shape first cell (hasattr 'shape' or non-
str iterable) and skip with a WARN instead of crashing.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline import prepare_dfs_for_catboost_joint


def test_skip_embedding_column_in_joint_cat_cast(caplog):
    """Embedding column listed in cat_features must be skipped, not crashed-on."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "cat_low": np.array(["A", "B", "C"], dtype=object)[rng.integers(0, 3, n)],
            "emb": [rng.normal(size=4).astype("float32") for _ in range(n)],
        }
    )

    with caplog.at_level(logging.WARNING, logger="mlframe.training.pipeline"):
        # Both real cat + the mis-classified embedding column passed in. The
        # function must process cat_low normally and skip emb with a WARN.
        prepare_dfs_for_catboost_joint(
            train_df=df,
            val_df=None,
            test_df=None,
            cat_features=["cat_low", "emb"],
        )

    # cat_low became a proper Categorical
    assert pd.api.types.is_categorical_dtype(df["cat_low"])
    # emb remains object-dtype with ndarray values - untouched
    assert df["emb"].dtype == object
    assert hasattr(df["emb"].iloc[0], "shape")
    # WARN fired
    assert any("looks like an embedding/list column" in rec.message for rec in caplog.records), "expected WARN about skipping embedding column"


def test_normal_cat_column_still_cast():
    """Sanity: regular cat_features still get joint-Categorical cast."""
    rng = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame(
        {
            "cat": np.array(["A", "B", "C", "D"], dtype=object)[rng.integers(0, 4, n)],
        }
    )
    val = pd.DataFrame(
        {
            "cat": np.array(["B", "C", "D", "E"], dtype=object)[rng.integers(0, 4, 50)],
        }
    )
    test = pd.DataFrame(
        {
            "cat": np.array(["X", "Y", "Z"], dtype=object)[rng.integers(0, 3, 30)],
        }
    )

    prepare_dfs_for_catboost_joint(
        train_df=train,
        val_df=val,
        test_df=test,
        cat_features=["cat"],
    )

    # Joint dtype includes train + val values
    expected = sorted(set(["A", "B", "C", "D", "E"]))
    assert list(train["cat"].cat.categories) == expected
    # Test column shares same dtype; out-of-set values become null codes
    assert train["cat"].dtype == val["cat"].dtype == test["cat"].dtype
    assert test["cat"].isna().all()
