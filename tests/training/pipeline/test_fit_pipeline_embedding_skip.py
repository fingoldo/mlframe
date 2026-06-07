"""Regression test for iter#43:

fit_and_transform_pipeline used to add object-of-ndarray embedding columns
to cat_features (object dtype passes the PANDAS_CATEGORICAL_DTYPES filter
in default sklearn-pipeline path). The downstream
``category_encoders.OrdinalEncoder.fit`` called ``X[col].unique()`` on the
embedding column, which pandas hashes - and ndarrays raise
``TypeError: unhashable type: 'numpy.ndarray'``.

The fix routes embedding-shape columns OUT of cat_features even when
their dtype is plain object. See `_looks_embedding` predicate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import PreprocessingBackendConfig
from mlframe.training.pipeline import fit_and_transform_pipeline


def test_embedding_object_column_excluded_from_cat_features():
    """Pandas object-dtype column with ndarray cells must NOT be passed to
    the categorical encoder - the encoder calls .unique() which hashes
    cells and raises TypeError on ndarray.

    Successful return (no exception) is the contract.
    """
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "x0": rng.normal(size=n).astype("float32"),
        "x1": rng.normal(size=n).astype("float32"),
        # Real categorical (string-object) column - should be encoded
        "cat_low": np.array(["A", "B", "C"], dtype=object)[rng.integers(0, 3, n)],
        # Embedding (object-of-ndarray) - must be skipped
        "emb": [rng.normal(size=4).astype("float32") for _ in range(n)],
    })

    config = PreprocessingBackendConfig(
        categorical_encoding="ordinal",
        skip_categorical_encoding=False,
    )

    # Should NOT raise
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

    # After ordinal encoding, cat_low is numeric; emb is preserved untouched
    # (object-dtype-of-ndarray). The key guarantee: no exception raised.
    assert "emb" in train_out.columns, "embedding column survived the pipeline"
    assert train_out["emb"].dtype == object, "embedding column preserved as object"
    # cat_low got ordinal-encoded -> integer dtype now
    assert np.issubdtype(train_out["cat_low"].dtype, np.integer), (
        f"cat_low should be ordinal-encoded to int, got {train_out['cat_low'].dtype}"
    )


def test_normal_cat_only_input_works():
    """Sanity: when no embedding column is present, categorical encoding
    proceeds as before."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "x0": rng.normal(size=n).astype("float32"),
        "cat_low": np.array(["A", "B", "C"], dtype=object)[rng.integers(0, 3, n)],
    })

    config = PreprocessingBackendConfig(
        categorical_encoding="ordinal",
        skip_categorical_encoding=False,
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
    # cat_low ordinal-encoded to integer
    assert np.issubdtype(train_out["cat_low"].dtype, np.integer)
