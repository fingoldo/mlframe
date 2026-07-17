"""Regression test for iter#44:

_auto_detect_feature_types on a pandas frame with an object-dtype column
holding ndarrays (embedding vectors) used to call df[col].nunique(),
which hashes cells via pandas PyObjectHashTable and raised
``TypeError: unhashable type: 'numpy.ndarray'``.

The fix: pre-check first-cell shape; route ndarray/list cells to
embedding_features and skip the cardinality check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.core._misc_helpers import _auto_detect_feature_types
from mlframe.training.configs import FeatureTypesConfig


def test_pandas_emb_column_routed_to_embedding_features():
    """Object-dtype column with ndarray cells must end up in
    embedding_features, not crash nunique()."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "cat_low": np.array(["A", "B", "C"], dtype=object)[rng.integers(0, 3, n)],
            "emb": [rng.normal(size=4).astype("float32") for _ in range(n)],
        }
    )

    cfg = FeatureTypesConfig()

    text_features, embedding_features, _dropped = _auto_detect_feature_types(
        df=df,
        feature_types_config=cfg,
        cat_features=[],
        verbose=False,
    )

    assert "emb" in embedding_features, f"emb should be routed to embedding_features; got {embedding_features}"
    # cat_low has cardinality 3 (low), should NOT be text-promoted
    assert "cat_low" not in text_features


def test_pandas_no_embedding_column_works():
    """Sanity: pandas auto-detect still works when no embedding column present."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "x0": rng.normal(size=n).astype("float32"),
            "cat_low": np.array(["A", "B", "C"], dtype=object)[rng.integers(0, 3, n)],
        }
    )

    _text_features, embedding_features, _dropped = _auto_detect_feature_types(
        df=df,
        feature_types_config=FeatureTypesConfig(),
        cat_features=[],
        verbose=False,
    )
    assert "emb" not in embedding_features
