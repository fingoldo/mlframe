"""Regression: ``_auto_detect_feature_types`` must recognise
``pl.Array(...)`` (polars>=0.20 fixed-size embedding dtype) and
``pl.List(pl.Int*)`` (quantized 8-bit / 16-bit embeddings) as
embedding columns, not just ``pl.List(pl.Float32/Float64)``.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def _has_pl_array():
    return hasattr(pl, "Array")


def test_pl_list_int8_detected_as_embedding():
    """Quantized 8-bit embedding stored as List(Int8) -> embedding."""
    from mlframe.training.core._misc_helpers import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    df = pl.DataFrame(
        {
            "user_id": pl.Series(["u1", "u2", "u3"], dtype=pl.Utf8),
            "quant_emb": pl.Series(
                [[1, -2, 3, 4], [-1, 0, 5, -3], [2, 2, 2, 2]],
                dtype=pl.List(pl.Int8),
            ),
        }
    )
    cfg = FeatureTypesConfig(auto_detect_feature_types=True)
    text_features, embedding_features, _ = _auto_detect_feature_types(
        df,
        feature_types_config=cfg,
        cat_features=[],
        verbose=False,
    )
    assert "quant_emb" in embedding_features, f"List(Int8) column not detected as embedding; text={text_features}, emb={embedding_features}"


@pytest.mark.skipif(not _has_pl_array(), reason="pl.Array requires polars>=0.20")
def test_pl_array_float32_detected_as_embedding():
    """Fixed-size pl.Array(Float32, N) -> embedding."""
    from mlframe.training.core._misc_helpers import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    rows = [list(np.random.random(4).astype(np.float32)) for _ in range(3)]
    df = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.Int32),
            "fixed_emb": pl.Series(rows, dtype=pl.Array(pl.Float32, 4)),
        }
    )
    cfg = FeatureTypesConfig(auto_detect_feature_types=True)
    text_features, embedding_features, _ = _auto_detect_feature_types(
        df,
        feature_types_config=cfg,
        cat_features=[],
        verbose=False,
    )
    assert "fixed_emb" in embedding_features, f"pl.Array(Float32, 4) column not detected as embedding; text={text_features}, emb={embedding_features}"


def test_pl_list_float32_still_detected_as_embedding():
    """Backward-compat: the original List(Float32) detection still works."""
    from mlframe.training.core._misc_helpers import _auto_detect_feature_types
    from mlframe.training.configs import FeatureTypesConfig

    rows = [list(np.random.random(4).astype(np.float32)) for _ in range(3)]
    df = pl.DataFrame(
        {
            "id": pl.Series([1, 2, 3], dtype=pl.Int32),
            "float_emb": pl.Series(rows, dtype=pl.List(pl.Float32)),
        }
    )
    cfg = FeatureTypesConfig(auto_detect_feature_types=True)
    _, embedding_features, _ = _auto_detect_feature_types(
        df,
        feature_types_config=cfg,
        cat_features=[],
        verbose=False,
    )
    assert "float_emb" in embedding_features
