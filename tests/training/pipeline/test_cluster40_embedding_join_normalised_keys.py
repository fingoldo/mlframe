"""Cluster 40 regression: the embedding join matches entity ids across int/float dtype drift and zero-fills unknown ids (audit 2026-11-20, finding 40-F23)."""

import numpy as np
import pandas as pd

from mlframe.training.pipeline._latent_interaction_svd_composite_fe import _join_embeddings


def test_int_and_float_ids_match_and_unknown_ids_get_zeros():
    """A float id 3.0 finds the entity stored as int 3; an unseen id and NaN get the zero vector (40-F23)."""
    row_emb = pd.DataFrame({"e0": [1.0, 3.0], "e1": [10.0, 30.0]}, index=[1, 3])
    df = pd.DataFrame({"x": [0, 1, 2, 3]})
    group_ids = np.array([3.0, 1.0, 99.0, np.nan], dtype=object)
    out = _join_embeddings(df, group_ids, row_emb, "emb")
    assert list(out["emb_e0"]) == [3.0, 1.0, 0.0, 0.0]
    assert list(out["emb_e1"]) == [30.0, 10.0, 0.0, 0.0]
