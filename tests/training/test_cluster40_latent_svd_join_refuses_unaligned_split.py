"""Audit 2026-11-20 finding 40-F15: the latent-SVD join refuses a split that has rows but no aligned group ids."""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline._latent_interaction_svd_composite_fe import _join_embeddings


def test_split_with_rows_and_no_group_ids_is_refused():
    """A val split with rows but no index alignment raises, naming the split, instead of passing through without embedding columns."""
    df = pd.DataFrame({"x": np.arange(4.0)})
    row_emb = pd.DataFrame({"e0": [1.0, 2.0]}, index=[0, 1])
    with pytest.raises(ValueError, match="val split has no row index aligned"):
        _join_embeddings(df, None, row_emb, "lat", "val")


def test_missing_frame_is_still_passed_through():
    """A split that is absent (None frame) is not an error."""
    row_emb = pd.DataFrame({"e0": [1.0]}, index=[0])
    assert _join_embeddings(None, None, row_emb, "lat", "val") is None
