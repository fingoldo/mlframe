"""Audit 2026-11-20 finding 39-F35: when min_df prunes every term, the TF-IDF encoder emits a zero-width matrix instead of crashing."""

import pandas as pd

from mlframe.training.feature_handling.handlers import TfidfParams
from mlframe.training.feature_handling.text_encoder import TextColumnEncoder


def test_min_df_pruning_every_term_gives_zero_width_not_a_crash():
    """Every word appears in one document and min_df=2 removes them all; sklearn raises 'no terms remain', which is handled."""
    df = pd.DataFrame({"t": ["alpha", "beta", "gamma"]})
    enc = TextColumnEncoder("t", TfidfParams(min_df=2, max_features=50))
    out = enc.fit_transform(df)
    assert out.shape == (3, 0)
    assert enc._empty_vocab is True
