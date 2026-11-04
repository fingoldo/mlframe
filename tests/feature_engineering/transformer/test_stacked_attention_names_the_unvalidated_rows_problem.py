"""stacked_attention explains why two layers cannot be stacked over a splitter that leaves rows unvalidated."""
import numpy as np
import pytest
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_engineering.transformer.stacked_attention import compute_stacked_row_attention


def test_time_series_split_with_two_layers_raises_a_message_about_unvalidated_rows():
    """The first TimeSeriesSplit block has no out-of-fold value, so layer 1 output has NaN rows; the error must say so instead of a generic input-validation failure."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    with pytest.raises(ValueError, match="without an out-of-fold value"):
        compute_stacked_row_attention(X, y, None, TimeSeriesSplit(n_splits=3), seed=1, n_layers=2, n_heads=2, head_dim=4, k=5)
