"""Rows that no fold validates (the first block of a TimeSeriesSplit) read NaN, not a real-looking 0, in the OOF transformers."""
import numpy as np
import pytest
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_engineering.transformer.variance_baseline import compute_variance_baseline_features
from mlframe.feature_engineering.transformer.target_quantile import compute_target_quantile_attention
from mlframe.feature_engineering.transformer.sign_residual_baseline import compute_sign_residual_baseline_features


@pytest.mark.parametrize("fn", [compute_variance_baseline_features, compute_target_quantile_attention, compute_sign_residual_baseline_features])
def test_unvalidated_rows_are_nan(fn):
    """TimeSeriesSplit never validates the first block, so those rows must be NaN."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = (X[:, 0] + 0.3 * rng.standard_normal(200) > 0).astype(np.float32)
    kwargs = {"task": "binary"} if fn is not compute_target_quantile_attention else {}
    df = fn(X, y, None, TimeSeriesSplit(n_splits=3), seed=1, **kwargs)
    first = df.to_numpy()[:20]
    assert np.isnan(first).all(), first[:2]
