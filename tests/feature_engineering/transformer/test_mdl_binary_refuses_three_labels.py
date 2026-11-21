"""The binary MDL binning refuses a target with more than two labels instead of mis-counting (audit 2026-11-20, finding 09-F11)."""
import numpy as np
import pytest

from mlframe.feature_engineering.transformer.mdl_binning_pairwise import compute_mdl_binning_pairwise_features


def test_binary_task_with_three_labels_raises():
    """Three classes under task='binary' used to index past the two-class count arrays."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.standard_normal((n, 3))
    y = rng.integers(0, 3, n)
    with pytest.raises(ValueError, match="at most two distinct labels"):
        compute_mdl_binning_pairwise_features(X, y, rng.standard_normal((5, 3)), splitter=None, seed=0, task="binary")
