"""per_class_spectral builds one graph per class, so it refuses a target that is not 0/1."""
import numpy as np
import pytest

from mlframe.feature_engineering.transformer.per_class_spectral import compute_per_class_spectral_attention


def test_continuous_target_raises():
    """A regression y used to be silently thresholded at 0.5."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="binary"):
        compute_per_class_spectral_attention(X, rng.random(100).astype(np.float32), X[:5], seed=1)
