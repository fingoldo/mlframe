"""pairwise_kl_divergence takes each baseline's sigma from out-of-fold residuals."""
import numpy as np

from mlframe.feature_engineering.transformer.pairwise_kl_divergence import _fit_3baselines_with_sigma


def test_sigmas_of_an_unpredictable_target_are_near_its_std():
    """y ~ N(0, 1) independent of X: every baseline's honest error std is about 1; the deeper booster's in-sample residual std was about half of that."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20)).astype(np.float32)
    y = rng.standard_normal(300).astype(np.float32)
    _preds, sigmas = _fit_3baselines_with_sigma(X, y, X[:10], "regression", 1)
    assert min(sigmas) > 0.85, sigmas
