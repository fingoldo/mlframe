"""variance_baseline learns sigma^2 from out-of-fold residuals, so it does not understate the noise of an unpredictable target."""
import numpy as np

from mlframe.feature_engineering.transformer.variance_baseline import compute_variance_baseline_features


def test_sigma2_on_a_noise_target_is_near_its_true_variance():
    """y ~ N(0, 1) independent of X: the mean predicted sigma^2 should be near 1; in-sample residuals of the mu model shrank it to about 0.47."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20)).astype(np.float32)
    y = rng.standard_normal(300).astype(np.float32)
    Xq = rng.standard_normal((2000, 20)).astype(np.float32)
    feats = compute_variance_baseline_features(X, y, Xq, seed=1, task="regression").to_numpy()
    assert float(feats[:, 1].mean()) > 0.8
