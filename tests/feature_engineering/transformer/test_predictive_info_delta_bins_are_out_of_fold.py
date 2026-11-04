"""predictive_info_delta must not report information on a target the features say nothing about."""
import numpy as np

from mlframe.feature_engineering.transformer.predictive_info_delta import compute_predictive_info_delta_features


def test_noise_target_gives_no_information_delta():
    """Pure-noise X and Bernoulli(0.5) y: the per-bin entropies of in-sample predictions made extreme bins look pure (about 0.4 on held-out rows at n=300, d=20)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20)).astype(np.float32)
    y = (rng.random(300) > 0.5).astype(np.float32)
    Xq = rng.standard_normal((2000, 20)).astype(np.float32)
    df = compute_predictive_info_delta_features(X, y, Xq, seed=1, task="binary")
    assert float(np.abs(df["pinfo_delta"].to_numpy()).mean()) < 0.1
