"""Empty residual bands (tied residuals) must not receive softmax weight."""
import numpy as np

import mlframe.feature_engineering.transformer.multi_temp_residual_band as mod


def test_constant_baseline_on_binary_target_gives_empty_bands_zero_weight(monkeypatch):
    """A constant 0.5 baseline makes every |residual| equal, so only the first band holds rows; the others used to share the weight through their placeholder centroid."""
    monkeypatch.setattr(mod, "_fit_baseline_predict", lambda Xt, y, **kw: np.full(len(y), 0.5, dtype=np.float32))
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3)).astype(np.float32)
    y = (rng.random(200) > 0.5).astype(np.float32)
    df = mod.compute_multi_temp_residual_band_features(X, y, rng.standard_normal((10, 3)).astype(np.float32), seed=1, task="binary", n_bands=4, temps=(1.0,))
    w = df.select([c for c in df.columns if "_w_R" in c]).to_numpy()
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert (w[:, 1:] == 0.0).all() and np.allclose(w[:, 0], 1.0)
