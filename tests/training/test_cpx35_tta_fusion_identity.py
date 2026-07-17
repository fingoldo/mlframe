"""CPX35: the fused tta_point_mean_spread is FP-identical to the legacy three-sweep (point/mean/spread) path and uses n model calls, not 2n+1.

The fused helper reuses one clean pass as both the point estimate and the first augmentation member and accumulates the mean/spread of the
n-1 jittered passes via Welford, so it calls predict_fn exactly n times where the legacy point + tta_predict + tta_predict_spread trio called
it 2n+1 times. mean/spread match the two-pass np.mean/np.std references to FP reduction-order tolerance (~1e-9); point is bit-identical.
"""

from __future__ import annotations

import numpy as np

from mlframe.training._tta import tta_predict, tta_predict_spread, tta_point_mean_spread


def _model():
    w = np.array([1.3, -0.7, 0.5, 2.0, -1.1])
    calls = {"n": 0}

    def predict(Z):
        calls["n"] += 1
        return Z[:, :5] @ w + 0.4 * np.sin(7.0 * Z[:, 6])

    return predict, calls


def test_fused_matches_legacy_and_uses_n_calls():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4000, 8))
    n, sigma = 16, 0.03

    p_legacy, c_legacy = _model()
    point_ref = np.asarray(p_legacy(X), dtype=np.float64)
    mean_ref = np.asarray(tta_predict(p_legacy, X, n=n, sigma_scale=sigma, seed=0), dtype=np.float64)
    spread_ref = np.asarray(tta_predict_spread(p_legacy, X, n=n, sigma_scale=sigma, seed=0), dtype=np.float64)
    assert c_legacy["n"] == 2 * n + 1  # 3 clean + 2*(n-1) jittered

    p_fused, c_fused = _model()
    point, mean, spread = tta_point_mean_spread(p_fused, X, n=n, sigma_scale=sigma, seed=0)
    assert c_fused["n"] == n  # 1 clean + (n-1) jittered

    assert np.array_equal(point, point_ref)  # clean pass bit-identical
    assert np.max(np.abs(mean - mean_ref)) < 1e-9
    assert np.max(np.abs(spread - spread_ref)) < 1e-9


def test_fused_noop_when_sigma_zero_or_single_sample():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 4))

    def f(Z):
        return Z[:, 0] * 2.0

    for kwargs in ({"n": 16, "sigma_scale": 0.0}, {"n": 1, "sigma_scale": 0.1}):
        point, mean, spread = tta_point_mean_spread(f, X, **kwargs)
        assert np.array_equal(point, np.asarray(f(X), dtype=np.float64))
        assert np.array_equal(mean, point)
        assert np.allclose(spread, 0.0)


def test_fused_spread_zero_for_constant_model():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 3))

    def const(Z):
        return np.ones(Z.shape[0])

    _, _, spread = tta_point_mean_spread(const, X, n=16, sigma_scale=0.1, seed=1)
    assert np.allclose(spread, 0.0)
