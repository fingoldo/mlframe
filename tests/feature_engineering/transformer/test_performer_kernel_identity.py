"""Audit 2026-11-20 finding 09-F1: the positive random-feature map must approximate the softmax kernel the module documents.

With unscaled directions W ~ N(0, I) and phi(x) = exp(W^T x - ||x||^2 / 2), E[phi(q).phi(k)] = exp(q.k) exactly. The module
documents the scaled kernel exp(q.k / sqrt(d)), obtained by scaling the inputs by d^{-1/4}. Scaling W by 1/sqrt(d) (the old code)
gave exp(q.k / d) times a norm factor that does not cancel, so attention was biased toward train rows near the origin.
"""

import numpy as np

def _mean_kernel_estimate(q, k, d, n_features, seed):
    """Monte-Carlo estimate of phi(q).phi(k) for the unshifted positive map on the d^{-1/4}-scaled inputs.

    The production map subtracts one global shift from every log-feature; that shift only rescales the kernel, and it cancels in the
    numer/denom ratio, so the identity is checked on the raw map here.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, n_features))
    scale = d**-0.25

    def phi(x):
        """Unshifted positive random-feature map on the scaled input."""
        xs = x * scale
        return np.exp(xs @ W - 0.5 * (xs**2).sum())

    return float(phi(q) @ phi(k)) / n_features  # the Monte-Carlo mean over the random features


def test_positive_features_approximate_scaled_softmax_kernel_for_large_norm_keys():
    """A key with large norm (||k||^2 = 8 at d = 8) must NOT be down-weighted by exp(-3.5) relative to the origin."""
    d, n_features = 8, 50_000
    q = np.full(d, 0.5)
    k_far = np.full(d, 1.0)  # ||k||^2 = 8
    expected = np.exp(q @ k_far / np.sqrt(d))
    got = _mean_kernel_estimate(q, k_far, d, n_features, seed=0)
    assert abs(got / expected - 1.0) < 0.05, f"kernel estimate {got:.4f} vs exp(q.k/sqrt(d)) {expected:.4f}"


def test_kernel_estimate_is_independent_of_key_norm_offset():
    """Scaling a key away from the origin must follow exp(q.k/sqrt(d)), not an extra norm-dependent factor."""
    d, n_features = 8, 50_000
    q = np.array([0.3, -0.2, 0.1, 0.4, -0.5, 0.2, 0.0, 0.1])
    for scale_k in (0.0, 0.5, 1.0):
        k = scale_k * np.ones(d)
        expected = np.exp(q @ k / np.sqrt(d))
        got = _mean_kernel_estimate(q, k, d, n_features, seed=1)
        assert abs(got / expected - 1.0) < 0.08, f"scale={scale_k}: got {got:.4f}, expected {expected:.4f}"
