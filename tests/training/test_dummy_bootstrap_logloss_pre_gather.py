"""Regression: ``_vectorized_bootstrap_logloss_samples`` pre-gather refactor.

iter118 moved the per-element log-loss computation (np.clip + 2x np.log +
np.where) BEFORE the (n_resamples, n) bootstrap-index gather. The bootstrap
itself becomes a single gather of the pre-computed per-element loss vector,
collapsing the 8x duplicated numpy work on the (n_resamples, n) tensor.

The optimisation is bit-equivalent (np.log / np.clip / np.where are
elementwise; gathering after they run produces the same numbers as gathering
before and running them on the gathered tensor). This test pins:
  (1) bit-equivalence on the 1-D binary path
  (2) bit-equivalence on the 2-D multilabel-macro path
  (3) shape-rejection short-circuits stay correct

so a future refactor that subtly changes ordering / type-promotion (e.g.
casts y to bool before pre-gather) fails fast at unit-test time rather than
silently shifting bootstrap-CI bounds in fuzz runs.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.dummy_baselines import _vectorized_bootstrap_logloss_samples


def _legacy_post_gather(y, p, n_resamples, seed, eps=1e-15):
    """Reference oracle: the pre-iter118 ``post-gather`` implementation."""
    if n_resamples <= 0 or len(y) < 10 or y.shape != p.shape:
        return None
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(n_resamples, n))
    y_r = y[idx]
    p_r = p[idx]
    p_clip = np.clip(p_r, eps, 1.0 - eps)
    is_pos = y_r > 0.5
    elem = -np.where(is_pos, np.log(p_clip), np.log(1.0 - p_clip))
    if y.ndim == 1:
        return elem.mean(axis=1)
    if y.ndim == 2:
        return elem.mean(axis=(1, 2))
    return None


def test_pre_gather_1d_matches_post_gather_oracle():
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, size=n).astype(np.float64)
    p = np.clip(rng.random(n), 1e-3, 1 - 1e-3)
    out_new = _vectorized_bootstrap_logloss_samples(y, p, 300, seed=42)
    out_old = _legacy_post_gather(y, p, 300, seed=42)
    assert out_new is not None and out_old is not None
    assert np.array_equal(out_new, out_old), (
        "pre-gather output must be bit-identical to the post-gather oracle"
    )


def test_pre_gather_2d_multilabel_matches_post_gather_oracle():
    rng = np.random.default_rng(0)
    n, K = 500, 4
    y = rng.integers(0, 2, size=(n, K)).astype(np.float64)
    p = np.clip(rng.random((n, K)), 1e-3, 1 - 1e-3)
    out_new = _vectorized_bootstrap_logloss_samples(y, p, 300, seed=42)
    out_old = _legacy_post_gather(y, p, 300, seed=42)
    assert out_new is not None and out_old is not None
    assert np.array_equal(out_new, out_old)


def test_pre_gather_returns_none_on_shape_mismatch():
    """y.shape != p.shape short-circuits to None for the sklearn fallback."""
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
    p = np.array([[0.1, 0.9]] * 12, dtype=np.float64)  # (n, 2) vs y (n,)
    assert _vectorized_bootstrap_logloss_samples(y, p, 100, seed=0) is None


def test_pre_gather_returns_none_for_tiny_n():
    """n<10 short-circuits to None (caller falls back to the sklearn loop)."""
    y = np.array([0, 1, 0, 1, 0], dtype=np.float64)
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.3], dtype=np.float64)
    assert _vectorized_bootstrap_logloss_samples(y, p, 100, seed=0) is None


def test_multiclass_path_matches_sklearn_log_loss():
    """iter120 extension: multiclass y (n,) int + p (n, K) softmax. Must
    match the sklearn-per-resample loop at fp64 epsilon and run ~120x
    faster (1500 ms -> 12 ms at n=1500 / K=4 / R=1000)."""
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(0)
    n, K = 600, 4
    y = rng.integers(0, K, size=n).astype(np.int64)
    logits = rng.standard_normal((n, K))
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    samples_new = _vectorized_bootstrap_logloss_samples(y, p, 200, seed=42)
    assert samples_new is not None and samples_new.shape == (200,)

    # sklearn reference -- replay the per-resample log_loss with the same seed.
    ref = np.empty(200, dtype=np.float64)
    rng_ref = np.random.default_rng(42)
    for r in range(200):
        idx = rng_ref.integers(0, n, size=n)
        ref[r] = log_loss(y[idx], p[idx], labels=np.arange(K))
    assert np.abs(samples_new - ref).max() < 1e-12, (
        f"multiclass bootstrap log-loss must match sklearn at fp64 epsilon "
        f"(max abs diff={np.abs(samples_new - ref).max():.2e})"
    )


def test_multiclass_path_rejects_out_of_range_labels():
    """y label outside [0, K-1] -> None (caller falls back). Prevents a silent
    fancy-index out-of-range that would read garbage memory."""
    n, K = 20, 3
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                  1, 2, 0, 1, 2, 0, 1, 2, 0, K + 1], dtype=np.int64)  # last is invalid
    p = np.full((n, K), 1.0 / K)
    assert _vectorized_bootstrap_logloss_samples(y, p, 50, seed=0) is None


def test_multiclass_path_rejects_float_labels():
    """Float-y vs (n, K) -p shape mismatch should still fail-closed: float
    labels aren't valid class indices, so the multiclass branch's
    ``dtype.kind in ('i', 'u')`` guard short-circuits to None."""
    n, K = 20, 3
    y = np.full(n, 0.5)  # float, not integer
    p = np.full((n, K), 1.0 / K)
    assert _vectorized_bootstrap_logloss_samples(y, p, 50, seed=0) is None
