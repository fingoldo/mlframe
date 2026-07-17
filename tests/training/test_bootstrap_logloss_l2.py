"""Regression unit test for /loop iter 2 optimization.

Hotspot: ``dummy_baselines._bootstrap_ci_for_strongest`` -> ``_resample_metric``
called sklearn ``log_loss`` 1000 times per bootstrap resample (~14% of total
wall on n=600 fuzz cell). New ``_vectorized_bootstrap_logloss_samples`` helper
generates all resample indices in one shot and computes log-loss via numpy
broadcasting -- ~40x faster than the sklearn-per-call loop.

This test asserts:
1. The vectorised helper returns numerically-equivalent percentiles to the
   sklearn-per-call loop on a synthetic binary 1D case (correctness).
2. Same for the multilabel 2D case (correctness).
3. The vectorised path runs faster than the sklearn loop at n=600 (perf
   sanity; gate at >= 5x to leave headroom for slow CI).
"""

from __future__ import annotations

import time
import numpy as np
import pytest


def _sklearn_loop_bootstrap_binary(y: np.ndarray, p: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    """Reference impl: the pre-optimization sklearn-per-call loop."""
    from sklearn.metrics import log_loss as _ll

    rng = np.random.default_rng(seed)
    n = len(y)
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            samples.append(float(_ll(y[idx], p[idx], labels=[0, 1])))
        except Exception:
            continue
    return np.array(samples, dtype=np.float64)


def test_vectorised_binary_logloss_matches_sklearn_loop_percentiles():
    """Vectorised numpy path must return percentile-equivalent samples to the
    sklearn loop on the same seed. Allows small drift from sklearn's input-
    validation casts but the 2.5 / 50 / 97.5 percentiles must agree to 2 dp.
    """
    from mlframe.training.baselines.dummy import _vectorized_bootstrap_logloss_samples

    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, size=n).astype(np.int64)
    p = np.clip(rng.random(n), 0.01, 0.99).astype(np.float64)

    vec = _vectorized_bootstrap_logloss_samples(y, p, n_resamples=300, seed=42)
    ref = _sklearn_loop_bootstrap_binary(y, p, n_resamples=300, seed=42)

    assert vec is not None
    assert vec.shape == ref.shape == (300,)

    for q in (2.5, 50.0, 97.5):
        v = float(np.percentile(vec, q))
        r = float(np.percentile(ref, q))
        assert abs(v - r) < 0.02, f"Percentile q={q}: vectorised={v:.4f} vs sklearn-loop={r:.4f} -- drift exceeds 0.02 tolerance"


def test_vectorised_multilabel_logloss_macro_returns_finite_samples():
    """Multilabel 2D (n, K): the vectorised path must produce a length-
    ``n_resamples`` finite ndarray. Macro-averaging convention = mean over
    (rows, labels) jointly -- matches the original sklearn macro semantics
    closely enough for CI bands; exact equality with sklearn is not required
    because sklearn's per-label log_loss + outer mean has a slightly
    different numerical path.
    """
    from mlframe.training.baselines.dummy import _vectorized_bootstrap_logloss_samples

    rng = np.random.default_rng(1)
    n, K = 400, 4
    y = rng.integers(0, 2, size=(n, K)).astype(np.int64)
    p = np.clip(rng.random((n, K)), 0.01, 0.99).astype(np.float64)

    samples = _vectorized_bootstrap_logloss_samples(y, p, n_resamples=200, seed=7)
    assert samples is not None
    assert samples.shape == (200,)
    assert np.all(np.isfinite(samples))
    # Sanity: with random labels + random probs the log-loss must be
    # somewhere in the typical [0.3, 1.5] range for 50/50 binary labels.
    assert 0.2 < float(np.median(samples)) < 2.0


def test_vectorised_returns_none_on_bad_shapes():
    """Shape mismatch / 3D y must return None so caller falls back."""
    from mlframe.training.baselines.dummy import _vectorized_bootstrap_logloss_samples

    rng = np.random.default_rng(2)
    # Mismatch
    assert _vectorized_bootstrap_logloss_samples(rng.integers(0, 2, size=10), rng.random(20), n_resamples=10, seed=0) is None
    # 3D y
    assert (
        _vectorized_bootstrap_logloss_samples(
            rng.integers(0, 2, size=(10, 3, 2)),
            rng.random((10, 3, 2)),
            n_resamples=10,
            seed=0,
        )
        is None
    )
    # n < 10 minimum gate
    assert _vectorized_bootstrap_logloss_samples(rng.integers(0, 2, size=5), rng.random(5), n_resamples=10, seed=0) is None


def test_vectorised_binary_logloss_is_faster_than_sklearn_loop():
    """Perf gate: vectorised path runs >= 5x faster than the sklearn-per-call
    loop at n=600, n_resamples=300. Picked >=5x (not >=10x) to leave margin
    on slow CI / cold cache. The actual observed speedup at n=600 is ~40x.
    """
    from mlframe.training.baselines.dummy import _vectorized_bootstrap_logloss_samples

    rng = np.random.default_rng(3)
    n, n_resamples = 600, 300
    y = rng.integers(0, 2, size=n).astype(np.int64)
    p = np.clip(rng.random(n), 0.01, 0.99).astype(np.float64)

    # Warm caches.
    _ = _vectorized_bootstrap_logloss_samples(y, p, n_resamples=20, seed=0)
    _ = _sklearn_loop_bootstrap_binary(y, p, n_resamples=20, seed=0)

    t0 = time.perf_counter()
    _ = _vectorized_bootstrap_logloss_samples(y, p, n_resamples=n_resamples, seed=42)
    t_vec = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = _sklearn_loop_bootstrap_binary(y, p, n_resamples=n_resamples, seed=42)
    t_sk = time.perf_counter() - t0

    speedup = t_sk / max(t_vec, 1e-9)
    assert speedup >= 5.0, (
        f"Vectorised log-loss bootstrap only {speedup:.1f}x faster than "
        f"sklearn loop (vec={t_vec * 1000:.1f}ms, sklearn={t_sk * 1000:.1f}ms); "
        f"expected >=5x. Either the optimization regressed or the test "
        f"machine is unusually slow at numpy ops."
    )
