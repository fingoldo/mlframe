"""The KS and Wasserstein-1 distances of ks_shift equal scipy's exact two-sample values."""

from __future__ import annotations

import numpy as np
from scipy import stats

from mlframe.feature_engineering.transformer.ks_shift import _ks_and_wasserstein


def _cases():
    """Local samples around, far from and constant against a global sample, plus a gap case."""
    rng = np.random.default_rng(0)
    global_sorted = np.sort(rng.normal(size=500) * 3 + 50)
    rows = [rng.normal(size=12) * 2 + 51 for _ in range(20)]
    rows += [np.full(12, 100.0), np.full(12, -10.0), np.array([0.0, 1.0] * 6), rng.uniform(0, 200, 12)]
    return np.vstack(rows), global_sorted


def test_ks_and_w1_equal_scipy():
    """Both distances agree with scipy to float32 rounding; the old kernel evaluated the local CDF at the wrong side and ignored mass outside the neighbours' range."""
    local, g = _cases()
    ks, w1 = _ks_and_wasserstein(local, g)
    ks_ref = np.array([stats.ks_2samp(r, g).statistic for r in local])
    w1_ref = np.array([stats.wasserstein_distance(r, g) for r in local])
    np.testing.assert_allclose(ks, ks_ref, atol=1e-6)
    np.testing.assert_allclose(w1, w1_ref, rtol=1e-4, atol=1e-5)


def test_the_documented_failing_examples():
    """local = [0, 1] against global = [0.5] has W1 = 0.5 (it was 0); a constant local sample above the global range has KS = 1 and W1 = the mean gap."""
    _ks, w1 = _ks_and_wasserstein(np.array([[0.0, 1.0]]), np.array([0.5]))
    assert abs(float(w1[0]) - 0.5) < 1e-6
    g = np.linspace(0.0, 1.0, 100)
    ks2, w12 = _ks_and_wasserstein(np.full((1, 4), 100.0), g)
    assert abs(float(ks2[0]) - 1.0) < 1e-6 and abs(float(w12[0]) - 99.5) < 1e-3
