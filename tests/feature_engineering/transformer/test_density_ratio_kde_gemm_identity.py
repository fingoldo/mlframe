"""Regression: density_ratio _gaussian_kde_log gemm-expansion stays numerically equivalent to the naive broadcast form.

Pins that the squared-distance-via-sgemm rewrite (|x-y|^2 = |x|^2 + |y|^2 - 2 x.y^T) does not move the log-KDE output
beyond fp32 reduction-order noise (~1e-5 absolute on a ~O(1..100) log-density), well below any selection-altering delta.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.density_ratio import _gaussian_kde_log


def _kde_broadcast_reference(X_query, X_train_subset, h, chunk=1000):
    n_q = X_query.shape[0]
    n_t = X_train_subset.shape[0]
    if n_t < 1:
        return np.full(n_q, -30.0, dtype=np.float32)
    out = np.zeros(n_q, dtype=np.float32)
    h_sq = max(h * h, 1e-9)
    log_N = np.log(n_t)
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        Xq = X_query[start:end]
        d2 = ((Xq[:, None, :] - X_train_subset[None, :, :]) ** 2).sum(axis=2)
        logits = -d2 / (2.0 * h_sq)
        m = logits.max(axis=1, keepdims=True)
        lse = m.ravel() + np.log(np.exp(logits - m).sum(axis=1) + 1e-30)
        out[start:end] = (lse - log_N).astype(np.float32)
    return out


@pytest.mark.parametrize("n_q,n_t,d,h", [(500, 300, 20, 1.5), (1500, 800, 40, 0.7), (200, 100, 80, 3.0)])
def test_kde_gemm_matches_broadcast(n_q, n_t, d, h):
    rng = np.random.default_rng(123)
    Xq = rng.standard_normal((n_q, d)).astype(np.float32)
    Xt = rng.standard_normal((n_t, d)).astype(np.float32)
    new = _gaussian_kde_log(Xq, Xt, h=h)
    ref = _kde_broadcast_reference(Xq, Xt, h)
    assert new.dtype == np.float32
    assert np.max(np.abs(new - ref)) < 1e-3, "gemm KDE diverged from broadcast beyond fp32 noise"
    # relative on the meaningful-magnitude entries
    rel = np.abs(new - ref) / (np.abs(ref) + 1e-6)
    assert np.max(rel) < 1e-4


def test_kde_empty_train_subset_sentinel():
    Xq = np.zeros((5, 3), dtype=np.float32)
    Xt = np.zeros((0, 3), dtype=np.float32)
    out = _gaussian_kde_log(Xq, Xt, h=1.0)
    assert out.shape == (5,)
    assert np.all(out == -30.0)
