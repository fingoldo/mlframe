"""Regression: the f_classif constant-column gate must use CENTERED variance, not raw total_sumsq.

A large-mean, low-but-real-variance informative column has a huge uncentered ``total_sumsq``; gating
``sst <= eps * |total_sumsq| * N`` ballooned the threshold above the column's genuine centered
variance and silently dropped it (assigned the -inf constant sentinel). The fix gates against the
centered FP-cancellation floor ``eps * max(|total_sumsq|, |correction|)`` so such columns survive.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter_univariate import f_classif_chunked


def test_large_mean_low_variance_informative_column_not_dropped():
    """Large mean low variance informative column not dropped."""
    rng = np.random.default_rng(0)
    n = 2000
    y = (np.arange(n) % 2).astype(int)

    # Large mean (1e7), small but class-correlated variance -> genuinely informative for f_classif.
    # The huge uncentered total_sumsq is what the pre-fix gate scaled against, ballooning the
    # threshold above this column's real centered variance and dropping it.
    big_mean = 1.0e7
    col = big_mean + np.where(y == 1, 1.0, -1.0) + rng.normal(scale=0.05, size=n)
    # A noise reference column with no class signal.
    noise = big_mean + rng.normal(scale=0.05, size=n)

    X = np.column_stack([col, noise]).astype(np.float64)
    f = f_classif_chunked(X, y)

    assert np.isfinite(f[0]), "large-mean low-variance informative column was wrongly gated as constant (-inf)"
    assert f[0] > f[1], "informative column should outrank the noise column"


def test_truly_constant_column_still_dropped():
    """Truly constant column still dropped."""
    n = 200
    y = (np.arange(n) % 2).astype(int)
    const = np.full(n, 1.0e6, dtype=np.float64)
    signal = np.where(y == 1, 1.0, -1.0).astype(np.float64)
    X = np.column_stack([const, signal])
    f = f_classif_chunked(X, y)
    assert f[0] == -np.inf, "a literally-constant column must still receive the -inf sentinel"
