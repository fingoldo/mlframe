"""Cluster 08 regression: a failed BGMM fit falls back to bootstrap at WARNING level (audit 2026-11-20, finding 08-F4).

The fallback used to log at INFO, so a bootstrap silently stood in for a BGM sample. It is now a WARNING, and the output is still the bootstrap
rows of the right shape.
"""

import logging
from unittest.mock import patch

import numpy as np

from mlframe.feature_engineering.transformer import _bgmm_sample as bs


def test_bgm_failure_logs_warning_and_returns_bootstrap(caplog):
    """A raising BGM fit gives bootstrap rows and a WARNING (08-F4)."""
    X = np.random.default_rng(0).normal(size=(30, 2))
    with patch("sklearn.mixture.BayesianGaussianMixture.fit", side_effect=RuntimeError("boom")), caplog.at_level(logging.WARNING):
        out = bs.fit_bgmm_and_sample(X, 10, 3, 0, caller="t")
    assert out.shape == (10, 2)
    assert any(r.levelno == logging.WARNING and "falling back to bootstrap" in r.getMessage() for r in caplog.records)
