"""Cluster 08 regression: the BGM sampler reports whether its rows came from the mixture or from a bootstrap (audit 2026-11-20, finding 08-F4).

With return_source=True the caller gets (samples, source). A bootstrap fallback, including a fit that is too small for the requested
components, is reported as "bootstrap" so it cannot pass for a mixture sample. The default return is unchanged.
"""

import numpy as np

from mlframe.feature_engineering.transformer._bgmm_sample import fit_bgmm_and_sample


def test_too_few_rows_reports_bootstrap():
    """A minority set too small for the mixture is a bootstrap, and the source says so (08-F4)."""
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    samples, source = fit_bgmm_and_sample(X, 5, 4, 0, caller="t", return_source=True)
    assert source == "bootstrap"
    assert samples.shape == (5, 2)


def test_mixture_sample_reports_bgmm():
    """A well-sized minority set is sampled from the mixture (08-F4)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    samples, source = fit_bgmm_and_sample(X, 10, 2, 0, caller="t", return_source=True)
    assert source == "bgmm"
    assert samples.shape == (10, 2)


def test_default_return_is_the_array_only():
    """Control: without the flag the function still returns just the array (08-F4)."""
    X = np.arange(6, dtype=np.float64).reshape(3, 2)
    out = fit_bgmm_and_sample(X, 4, 4, 0, caller="t")
    assert isinstance(out, np.ndarray) and out.shape == (4, 2)
