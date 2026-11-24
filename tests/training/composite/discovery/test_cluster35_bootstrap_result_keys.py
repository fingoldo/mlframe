"""Audit 2026-11-20 finding 35-F45: the bootstrap posterior result has the same keys as the conjugate one.

Code that switches between the two variants read posterior_kind and friends and raised KeyError on the bootstrap result.
"""

import numpy as np

from mlframe.training.composite.discovery.bayesian import bayesian_alpha_fit, bayesian_alpha_fit_bootstrap

_NEEDED = ("posterior_kind", "degrees_of_freedom", "n_samples", "sigma2_posterior_mean")


def test_bootstrap_result_carries_the_conjugate_keys():
    """Every key the conjugate posterior returns is present on the bootstrap result."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=80)
    y = 1.5 * base + 0.2 * rng.normal(size=80)
    conj = bayesian_alpha_fit(y, base, n_samples=100, random_state=0)
    boot = bayesian_alpha_fit_bootstrap(y, base, n_bootstrap=50, random_state=0)
    assert _NEEDED
    assert all(key in boot for key in _NEEDED), [key for key in _NEEDED if key not in boot]
    assert boot["posterior_kind"] == "bootstrap"
    assert set(conj) - set(boot) == set()
