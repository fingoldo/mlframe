"""Audit 2026-11-20 finding 35-F41: bayesian_alpha_fit drops non-finite pairs, counts them, and logs it.

A single NaN used to reach the regression and make the whole posterior NaN with no warning.
"""

import logging

import numpy as np

from mlframe.training.composite.discovery.bayesian import bayesian_alpha_fit


def test_nonfinite_pair_is_dropped_counted_and_logged(caplog):
    """One NaN in y gives a finite posterior, n_dropped_nonfinite == 1, and a WARNING."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=60)
    y = 2.0 * base + 0.1 * rng.normal(size=60)
    y[7] = np.nan
    with caplog.at_level(logging.WARNING):
        res = bayesian_alpha_fit(y, base, n_samples=200, random_state=0)
    assert res["n_dropped_nonfinite"] == 1
    assert np.isfinite(res["alpha_mean"])
    assert any("dropped 1 non-finite" in r.getMessage() for r in caplog.records)
