"""Cluster 22 regression: robust location estimators refuse NaN/inf instead of returning a biased or NaN value (audit 2026-11-20, finding 22-F44)."""

import numpy as np
import pytest

from mlframe.core.robust_location import robust_mean_mestimator, trimmed_mean


def test_trimmed_mean_refuses_nan():
    """A NaN in the sample raises, instead of sorting to the end and biasing the trim (22-F44)."""
    with pytest.raises(ValueError, match="non-finite"):
        trimmed_mean(np.array([1.0, 2.0, np.nan, 4.0]), proportion=0.1)


def test_mestimator_refuses_inf():
    """An infinity raises, instead of propagating through the IRLS iterations (22-F44)."""
    with pytest.raises(ValueError, match="non-finite"):
        robust_mean_mestimator(np.array([1.0, np.inf, 3.0]))


def test_clean_sample_still_estimates():
    """Control: a finite sample gives a finite location (22-F44)."""
    assert np.isfinite(trimmed_mean(np.array([1.0, 2.0, 3.0, 4.0]), proportion=0.25))
    assert np.isfinite(robust_mean_mestimator(np.array([1.0, 2.0, 3.0, 4.0])))
