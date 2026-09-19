"""Discrete bases keep one bin per value in the quantile-binned residual transforms.

``np.unique`` of the quantile edges merges the values of a binary base into a single bin, so the per-bin median residual
degenerated to the global median. A production run hit this on the binary ``has_explicit_budget`` base.
"""

from __future__ import annotations

import warnings

import numpy as np

from mlframe.training.composite._quantile_edges import quantile_bin_edges
from mlframe.training.composite.transforms.simple import _median_residual_fit, _median_residual_g


def _binary_fixture(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    base = (rng.random(n) < 0.3).astype(np.float64)
    y = np.where(base == 1.0, 10.0, 1.0) + rng.normal(scale=0.1, size=n)
    return y, base


def test_binary_base_median_residual_uses_one_bin_per_value():
    y, base = _binary_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        params = _median_residual_fit(y, base)
    g = _median_residual_g(np.array([0.0, 1.0]), params)
    np.testing.assert_allclose(g, [1.0, 10.0], atol=0.05)


def test_quantile_residual_binary_base_has_two_bins():
    from mlframe.training.composite.transforms.nonlinear import _quantile_residual_fit

    y, base = _binary_fixture()
    params = _quantile_residual_fit(y, base)
    assert int(params["n_bins"]) == 2
    np.testing.assert_allclose(sorted(params["bin_medians"]), [1.0, 10.0], atol=0.05)


def test_edges_helper():
    np.testing.assert_array_equal(quantile_bin_edges(np.array([0.0, 0.0, 1.0, 1.0]), 20), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(quantile_bin_edges(np.array([3.0, 3.0]), 20), [3.0])
    continuous = np.linspace(0.0, 1.0, 1001)
    assert quantile_bin_edges(continuous, 10).size == 11
