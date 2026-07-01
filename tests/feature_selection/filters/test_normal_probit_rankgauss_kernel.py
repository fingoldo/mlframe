"""iter69 regression sensors: standard-normal inverse-CDF / CDF hot paths use the
bare ``scipy.special.ndtri`` / ``ndtr`` kernels (bit-identical to ``norm.ppf`` /
``norm.cdf`` but ~2.4x faster -- no rv_continuous wrapper). Pins output identity
to the norm-based reference so a future revert to norm.ppf is caught only by speed,
never by a numeric drift, and so a kernel swap that is NOT bit-identical fails here.
"""
from __future__ import annotations

import numpy as np
import pytest

from scipy.stats import norm

from mlframe.feature_selection.filters._fastmi import _probit
from mlframe.feature_selection.filters._extra_fe_families import _rank_to_gauss
from mlframe.training.composite.transforms.unary import (
    quantile_normal_y_fit,
    quantile_normal_y_forward,
    quantile_normal_y_inverse,
)


def test_probit_bit_identical_to_norm_ppf():
    rng = np.random.default_rng(11)
    u = rng.uniform(1e-9, 1.0 - 1e-9, 20000)
    assert np.array_equal(_probit(u), norm.ppf(u))
    # +/-inf edge at exact 0 / 1 must survive the swap.
    edge = np.array([0.0, 1.0, 0.5])
    assert np.array_equal(_probit(edge), norm.ppf(edge), equal_nan=True)


def test_rank_to_gauss_bit_identical_to_norm_ppf():
    n = 20000
    ranks = np.arange(n)
    got = _rank_to_gauss(ranks, n)
    u = np.clip((ranks.astype(np.float64) + 0.5) / float(n), 1e-6, 1.0 - 1e-6)
    assert np.array_equal(got, norm.ppf(u).astype(np.float64))


@pytest.mark.parametrize("n", [2000, 20000])
def test_quantile_normal_forward_inverse_bit_identical(n):
    rng = np.random.default_rng(13)
    y = np.exp(rng.standard_normal(n))
    params = quantile_normal_y_fit(y)
    knots_y = np.asarray(params["knots_y"], dtype=np.float64)
    knots_q = np.asarray(params["knots_q"], dtype=np.float64)

    # forward == norm.ppf reference
    q = np.interp(y.astype(np.float64), knots_y, knots_q)
    eps = 1.0 / (2.0 * len(knots_q))
    q = np.clip(q, eps, 1.0 - eps)
    assert np.array_equal(quantile_normal_y_forward(y, params), norm.ppf(q))

    # inverse == norm.cdf reference
    t = rng.standard_normal(n)
    ref = np.interp(norm.cdf(t.astype(np.float64)), knots_q, knots_y)
    assert np.array_equal(quantile_normal_y_inverse(t, params), ref)
