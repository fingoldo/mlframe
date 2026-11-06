"""The pure-Python frac-diff prefix inverse matches the njit kernel bit for bit.

The njit kernel scales by ``inv_w0 = 1.0 / weights[0]``. The Python fallback divided by ``weights[0]`` directly. For a non-exact
reciprocal (weights[0] = 0.4) the two round differently at every step, and the recurrence carries that difference forward, so the
fallback drifted from the njit result on machines without numba.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.training.composite.transforms._nonlinear_ewma_fracdiff as fd


@pytest.mark.skipif(getattr(fd, "_frac_diff_inverse_kernel_prefix", None) is None, reason="needs the numba kernel to compare against")
def test_python_fallback_prefix_inverse_is_bit_identical_to_njit(monkeypatch):
    """The numba-less fallback returns the same array as the njit kernel for a non-exact reciprocal weight."""
    rng = np.random.default_rng(3)
    lags = 3
    weights = np.array([0.4, -0.2, 0.1, -0.05])
    t_f = rng.normal(size=400)
    prefix = rng.normal(size=lags)

    njit_out = fd._frac_diff_inverse_prefix(t_f, lags, weights, prefix)
    monkeypatch.setattr(fd, "_HAS_NUMBA", False)
    fallback_out = fd._frac_diff_inverse_prefix(t_f, lags, weights, prefix)

    np.testing.assert_array_equal(fallback_out, njit_out)
