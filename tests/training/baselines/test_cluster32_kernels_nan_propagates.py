"""Cluster 32 regression: the bootstrap kernels do not compile in NaN-assuming fastmath, so a NaN input stays NaN (audit 2026-11-20, finding 32-F13)."""

import numpy as np
import pytest

from mlframe.training.baselines import _dummy_numba_kernels as k


def test_nan_fastmath_flags_exclude_nnan_and_ninf():
    """The flag set used by every kernel excludes nnan/ninf (32-F13)."""
    assert "nnan" not in k._NAN_SAFE_FASTMATH and "ninf" not in k._NAN_SAFE_FASTMATH


@pytest.mark.skipif(not k._NUMBA_AVAILABLE, reason="numba not installed")
def test_bootstrap_rmse_kernel_propagates_nan():
    """A NaN prediction makes the bootstrap samples NaN rather than a finite value (32-F13)."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, np.nan, 3.0, 4.0])
    out = k._numba_bootstrap_rmse_samples(y, p, 50, 7)
    assert np.isnan(out).any()
