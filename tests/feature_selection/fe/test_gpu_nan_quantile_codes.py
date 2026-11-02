"""The NaN-aware device binner returns the codes of the host binner (``discretize_2d_quantile_batch``) for columns holding NaN."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._gpu_nan_quantile_codes import nan_aware_quantile_codes_gpu
from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch


def _matrix(n=3000, seed=2):
    """Columns with scattered NaN, many NaN, ties, a constant, integers, heavy tails, and fully finite ones."""
    rng = np.random.default_rng(seed)
    cols = []
    for j in range(14):
        c = rng.standard_normal(n)
        if j % 3 == 0:
            c[rng.random(n) < 0.1] = np.nan
        if j == 4:
            c[rng.random(n) < 0.7] = np.nan
        if j == 5:
            c = np.round(c, 1)
        if j == 6:
            c = np.full(n, 2.5)
            c[::9] = np.nan
        if j == 7:
            c = rng.integers(0, 4, n).astype(float)
        if j == 8:
            c = np.exp(3 * rng.standard_normal(n))
            c[::17] = np.nan
        cols.append(c)
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("nbins", [3, 10, 16])
def test_codes_equal_the_host_binner(nbins):
    """Bit-for-bit equal codes for every column, finite and NaN-holding alike."""
    x = _matrix()
    got = cp.asnumpy(nan_aware_quantile_codes_gpu(cp.asarray(x), nbins))
    want = discretize_2d_quantile_batch(x, nbins, np.int32)
    np.testing.assert_array_equal(got, want)


def test_infinite_values_and_too_few_values_are_declined():
    """An infinite entry or a column with fewer than two non-NaN values returns ``None`` so the host keeps it."""
    x = _matrix()
    y = x.copy()
    y[5, 2] = np.inf
    assert nan_aware_quantile_codes_gpu(cp.asarray(y), 10) is None
    z = x.copy()
    z[:, 1] = np.nan
    z[0, 1] = 1.0
    assert nan_aware_quantile_codes_gpu(cp.asarray(z), 10) is None
