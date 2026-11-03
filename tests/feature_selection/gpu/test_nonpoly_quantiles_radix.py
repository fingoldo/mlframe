"""The non-polynomial cells' column quantiles through the radix select equal the column-sort formula, value for value."""

from __future__ import annotations

import math

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters.hermite_fe._nonpoly_cells_gpu import _quantile_columns


def _by_sort(M, fractions):
    """The column-sort reference: numpy's linear quantile from the sorted columns, in double."""
    n = M.shape[0]
    srt = cp.sort(M, axis=0)
    out = []
    for q in fractions:
        v = q * (n - 1)
        lo = math.floor(v)
        hi = min(lo + 1, n - 1)
        t = v - lo
        a, b = srt[lo].astype(cp.float64), srt[hi].astype(cp.float64)
        d = b - a
        out.append(b - d * (1.0 - t) if t >= 0.5 else a + d * t)
    return cp.stack(out)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fractions", [[0.25, 0.5, 0.75], [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99], [0.0, 1.0, 1 / 3]])
def test_equals_sort_formula(dtype, fractions):
    """Normal, constant, tied and heavy-tailed columns, odd row count."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3001, 9)).astype(dtype)
    x[:, 3] = 1.0
    x[:, 4] = np.round(x[:, 4])
    x[:, 5] = rng.standard_cauchy(3001).astype(dtype)
    M = cp.asarray(x)
    assert cp.array_equal(_quantile_columns(cp, M, fractions), _by_sort(M, fractions))
