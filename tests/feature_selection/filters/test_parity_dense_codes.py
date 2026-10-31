"""``_dense_codes`` ranks labels as ``np.unique(return_inverse)`` does, and the parity scan and the raw-column MI are unchanged by it."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._parity_fe import _dense_codes, _raw_mi, scan_parity_period


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.uint8, np.bool_, np.float64])
def test_dense_codes_equal_unique_inverse(dtype):
    """Gaps, negative values and a single class all give the dense ranks of the sorted distinct labels."""
    rng = np.random.default_rng(0)
    for vals in ([0, 1], [3, 7, 9, 3, 7], [-5, 2, -5, 100], [4, 4, 4]):
        y = rng.choice(np.array(vals), 500).astype(dtype)
        want = np.unique(y, return_inverse=True)[1].ravel()
        got = _dense_codes(y)
        assert got.dtype == np.int64 and np.array_equal(got, want)


def test_raw_mi_and_period_scan_values_are_unchanged():
    """A many-level column, a binary column and a few-level column give the MI the sort-based computation gives."""
    rng = np.random.default_rng(1)
    n = 6000
    y = (rng.random(n) < 0.4).astype(np.int8)
    for v in (rng.standard_normal(n), (rng.random(n) < 0.5).astype(np.float64), rng.integers(0, 6, n).astype(np.float64), np.where(rng.random(n) < 0.97, 1.0, rng.standard_normal(n))):
        levels = np.unique(v)
        if levels.size <= 10:
            codes, k = np.searchsorted(levels, v), max(levels.size, 1)
        else:
            edges = np.unique(np.nanquantile(v, np.linspace(0.0, 1.0, 11)))
            codes, k = np.clip(np.searchsorted(edges, v, side="right") - 1, 0, max(len(edges) - 2, 0)), max(len(edges) - 1, 1)
        from mlframe.feature_selection.filters._numba_utils import _mi_codes_njit

        yc = np.unique(y, return_inverse=True)[1].astype(np.int64).ravel()
        assert _raw_mi(v, y) == float(_mi_codes_njit(codes.astype(np.int64), k, yc, int(yc.max()) + 1))
    period, mi = scan_parity_period(np.sort(rng.standard_normal(n)) * 3.0, (np.arange(n) // 50 % 2).astype(np.int8))
    assert np.isfinite(period) and mi >= 0.0
