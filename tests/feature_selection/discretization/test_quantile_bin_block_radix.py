"""The radix quantile block binner gives the codes of the quickselect kernel, and the dispatching wrapper takes it only for long float32 blocks."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils import quantile_bin_block
from mlframe.feature_selection.filters.discretization import _quantile_radix as qr


def _block(n=6000, seed=0):
    """Columns of every kind the binner meets: continuous, ties, a constant, two values, a skewed one with both zeros, a heavy tail, a bimodal one and a few levels."""
    rng = np.random.default_rng(seed)
    rows = [
        rng.standard_normal(n),
        np.round(rng.standard_normal(n), 1),
        np.zeros(n),
        (rng.random(n) < 0.3).astype(float),
        np.where(rng.random(n) < 0.9, 0.0, rng.standard_normal(n)),
        rng.standard_cauchy(n),
        np.where(rng.random(n) < 0.5, rng.normal(-5, 0.2, n), rng.normal(5, 0.2, n)),
        rng.integers(0, 4, n).astype(float),
        np.arange(n, dtype=float),
        np.where(np.arange(n) % 2 == 0, -0.0, 0.0),
    ]
    return np.ascontiguousarray(np.vstack(rows).astype(np.float32))


@pytest.mark.parametrize("two_class", [True, False])
@pytest.mark.parametrize("n_bins", [2, 5, 10, 48])
def test_radix_codes_equal_the_quickselect_codes(two_class, n_bins):
    """Every column kind, both collapse rules, several bin counts: the same int16 codes."""
    v = _block()
    old = np.empty(v.shape, dtype=np.int16)
    new = np.empty(v.shape, dtype=np.int16)
    quantile_bin_block(v, n_bins, old, two_class)
    qr.quantile_bin_block_radix(v, n_bins, new, two_class)
    assert np.array_equal(old, new)


def test_the_wrapper_takes_the_radix_route_for_long_float32_blocks_only(monkeypatch):
    """float32 with at least the radix size a column: radix; short, float64 or switched off: the quickselect kernel."""
    calls = []
    real = qr.quantile_bin_block_radix
    monkeypatch.setattr(qr, "quantile_bin_block_radix", lambda *a: (calls.append(1), real(*a))[1])
    v = _block()
    codes = np.empty(v.shape, dtype=np.int16)
    qr.quantile_bin_block_fast(v, 10, codes, False)
    assert calls == [1]
    qr.quantile_bin_block_fast(v.astype(np.float64), 10, codes, False)
    qr.quantile_bin_block_fast(np.ascontiguousarray(v[:, :1000]), 10, codes[:, :1000], False)
    monkeypatch.setenv("MLFRAME_QBIN_RADIX", "0")
    qr.quantile_bin_block_fast(v, 10, codes, False)
    assert calls == [1]
    reference = np.empty(v.shape, dtype=np.int16)
    quantile_bin_block(v, 10, reference, False)
    assert np.array_equal(codes, reference)
