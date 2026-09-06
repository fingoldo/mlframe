"""Both |corr| kernels must survive an offset large relative to the spread.

They accumulated raw power sums and recovered the variance by subtraction (``saa - sa*sa/n``). That is
catastrophic cancellation on any column carrying a large offset -- an epoch timestamp, a price, a count --
and the destroyed variance then trips the near-constant guard, so the kernel returns a confident ``0.0``.

That ``0.0`` is not a neutral failure: it reads as "not redundant, keep" in the dedup gate and as "no signal,
drop" in the y-gate. The docstring's claim that the kernel is "FP-equivalent to np.corrcoef to ~1e-15 --
selection-safe" is what kept anyone from looking.

The reference is ``np.corrcoef`` on the same data, so these tests pin agreement with the definition rather
than a frozen number.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_core import _abs_corr_finite_njit, _abs_corr_zerofill_njit

# Offsets spanning "none", "large", and a real epoch-second timestamp -- the regime that produced 0.0.
_OFFSETS = (0.0, 1.0e3, 1.0e7, 1.7e9)


def _pair(offset: float, rho: float = 0.3, n: int = 4000):
    """A column carrying `offset` and a target correlated with it at approximately `rho`."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=n)
    noise = rng.normal(size=n)
    return base + offset, rho * base + np.sqrt(1.0 - rho**2) * noise


@pytest.mark.parametrize("offset", _OFFSETS, ids=[f"offset_{o:.0e}" for o in _OFFSETS])
def test_masked_kernel_matches_numpy_under_offset(offset: float):
    """The masked kernel must agree with `np.corrcoef`, whatever the column's offset."""
    a, y = _pair(offset)
    expected = abs(float(np.corrcoef(a, y)[0, 1]))
    got = float(_abs_corr_finite_njit(a, y, np.isfinite(y), 2))
    assert got == pytest.approx(expected, abs=1e-9), f"offset={offset:.0e}: kernel {got:.6f} vs numpy {expected:.6f}"


@pytest.mark.parametrize("offset", _OFFSETS, ids=[f"offset_{o:.0e}" for o in _OFFSETS])
def test_zerofill_kernel_matches_numpy_under_offset(offset: float):
    """The zero-fill twin carries the same defect and needs the same guarantee."""
    a, b = _pair(offset)
    expected = abs(float(np.corrcoef(a, b)[0, 1]))
    got = float(_abs_corr_zerofill_njit(a, b))
    assert got == pytest.approx(expected, abs=1e-9), f"offset={offset:.0e}: kernel {got:.6f} vs numpy {expected:.6f}"


def test_a_real_correlation_on_offset_data_is_not_reported_as_zero():
    """The sharpest form: a genuine association must not come back as exactly 0.0.

    `0.0` is the value that means "not redundant, keep" in one gate and "no signal, drop" in another, so a
    silently zeroed correlation is actionable in both directions.
    """
    a, y = _pair(1.7e9, rho=0.5)
    assert float(_abs_corr_finite_njit(a, y, np.isfinite(y), 2)) > 0.4, "a strong correlation on epoch-scale data was reported as absent"
    assert float(_abs_corr_zerofill_njit(a, y)) > 0.4, "a strong correlation on epoch-scale data was reported as absent"


def test_a_genuinely_constant_column_still_returns_zero():
    """The near-constant guard must survive the rewrite: a real constant has no correlation to report."""
    n = 500
    constant = np.full(n, 3.5)
    y = np.random.default_rng(1).normal(size=n)
    assert float(_abs_corr_finite_njit(constant, y, np.isfinite(y), 2)) == 0.0
    assert float(_abs_corr_zerofill_njit(constant, y)) == 0.0


def test_non_finite_rows_keep_their_documented_treatment():
    """The two kernels differ deliberately -- one masks, one zero-fills -- and that must not have changed."""
    a = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])

    masked = float(_abs_corr_finite_njit(a, b, np.isfinite(b), 2))
    assert masked == pytest.approx(abs(float(np.corrcoef(a[np.isfinite(a)], b[np.isfinite(a)])[0, 1])), abs=1e-9)

    zerofilled = float(_abs_corr_zerofill_njit(a, b))
    assert zerofilled == pytest.approx(abs(float(np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))[0, 1])), abs=1e-9)
