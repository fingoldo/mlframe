"""``quantile_bin_batched`` returns its codes at the requested dtype with the same values (PLAN-10: uint8 codes are an eighth of the int64 memory)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_shared import quantile_bin_batched


def _frame(n: int = 3000, k: int = 6) -> np.ndarray:
    """Continuous columns plus a tied one and a constant one (the fallback branches)."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((n, k))
    a[:, 1] = np.round(a[:, 1])
    a[:, 2] = 1.0
    return a


@pytest.mark.parametrize("nbins", [3, 10, 64, 255])
def test_uint8_codes_equal_int64_codes(nbins):
    """Same values, one eighth of the bytes; the default stays int64."""
    a = _frame()
    ref = quantile_bin_batched(a, nbins)
    small = quantile_bin_batched(a, nbins, dtype=np.uint8)
    assert ref.dtype == np.int64 and small.dtype == np.uint8
    np.testing.assert_array_equal(ref, small.astype(np.int64))


def test_uint8_refuses_codes_that_do_not_fit():
    """More than 255 bins cannot be held in one byte: an error, not wrapped codes."""
    with pytest.raises(ValueError):
        quantile_bin_batched(_frame(), 300, dtype=np.uint8)
