"""``_is_constant`` of a float column equals the mask-and-range definition for every shape of column, and stops at the first differing value."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _mrmr_degenerate as deg


def _reference(values: np.ndarray) -> bool:
    """The definition: at most one distinct non-NaN value, none at all is not constant."""
    non_nan = values[~np.isnan(values)]
    return bool(non_nan.size) and bool(np.unique(non_nan).size == 1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_the_definition_on_every_case(dtype):
    """Constant, constant with missing values, two values, all missing, empty, one value, infinities of one and of both signs, signed zeros."""
    cases = {
        "constant": np.full(50, 2.5),
        "constant_with_nan": np.where(np.arange(50) % 3 == 0, np.nan, 2.5),
        "two_values": np.array([1.0, 1.0, 2.0, 1.0]),
        "last_differs": np.concatenate([np.ones(1000), [3.0]]),
        "all_nan": np.full(10, np.nan),
        "empty": np.empty(0),
        "single": np.array([4.0]),
        "all_plus_inf": np.full(5, np.inf),
        "inf_and_minus_inf": np.array([np.inf, -np.inf, np.inf]),
        "inf_and_finite": np.array([np.inf, 1.0]),
        "signed_zeros": np.array([0.0, -0.0, 0.0]),
        "nan_then_values": np.array([np.nan, 1.0, 1.0, 2.0]),
    }
    for name, arr in cases.items():
        arr = arr.astype(dtype)
        assert deg._is_constant(arr) == _reference(arr), name
        assert deg._is_constant(arr[::-1]) == _reference(arr[::-1]), name


def test_a_strided_view_and_an_integer_column_keep_working():
    """A non-contiguous float view is read correctly; an integer column still goes through the unique count."""
    base = np.stack([np.full(100, 7.0), np.arange(100.0)], axis=1)
    assert deg._is_constant(base[:, 0]) is True and deg._is_constant(base[:, 1]) is False
    assert deg._is_constant(np.full(10, 3)) is True and deg._is_constant(np.arange(10)) is False


def test_the_scan_stops_at_the_first_differing_value():
    """A column that differs in its second entry costs next to nothing however long it is (the old form touched every row)."""
    import time

    x = np.concatenate([[0.0, 1.0], np.zeros(20_000_000)])
    deg._is_constant(x[:10])
    t = time.perf_counter()
    for _ in range(100):
        assert deg._is_constant(x) is False
    assert (time.perf_counter() - t) / 100 < 2e-3  # the full pass over 20 million values is about 15 ms
