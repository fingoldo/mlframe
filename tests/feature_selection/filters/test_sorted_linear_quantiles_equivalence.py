"""``_sorted_linear_quantiles`` must equal ``np.quantile(x, q)`` exactly: the grouped-quantile FE reads its quartiles from a sorted array now."""

import numpy as np

from mlframe.feature_selection.filters._grouped_quantile_fe import _sorted_linear_quantiles

_QS = (0.25, 0.75, 0.10, 0.90)


def test_sorted_linear_quantiles_equal_numpy_exactly_float64_and_float32():
    """Random arrays with ties: the sorted-array quantiles equal np.quantile bit for bit, for float64 and float32."""
    rng = np.random.default_rng(7)
    for trial in range(600):
        n = int(rng.integers(2, 400))
        base = rng.normal(size=n) * rng.choice([1.0, 1e3, 1e-3])
        if trial % 3 == 0:
            base = np.round(base, 1)  # heavy ties
        for dtype in (np.float64, np.float32):
            x = base.astype(dtype)
            ref = np.quantile(x, list(_QS))
            got = _sorted_linear_quantiles(np.sort(x), _QS)
            assert np.array_equal(ref, got), (trial, dtype)


def test_sorted_linear_quantiles_single_value_and_two_values():
    """Degenerate sizes (one and two values) match np.quantile exactly."""
    assert np.array_equal(_sorted_linear_quantiles(np.array([3.0]), _QS), np.quantile(np.array([3.0]), list(_QS)))
    two = np.array([1.0, 4.0])
    assert np.array_equal(_sorted_linear_quantiles(two, _QS), np.quantile(two, list(_QS)))
