"""Small integer powers of device arrays are computed by multiplication; numpy inputs and other exponents are untouched."""

from __future__ import annotations

import numpy as np
import pytest

from tests.feature_selection.gpu._gpu_support import has_device

from mlframe.feature_selection.filters._device_power import small_power


def _cp():
    """cupy with a usable device, or skip."""
    cp = pytest.importorskip("cupy")
    if not has_device(cp):
        pytest.skip("no usable CUDA device")
    return cp


def test_device_small_powers_are_the_multiplication_forms():
    """Square, cube and the reciprocal powers equal the product / division forms to the bit, zeros and negative zeros included."""
    cp = _cp()
    rng = np.random.default_rng(0)
    x_h = rng.standard_normal(100_000) * 3
    x_h[:4] = [0.0, -0.0, 1.0, -2.0]
    x = cp.asarray(x_h)
    with np.errstate(divide="ignore"):
        assert np.array_equal(cp.asnumpy(small_power(cp, x, 2)), x_h * x_h)
        assert np.array_equal(cp.asnumpy(small_power(cp, x, 3)), x_h * x_h * x_h)
        assert np.array_equal(cp.asnumpy(small_power(cp, x, -1)), 1.0 / x_h)
        assert np.array_equal(cp.asnumpy(small_power(cp, x, -2)), 1.0 / (x_h * x_h))
        assert np.array_equal(cp.asnumpy(small_power(cp, x, -3)), 1.0 / (x_h * x_h * x_h))


def test_numpy_integer_and_other_exponents_keep_power():
    """A numpy array, an integer device array and an exponent outside the list go through ``power`` unchanged."""
    cp = _cp()
    x_h = np.linspace(0.5, 3.0, 64)
    assert np.array_equal(small_power(np, x_h, 2), np.power(x_h, 2))
    assert np.array_equal(cp.asnumpy(small_power(cp, cp.asarray(x_h), 0.5)), cp.asnumpy(cp.power(cp.asarray(x_h), 0.5)))
    xi = cp.arange(1, 9)
    assert np.array_equal(cp.asnumpy(small_power(cp, xi, 2)), cp.asnumpy(cp.power(xi, 2)))


def test_the_square_is_closer_to_the_correctly_rounded_value_than_pow():
    """The device ``pow`` square disagrees with the correctly rounded product on a sizeable share of a normal sample; the multiplication form does not."""
    cp = _cp()
    x = cp.asarray(np.random.default_rng(1).standard_normal(1_000_000) * 3)
    assert int((cp.power(x, 2.0) != x * x).sum()) > 0
    assert int((small_power(cp, x, 2) != x * x).sum()) == 0
