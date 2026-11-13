"""``_safe_div`` of two vectors runs as one pass (``_safe_div_1d``) and gives the values of the expression form (float64 division, ``1e-9`` for an exactly zero denominator).

Checked against the expression for float32 and float64 operands with zeros, negative zeros, NaN and inf on both sides, through the njit-wrapped registry entry as well as the plain function, and for the shapes that keep the expression form
(scalars, matrices, a length-one denominator).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import feature_engineering as fe


def _expression(x, y):
    """The previous body of ``_safe_div``, as the reference."""
    y = np.asarray(y, dtype=np.float64)
    return np.asarray(x, dtype=np.float64) / np.where(y == 0.0, 1e-9, y)


def _operands(dtype, n=1000):
    """Vectors with zeros, negative zeros, NaN and inf on both sides."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    y[::7] = 0.0
    y[3] = -0.0
    y[5] = np.nan
    y[11] = np.inf
    x[13] = np.nan
    x[17] = np.inf
    x[19] = -np.inf
    return x, y


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("entry", ["plain", "registry"])
def test_vectors_give_the_expression_values(dtype, entry):
    """Same float64 values, NaN and inf included, from the plain function and from the njit-wrapped registry entry."""
    x, y = _operands(dtype)
    func = fe._safe_div if entry == "plain" else fe.create_binary_transformations("minimal")["div"]
    got = func(x, y)
    want = _expression(x, y)
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, want)


def test_strided_columns_are_accepted():
    """Columns of a row-major matrix (the way the pair search passes them) divide like their copies."""
    rng = np.random.default_rng(2)
    m = rng.standard_normal((500, 4)).astype(np.float32)
    m[::9, 1] = 0.0
    np.testing.assert_array_equal(fe._safe_div(m[:, 0], m[:, 1]), _expression(m[:, 0].copy(), m[:, 1].copy()))


@pytest.mark.parametrize("shape", ["scalar", "matrix", "length_one"])
def test_other_shapes_keep_the_expression_form(shape):
    """A scalar pair, two matrices and a length-one denominator are not the one-pass case and still divide as before."""
    if shape == "scalar":
        x, y = 3.0, 0.0
    elif shape == "matrix":
        rng = np.random.default_rng(3)
        x, y = rng.standard_normal((6, 5)), rng.standard_normal((6, 5))
        y[0, 0] = 0.0
    else:
        x, y = np.arange(5, dtype=np.float64), np.array([0.0])
    np.testing.assert_array_equal(fe._safe_div(x, y), _expression(x, y))
