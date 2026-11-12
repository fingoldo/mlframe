"""The shared unary op-code table covers codes 0 to 16 once each, and the numba ladder gives each high op its named function."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._numba_utils_usability import _apply_unary
from mlframe.feature_selection.filters._unary_op_table import UNARY_OP_CODES

_HIGH_REFERENCE = {
    "rint": np.rint,
    "qubed": lambda v: v * v * v,
    "invsquared": lambda v: 1.0 / (v * v),
    "invqubed": lambda v: 1.0 / (v * v * v),
    "cbrt": np.cbrt,
    "invcbrt": lambda v: np.power(v, -1.0 / 3.0),
    "invsqrt": lambda v: np.power(v, -0.5),
    "exp": np.exp,
}


def test_codes_are_the_contiguous_range_zero_to_sixteen():
    """Each code is used once, and the table covers every code the ladders dispatch on."""
    assert sorted(UNARY_OP_CODES.values()) == list(range(17))


@pytest.mark.parametrize("name", sorted(_HIGH_REFERENCE))
def test_numba_ladder_matches_the_named_function(name):
    """The numba ladder computes the function its table name says, on positive and negative inputs."""
    v = np.array([0.5, 1.7, 3.2, 9.0, -2.5, -0.75], dtype=np.float64)
    want = _HIGH_REFERENCE[name](v)
    got = np.array([_apply_unary(float(x), UNARY_OP_CODES[name], 0.0) for x in v])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0)
