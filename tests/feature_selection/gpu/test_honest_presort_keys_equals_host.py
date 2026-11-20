"""The composite-key device presort returns the order and tie flags of the per-column device sort (and of the host presort) on ties, constants, signed zeros and blocks of columns."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _honest_null_presort_gpu as pre
from mlframe.feature_selection.filters._honest_null_accept import _presort_columns, _tie_flags


def _columns(n=4000, k=9, seed=0):
    """Random float32 columns with a heavily tied one, a constant one, a column mixing -0.0 and +0.0 and a binary one."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)).astype(np.float32)
    x[:, 1] = np.round(x[:, 1] * 3) / 3
    x[:, 2] = 1.0
    x[:, 3] = np.where(rng.random(n) < 0.5, 0.0, -0.0).astype(np.float32)
    x[:, 4] = (x[:, 4] > 0).astype(np.float32)
    return x


def test_keys_presort_equals_host_presort():
    """Order and tie flags equal the host stable sort and tie flags for every column kind."""
    x = _columns()
    order, tie = pre.presort_and_ties_gpu(x)
    ref_order = _presort_columns(x)
    np.testing.assert_array_equal(order, ref_order)
    np.testing.assert_array_equal(tie, _tie_flags(x, ref_order))


def test_blocks_of_columns_do_not_change_the_result(monkeypatch):
    """A tiny memory budget forces blocks of a few columns: the result is the one-block result."""
    x = _columns(n=3000, k=11, seed=1)
    whole = pre.presort_and_ties_gpu(x)
    monkeypatch.setattr(pre, "_presort_column_block_bytes", lambda cp_, n, b: 3)
    blocked = pre.presort_and_ties_gpu(x)
    np.testing.assert_array_equal(whole[0], blocked[0])
    np.testing.assert_array_equal(whole[1], blocked[1])


def test_float64_columns_take_the_per_column_route():
    """float64 columns do not fit a 64-bit key: the composite-key route declines and the per-column device sort still gives the host's order."""
    x = _columns().astype(np.float64)
    order, tie = pre.presort_and_ties_gpu(x)
    ref_order = _presort_columns(x)
    np.testing.assert_array_equal(order, ref_order)
    np.testing.assert_array_equal(tie, _tie_flags(x, ref_order))
