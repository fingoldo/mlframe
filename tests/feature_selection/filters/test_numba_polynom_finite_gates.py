"""The numba polynomial optimizer's finiteness gates must see NaN/inf (they sit in ``fastmath`` kernels, where nnan/ninf let LLVM fold the test away)."""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _numba_polynom_optimizer as opt


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_all_finite_rejects_non_finite_values(bad):
    """A single NaN or +/-inf anywhere makes the scan return False; an all-finite array returns True."""
    arr = np.array([1.0, 2.0, bad, 3.0])
    assert opt._all_finite_njit(arr) is False
    assert opt._all_finite_njit(np.array([1.0, 2.0, 3.0])) is True


def test_fill_bf_batch_marks_rows_with_non_finite_combinations():
    """A candidate row whose combination holds NaN or inf is flagged non-finite and left unfilled; clean rows are filled."""
    ha = np.array([[1.0, np.nan, 2.0], [1.0, 2.0, 3.0], [np.inf, 1.0, 1.0]])
    hb = np.ones((3, 3))
    valid = np.array([True, True, True])
    bf_ids = np.array([opt.BF_ADD], dtype=np.int64)
    rows = np.zeros((3, 3))
    finite = np.zeros(3, dtype=np.bool_)
    opt._fill_bf_batch_njit(ha, hb, valid, bf_ids, rows, finite)
    assert finite.tolist() == [False, True, False]
    np.testing.assert_array_equal(rows[1], [2.0, 3.0, 4.0])
    assert not rows[0].any() and not rows[2].any()


def test_candidate_with_nan_polyeval_output_is_rejected():
    """A NaN in the input makes the polynomial output non-finite, so the candidate scores -inf instead of an MI computed from NaN columns."""
    rng = np.random.default_rng(0)
    n = 200
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    x_a[10] = np.nan
    y = (x_b > 0).astype(np.int64)
    score, raw, bf = opt._eval_one_candidate_njit(
        x_a, x_b, y, np.array([0.0, 1.0, 0.5]), np.array([0.0, 1.0, 0.5]), opt.BASIS_HERMITE, np.array([opt.BF_ADD, opt.BF_MUL], dtype=np.int64), 8, 0.0, False, True,
    )
    assert score == -np.inf and raw == 0.0 and bf == -1
