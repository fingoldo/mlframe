"""``row_argmax_cm`` must read the operand columns in their own element type.

The crit dtype is float32 by default, so the resident operands are float32; a kernel that read them as ``double`` read twice the bytes the stack holds, returned wrong indices and
faulted with an illegal address when the stack ended at the end of a mapped region (an abort inside the next thrust sort of the process).
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._row_argmax_gpu import row_argmax_cm


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [1, 31, 255, 257, 100_000])
@pytest.mark.parametrize("m", [2, 3, 4])
def test_row_argmax_matches_numpy(dtype, n, m):
    """The device index equals ``np.argmax`` over the stacked host columns, ties to the lowest operand."""
    rng = np.random.default_rng(n * 10 + m)
    cols = [rng.integers(-3, 4, n).astype(dtype) for _ in range(m)]  # small integers: many ties
    expected = np.argmax(np.column_stack(cols), axis=1).astype(np.float64)
    got = cp.asnumpy(row_argmax_cm(cp, [cp.asarray(c) for c in cols]))
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, expected)


def test_row_argmax_float32_is_exact_at_the_end_of_an_exact_size_buffer():
    """A float32 stack allocated exactly (no pool slack) is read in full and not past its end."""
    n, m = 1_000_000, 3
    rng = np.random.default_rng(0)
    cols = [rng.standard_normal(n).astype(np.float32) for _ in range(m)]
    expected = np.argmax(np.column_stack(cols), axis=1).astype(np.float64)
    got = cp.asnumpy(row_argmax_cm(cp, [cp.asarray(c) for c in cols]))
    np.testing.assert_array_equal(got, expected)
