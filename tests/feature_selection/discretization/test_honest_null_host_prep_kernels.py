"""The two parallel kernels of the honest null's host preparation equal the numpy forms they replaced."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _honest_null_device as dev


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int64])
def test_gather_and_class_counts_equal_numpy(dtype):
    """Labels of the evaluation rows per permutation (same integer type, C-contiguous) and their per-permutation class counts (int64, with a class nobody holds counted as 0)."""
    rng = np.random.default_rng(0)
    y_perm = rng.integers(0, 4, (37, 5000)).astype(dtype)
    y_perm[y_perm == 3] = 1  # class 3 is absent: the count matrix keeps its column when asked for 4 classes
    rows = np.sort(rng.choice(5000, 1500, replace=False)).astype(np.int64)
    got = dev._labels_at_rows(y_perm, rows)
    want = np.ascontiguousarray(y_perm[:, rows])
    assert got.dtype == want.dtype and got.flags.c_contiguous and np.array_equal(got, want)
    counts = dev._class_counts_rows(got, 4)
    ref = np.stack([np.bincount(want[b], minlength=4) for b in range(want.shape[0])])
    assert counts.dtype == ref.dtype and np.array_equal(counts, ref)
    assert int(counts[:, 3].sum()) == 0 and int(counts.sum()) == want.size
