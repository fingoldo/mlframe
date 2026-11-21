"""``_rows_to_host`` takes a device or a host row index and returns the same values as indexing on the host."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._binned_agg_linear_usability import _rows_to_host


def test_device_and_host_row_index_agree():
    """A device column with a host index and with a device index equals the host gather."""
    col = np.arange(1000, dtype=np.float32) * 0.5
    rows = np.sort(np.random.default_rng(0).choice(1000, size=300, replace=False))
    want = col[rows].astype(np.float64)
    np.testing.assert_array_equal(_rows_to_host(cp.asarray(col), rows), want)
    np.testing.assert_array_equal(_rows_to_host(cp.asarray(col), cp.asarray(rows)), want)
