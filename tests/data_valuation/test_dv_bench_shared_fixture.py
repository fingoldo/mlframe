"""The shared benchmark fixture builds the shapes and binary labels its callers rely on (audit 2026-11-20, finding 05-F6)."""

import numpy as np

from mlframe.data_valuation._benchmarks._dv_bench_shared import make_blob_fixture


def test_blob_fixture_shapes_and_binary_labels():
    """Train and validation blocks have the requested shapes; labels are 0/1."""
    X_tr, y_tr, X_va, y_va = make_blob_fixture(50, 20, 4, seed=3)
    assert X_tr.shape == (50, 4) and X_va.shape == (20, 4)
    assert set(np.unique(np.concatenate([y_tr, y_va]))) <= {0, 1}


def test_blob_fixture_is_deterministic_per_seed():
    """The same seed reproduces the same data."""
    a = make_blob_fixture(30, 10, 3, seed=1)
    b = make_blob_fixture(30, 10, 3, seed=1)
    np.testing.assert_array_equal(a[0], b[0])
