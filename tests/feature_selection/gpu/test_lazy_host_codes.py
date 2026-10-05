"""``LazyHostCodes`` stands in for a host copy of device codes and copies only when something reads it."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit
from mlframe.feature_selection.filters._lazy_host_codes import LazyHostCodes


def test_metadata_and_scalar_reductions_do_not_copy_the_array():
    """shape / size / dtype / len and max / min are answered without a bulk device->host copy."""
    codes = np.arange(10_000, dtype=np.int64) % 17
    lazy = LazyHostCodes(cp.asarray(codes), np.int64)
    with residency_audit() as rep:
        assert lazy.shape == (10_000,) and lazy.size == 10_000 and len(lazy) == 10_000 and lazy.dtype == np.int64
        assert int(lazy.max()) == 16 and int(lazy.min()) == 0
    assert not rep.bulk_d2h and rep.scalar_d2h_bytes < 64


def test_first_real_read_copies_once_and_matches_the_eager_copy():
    """np.asarray / indexing / ndarray methods return exactly the eager bytes, and the copy happens a single time."""
    codes = (np.arange(5000, dtype=np.int64) * 7) % 13
    lazy = LazyHostCodes(cp.asarray(codes), np.int64)
    with residency_audit() as rep:
        np.testing.assert_array_equal(np.asarray(lazy), codes)
        np.testing.assert_array_equal(lazy[10:20], codes[10:20])
        assert lazy.astype(np.int32).dtype == np.int32 and int(lazy.sum()) == int(codes.sum())
    assert len(rep.d2h) == 1


def test_residency_audit_counts_implicit_scalar_syncs():
    """``float()`` / ``int()`` / ``bool()`` / ``.item()`` on a device scalar are recorded as syncs, so a regression to per-candidate scalar reads is visible."""
    x = cp.arange(10.0)
    with residency_audit() as rep:
        float(x.sum())
        int(x.max())
        bool(x.any())
        x.sum().item()
    assert len(rep.scalar_syncs) == 4
