"""``duplicate_transfer_audit`` groups device-to-host copies of equal content, whatever width they were taken at, and restores cupy when it exits."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._gpu_strict_fe import duplicate_transfer_audit, residency_audit
from mlframe.feature_selection.filters._lazy_host_codes import LazyHostCodes


def test_the_same_column_copied_twice_is_one_group_and_distinct_columns_are_not():
    """Two copies of one array are a group with its second copy redundant; a different array of the same size is not part of it."""
    a = cp.asarray(np.linspace(0.0, 1.0, 5000))
    b = cp.asarray(np.linspace(1.0, 2.0, 5000))
    with duplicate_transfer_audit() as rep:
        cp.asnumpy(a)
        a.get()
        cp.asnumpy(b)
    groups = rep.groups()
    assert len(groups) == 1 and len(groups[0]) == 2
    assert rep.redundant_bytes == a.nbytes
    assert rep.total_bytes == 3 * a.nbytes


def test_a_dense_and_a_narrowed_copy_of_one_column_count_as_the_same_content():
    """A float64 copy and a float32 copy, and an int8 and an int64 copy, of one array are reported as repeats."""
    x = cp.asarray(np.linspace(-2.0, 2.0, 4000))
    codes = cp.asarray(np.arange(4000, dtype=np.int64) % 7)
    with duplicate_transfer_audit() as rep:
        cp.asnumpy(x)
        cp.asnumpy(x.astype(cp.float32))
        cp.asnumpy(codes)
        cp.asnumpy(codes.astype(cp.int8))
    assert len(rep.groups()) == 2
    # what one shared narrow copy per array would have saved: the whole of each wider copy
    assert rep.redundant_bytes == x.nbytes + codes.nbytes


def test_small_transfers_and_the_out_buffer_form_of_get_are_handled():
    """A scalar-sized copy is ignored; ``get(out=...)`` is recorded from the buffer that was filled."""
    small = cp.asarray(np.arange(10.0))
    big = cp.asarray(np.arange(3000.0))
    buf = np.empty(3000)
    with duplicate_transfer_audit() as rep:
        cp.asnumpy(small)
        big.get(out=buf)
        cp.asnumpy(big)
    assert [t.nbytes for t in rep.transfers] == [big.nbytes, big.nbytes]
    assert len(rep.groups()) == 1


def test_two_readers_of_one_lazy_array_share_a_copy_and_the_report_names_the_call_sites():
    """Two reads of a ``LazyHostCodes`` at the same dtype cross once; a dense read after a narrowed one is reported with its call site."""
    lazy = LazyHostCodes(cp.asarray(np.linspace(0.0, 1.0, 6000)), np.float64)
    with duplicate_transfer_audit() as rep:
        lazy.host_as(np.float32)
        lazy.host_as(np.float32)
        np.asarray(lazy)
    assert len(rep.transfers) == 2
    assert rep.redundant_bytes == 6000 * 8  # the dense float64 copy that the narrowed one made unnecessary
    assert "test_duplicate_transfer_audit.py" in rep.summary() or "_lazy_host_codes.py" in rep.summary()


def test_cupy_is_restored_and_the_audit_nests_inside_the_residency_audit():
    """The patched entry points are restored on exit, and both audits see the same transfer when nested."""
    orig_asnumpy, orig_get = cp.asnumpy, cp.ndarray.get
    x = cp.asarray(np.arange(4000.0))
    with residency_audit() as res, duplicate_transfer_audit() as dup:
        cp.asnumpy(x)
    assert cp.asnumpy is orig_asnumpy and cp.ndarray.get is orig_get
    assert len(dup.transfers) == 1 and res.d2h == [x.nbytes]
