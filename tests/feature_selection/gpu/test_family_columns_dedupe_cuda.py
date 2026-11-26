"""The device comparison of a family's partitions gives the host one's ``(keep, inverse)``: relabelled and reversed copies are grouped, distinct partitions are not, and anything the device
does not take is handed back to the host path."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:  # audit-exempt: except_skip_masks_call_under_test -- optional CUDA device guard: the probe decides whether the module can run, it is not the code under test
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._family_columns_dedupe import unique_columns
from mlframe.feature_selection.filters._family_columns_dedupe_cuda import MAX_CODES, unique_columns_device


def _family(k, n, codes, seed, dtype):
    """``k`` columns of ``n`` rows: random partitions, relabelled copies, reversed copies and exact copies of them, shuffled."""
    rng = np.random.default_rng(seed)
    base = [rng.integers(0, codes, n) for _ in range(max(2, k // 3))]
    cols = []
    for i in range(k):
        b = base[i % len(base)]
        kind = i % 4
        if kind == 1:
            perm = rng.permutation(codes)
            cols.append(perm[b])
        elif kind == 2:
            cols.append(codes - 1 - b)
        else:
            cols.append(b.copy())
    order = rng.permutation(k)
    return np.stack([cols[i] for i in order]).astype(dtype)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32])
@pytest.mark.parametrize("k, n, codes", [(48, 3001, 7), (200, 20000, 20), (33, 64, 3)])
def test_the_device_grouping_equals_the_host_one(dtype, k, n, codes):
    """Same keep and inverse as ``unique_columns`` for several code types, sizes and code counts, on two seeds."""
    for seed in (0, 1):
        cols = _family(k, n, codes, seed, dtype)
        host = unique_columns(cols)
        dev = unique_columns_device(cp, cp.asarray(cols))
        assert host is not None and dev is not None and dev is not False
        assert np.array_equal(dev[0], host[0]) and np.array_equal(dev[1], host[1])


def test_all_distinct_gives_none_like_the_host():
    """Every column its own partition: ``None`` from both."""
    rng = np.random.default_rng(3)
    cols = rng.integers(0, 9, (40, 5000)).astype(np.int8)
    assert unique_columns(cols) is None and unique_columns_device(cp, cp.asarray(cols)) is None


def test_codes_the_device_does_not_take_go_back_to_the_host():
    """Negative codes or codes at the table limit: ``False`` (the caller uses the host path)."""
    cols = np.zeros((4, 100), dtype=np.int16)
    cols[0, 0] = -1
    assert unique_columns_device(cp, cp.asarray(cols)) is False
    cols[0, 0] = MAX_CODES
    assert unique_columns_device(cp, cp.asarray(cols)) is False
