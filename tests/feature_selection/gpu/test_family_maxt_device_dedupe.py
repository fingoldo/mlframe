"""The device family gate scores the distinct partitions of a family: the repeats of a code column (identical, or reversed by a decreasing transform) are dropped before the bit sets and the null kernel, the flags stay those of the whole family."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _family_maxt_gate as gate
from mlframe.feature_selection.filters import _family_maxt_gate_device as device

if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

N = 4000
BINS = 6


class _Batch:
    """A one-column batch that holds its device codes, the way the lazy columns of a pair family do."""

    def __init__(self, codes):
        """Keep the ``(n, 1)`` device column."""
        self.dev = cp.asarray(codes.reshape(-1, 1))
        self.shape = self.dev.shape


def _family(seed: int = 0, base: int = 150, copies: int = 70, reversed_copies: int = 60):
    """``(columns, y)``: ``base`` distinct partitions (the first three carry the target), ``copies`` identical repeats and ``reversed_copies`` repeats with the codes reversed."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=N).astype(np.int64)
    cols = []
    for j in range(base):
        noise = rng.integers(0, BINS, size=N)
        cols.append(np.clip(noise if j >= 3 else (y * 3 + noise // 2), 0, BINS - 1).astype(np.int8))
    pick = rng.permutation(base)
    cols += [cols[int(j)].copy() for j in pick[:copies]]
    cols += [(BINS - 1 - cols[int(j)]).astype(np.int8) for j in pick[copies : copies + reversed_copies]]
    order = rng.permutation(len(cols))  # the repeats are not next to their originals
    return [cols[int(i)] for i in order], y


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two code columns split the rows the same way."""
    a, b = a.astype(np.int64), b.astype(np.int64)
    return len(np.unique(a)) == len(np.unique(b)) == len(np.unique(a * 64 + b))


def test_the_repeats_of_a_partition_are_dropped_and_mapped_back():
    """The compact matrix has one column per distinct partition; every original column maps to a column that splits the rows the same way."""
    cols, _y = _family()
    compact, inverse = device._distinct_columns_on_device(cp, [cp.asarray(c.reshape(-1, 1)) for c in cols])
    host = cp.asnumpy(compact)
    assert host.shape == (N, 150)
    assert inverse.shape == (len(cols),)
    assert all(_same_partition(cols[j], host[:, int(inverse[j])]) for j in range(len(cols)))


def test_a_small_or_repeat_free_family_is_left_alone():
    """Below the column threshold, or with every partition distinct, nothing is gathered."""
    cols, _y = _family()
    assert device._distinct_columns_on_device(cp, [cp.asarray(c.reshape(-1, 1)) for c in cols[: device.DEDUPE_MIN_COLUMNS - 1]]) is None
    free, _y = _family(copies=0, reversed_copies=0)
    assert device._distinct_columns_on_device(cp, [cp.asarray(c.reshape(-1, 1)) for c in free]) is None


@pytest.mark.parametrize("seed", [0, 1])
def test_the_flags_are_those_of_the_whole_family(monkeypatch, seed):
    """The pass flags of every column are the same with the repeats dropped, without dropping them, and on the host gate; the real signal columns and their repeats pass."""
    cols, y = _family(seed)
    batches = [_Batch(c) for c in cols]
    with_dedupe = device.family_pass_batches_device(batches, y, 0.01, 999, 0)
    monkeypatch.setattr(device, "DEDUPE_MIN_COLUMNS", 10**9)
    without = device.family_pass_batches_device(batches, y, 0.01, 999, 0)
    host = gate.family_pass_batches([c.reshape(-1, 1) for c in cols], y, alpha=0.01, n_perm=999)
    assert with_dedupe is not None and without is not None
    assert with_dedupe == without
    assert with_dedupe == [bool(f) for f in host]
    assert sum(with_dedupe) >= 3  # the signal columns and their repeats
