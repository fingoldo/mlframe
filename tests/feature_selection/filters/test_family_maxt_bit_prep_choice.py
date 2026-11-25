"""The bit-set preparation of a family: host bit sets on the CPU path, a device copy under device mode, and a cache that a family's blocks reuse.

The collaborators are counted, not replaced, so each test pins which of them ran for a given call.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_selection.filters._family_maxt_gate as gate

cp = pytest.importorskip("cupy")


def _family(seed: int = 4):
    """Four candidate columns of four bins on 300 rows with a three-class label."""
    rng = np.random.default_rng(seed)
    n = 300
    y = rng.integers(0, 3, n).astype(np.int64)
    cols = rng.integers(0, 4, (4, n)).astype(np.int64)
    n_x = np.full(4, 4, dtype=np.int64)
    return cols, n_x, y


def test_host_path_builds_host_bit_sets_and_no_device_copy():
    """Without device mode the host bit sets are built and no device copy exists."""
    cols, n_x, y = _family()
    cache: dict = {}
    bit_prep, dev_bits = gate._family_bit_prep(cols, n_x, y, 3, False, cache)
    assert bit_prep is not None
    assert dev_bits is None
    assert "dev_bits" not in cache


def test_device_mode_streams_the_groups_so_the_host_holds_none():
    """Under device mode the column groups go to the device as they are built, so the host copy of the family keeps no groups."""
    cols, n_x, y = _family(9)
    bit_prep, dev_bits = gate._family_bit_prep(cols, n_x, y, 3, True, {})
    assert dev_bits is not None
    assert list(bit_prep[3]) == []


def test_the_cache_is_reused_by_later_calls_on_the_same_family(monkeypatch):
    """A second call with the same cache returns the first call's bit sets without building them again."""
    cols, n_x, y = _family(5)
    cache: dict = {}
    real = gate.bitset_groups
    calls: list[int] = []

    def counting(*args, **kwargs):
        """Count the host builds, then build as before."""
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gate, "bitset_groups", counting)
    first, _ = gate._family_bit_prep(cols, n_x, y, 3, False, cache)
    second, _ = gate._family_bit_prep(cols, n_x, y, 3, False, cache)
    assert len(calls) == 1
    assert second is first


def test_device_mode_builds_a_device_copy_once_per_family(monkeypatch):
    """Under device mode a device copy is built, and a later call with the same cache does not build it again."""
    cols, n_x, y = _family(6)
    cache: dict = {}
    real = gate._device_bitsets
    calls: list[int] = []

    def counting(*args, **kwargs):
        """Count the device builds, then build as before."""
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(gate, "_device_bitsets", counting)
    _, dev_first = gate._family_bit_prep(cols, n_x, y, 3, True, cache)
    _, dev_second = gate._family_bit_prep(cols, n_x, y, 3, True, cache)
    assert dev_first is not None
    assert dev_second is dev_first
    assert len(calls) <= 1


def test_a_host_call_on_a_shared_cache_does_not_block_a_later_device_call():
    """A host-only call sharing the cache with a later device call leaves the device copy available to that device call."""
    cols, n_x, y = _family(8)
    cache: dict = {}
    gate._family_bit_prep(cols, n_x, y, 3, False, cache)
    _, dev_bits = gate._family_bit_prep(cols, n_x, y, 3, True, cache)
    from mlframe.feature_selection.filters._family_maxt_cuda_bits import DeviceBitsets
    assert isinstance(dev_bits, DeviceBitsets)


def test_too_many_bins_keeps_the_histogram_kernels_and_returns_nothing():
    """A column with more bins than the bit-set limit keeps the histogram kernels: neither bit-set form is built."""
    cols, _, y = _family(7)
    n_x_big = np.full(4, gate.BITSET_MAX_BINS + 1, dtype=np.int64)
    bit_prep, dev_bits = gate._family_bit_prep(cols, n_x_big, y, 3, True, None)
    assert bit_prep is None
    assert dev_bits is None
