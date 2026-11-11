"""Behavioural pins for the family max-T gate's block budget and device-label guard, which the mutation sweep found unpinned.

The budget narrows a permutation block to a byte cap that depends on whether the row count fits in a uint16 index. The device-label
guard decides whether permutations are drawn on the GPU, and must restore the generator when the device declines.
"""

import numpy as np
import pytest

import mlframe.feature_selection.filters._family_maxt_gate as gate
from mlframe.feature_selection.filters import _perm_labels_device as dev


@pytest.mark.parametrize("n", [65535, 65536, 70000])
def test_budgeted_perm_block_switches_index_width_at_uint16_max(n):
    """Rows up to the uint16 maximum cost two bytes of index each; above it four. A huge request is narrowed to exactly the byte budget."""
    per_row = 2 if n <= np.iinfo(np.uint16).max else 4
    assert gate._budgeted_perm_block(10**9, n) == gate._PERM_BLOCK_BUDGET_BYTES // (per_row * n)


def test_device_labels_stay_off_without_the_env_flag(monkeypatch):
    """Without MLFRAME_PERM_LABELS_DEVICE=1 the device is never asked, even on a large family."""
    monkeypatch.delenv("MLFRAME_PERM_LABELS_DEVICE", raising=False)
    calls = []
    monkeypatch.setattr(dev, "device_label_bits", lambda *a: calls.append(a) or np.zeros(1))
    y = np.zeros(gate._DEVICE_LABELS_MIN_ROWS, dtype=np.int64)
    assert gate._device_label_bits_or_none(np.random.default_rng(0), y, 2, 2, 1) is None
    assert calls == []


def test_device_labels_need_rows_at_the_minimum(monkeypatch):
    """The device is used at the row minimum and not one row below it."""
    monkeypatch.setenv("MLFRAME_PERM_LABELS_DEVICE", "1")
    calls = []
    monkeypatch.setattr(dev, "device_label_bits", lambda *a: calls.append(a) or np.zeros(1))
    below = np.zeros(gate._DEVICE_LABELS_MIN_ROWS - 1, dtype=np.int64)
    assert gate._device_label_bits_or_none(np.random.default_rng(0), below, 2, 2, 1) is None
    assert calls == []
    at = np.zeros(gate._DEVICE_LABELS_MIN_ROWS, dtype=np.int64)
    assert gate._device_label_bits_or_none(np.random.default_rng(0), at, 2, 2, 1) is not None
    assert len(calls) == 1


def test_device_decline_leaves_the_generator_untouched(monkeypatch):
    """When the device returns None the caller's generator is in the state it had before the call, so the host draw that follows is unchanged."""
    monkeypatch.setenv("MLFRAME_PERM_LABELS_DEVICE", "1")
    monkeypatch.setattr(dev, "device_label_bits", lambda *a: None)
    rng = np.random.default_rng(3)
    reference = np.random.default_rng(3)
    y = np.zeros(gate._DEVICE_LABELS_MIN_ROWS, dtype=np.int64)
    assert gate._device_label_bits_or_none(rng, y, 2, 2, 1) is None
    assert rng.integers(0, 10**9) == reference.integers(0, 10**9)


def test_device_success_passes_the_block_arguments_and_advances_one_seed(monkeypatch):
    """A device draw receives ``(y, m, seed, n_y, words)`` in that order, with the seed taken from the caller's generator, which advances by one draw."""
    monkeypatch.setenv("MLFRAME_PERM_LABELS_DEVICE", "1")
    calls = []
    monkeypatch.setattr(dev, "device_label_bits", lambda *a: calls.append(a) or np.zeros(1))
    rng = np.random.default_rng(5)
    reference = np.random.default_rng(5)
    y = np.zeros(gate._DEVICE_LABELS_MIN_ROWS, dtype=np.int64)
    gate._device_label_bits_or_none(rng, y, 3, 2, 1)
    expected_seed = int(reference.integers(0, 2**31 - 1))
    assert len(calls) == 1
    _y, m, seed, n_y, words = calls[0]
    assert (m, seed, n_y, words) == (3, expected_seed, 2, 1)
    assert rng.integers(0, 10**9) == reference.integers(0, 10**9)
