"""The resample block is built once on the device and shared; tail dependence reads it in place and gives the host kernel's values."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _orth_auto_scorer_fe as auto
from mlframe.feature_selection.filters import _tail_dependence_gpu as tail_gpu
from mlframe.feature_selection.filters._orthogonal_tail_dependence_fe import tail_dependence_columns


def _block(n: int, k: int, seed: int):
    """An ``(n, k)`` float32 block with ties and a target that shares the extremes of some columns."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    cols = [y + 0.3 * rng.standard_normal(n), np.round(rng.standard_normal(n), 1), rng.standard_normal(n), np.where(np.abs(y) > 1.8, y, 0.0) + 0.05 * rng.standard_normal(n)]
    cols += [rng.standard_normal(n) for _ in range(k - len(cols))]
    return np.ascontiguousarray(np.stack(cols, axis=1).astype(np.float32)), y


def test_device_block_is_the_transposed_block(monkeypatch):
    """The block uploaded as it is and transposed on the device equals the host transpose, element for element."""
    monkeypatch.setattr(tail_gpu, "use_device", lambda n_rows: True)
    block, _ = _block(20000, 6, seed=1)
    dev = auto._device_block(block)
    assert dev is not None
    assert dev.shape == (6, 20000)
    np.testing.assert_array_equal(cp.asnumpy(dev), np.ascontiguousarray(block.T))


def test_tail_dependence_from_a_device_block_equals_the_host_value(monkeypatch):
    """Reading the shared device block in place, and fetching it back when the device path is not taken, both give the host kernel's value."""
    block, y = _block(20000, 6, seed=2)
    rows = np.ascontiguousarray(block.T)
    monkeypatch.setattr(tail_gpu, "use_device", lambda n_rows: False)
    want = tail_dependence_columns(rows, y)
    assert (want > 0).any() and (want == 0).any(), "the data must exercise a positive and a zero value"
    dev = cp.asarray(rows)
    np.testing.assert_array_equal(tail_dependence_columns(None, y, device_block=dev), want)  # host kernel on a block fetched from the device
    monkeypatch.setattr(tail_gpu, "use_device", lambda n_rows: True)
    np.testing.assert_array_equal(tail_dependence_columns(None, y, device_block=dev), want)  # device kernel reading the block in place
