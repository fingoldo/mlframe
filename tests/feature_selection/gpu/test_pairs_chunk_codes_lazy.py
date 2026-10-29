"""Pair-search chunk codes stay on the device until the family gate asks for columns; the columns it gets equal the eager host fill."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_chunk_codes import (
    ChunkCodes,
    LazyCodeColumn,
    code_column,
    resolve_code_columns,
)
from mlframe.feature_selection.filters._gpu_resident_shared import _stash_deferred_host_fill
from mlframe.feature_selection.filters._gpu_resident_fe import clear_resident_codes_handoff


def _deferred_chunk(n=40, k=7):
    """An unfilled host buffer with a registered deferred fill from a device array; returns (host, device_values, captured)."""
    dev = cp.asarray(np.random.default_rng(1).integers(0, 10, (n, k)).astype(np.int8))
    host = np.zeros((n, k), dtype=np.int8)
    _stash_deferred_host_fill(host, dev)
    chunk = ChunkCodes.capture(host)
    clear_resident_codes_handoff(host)  # what the dispatch does when it ends
    return host, cp.asnumpy(dev), chunk


def test_lazy_columns_match_device_codes_without_filling_host():
    """Columns resolved from a deferred chunk equal the device codes, the host buffer stays unfilled, and the order and repeats of the request are kept."""
    host, ref, chunk = _deferred_chunk()
    assert isinstance(chunk, ChunkCodes)
    cols = [code_column(chunk, j) for j in (5, 1, 5)]
    assert all(isinstance(c, LazyCodeColumn) and c.shape == (40, 1) for c in cols)
    got = resolve_code_columns(cols)
    for g, j in zip(got, (5, 1, 5)):
        np.testing.assert_array_equal(g, ref[:, j : j + 1])
    assert not chunk.host_ready()
    assert not host.any(), "resolving columns must not fill the whole host block"


def test_full_host_fill_after_dispatch_matches_device():
    """A stray full read fills the host block with the device bytes and later column reads come from the host copy."""
    _host, ref, chunk = _deferred_chunk()
    np.testing.assert_array_equal(np.asarray(chunk), ref)
    assert chunk.host_ready()
    np.testing.assert_array_equal(resolve_code_columns([code_column(chunk, 3)])[0], ref[:, 3:4])


def test_filled_host_buffer_is_not_wrapped_and_other_items_pass_through():
    """A buffer with no pending deferred fill is returned as is; arrays and foreign objects in a gate batch are passed through untouched."""
    host = np.arange(12, dtype=np.int8).reshape(4, 3)
    assert ChunkCodes.capture(host) is host
    col = code_column(host, 1)
    np.testing.assert_array_equal(col, host[:, 1:2])
    sentinel = object()
    out = resolve_code_columns([col, sentinel])
    assert out[0] is col and out[1] is sentinel


def test_switch_off_fills_the_host_block(monkeypatch):
    """MLFRAME_PAIR_LAZY_CODES=0 restores the eager fill of the whole block."""
    monkeypatch.setenv("MLFRAME_PAIR_LAZY_CODES", "0")
    dev = cp.asarray(np.arange(20, dtype=np.int8).reshape(5, 4))
    host = np.zeros((5, 4), dtype=np.int8)
    _stash_deferred_host_fill(host, dev)
    got = ChunkCodes.capture(host)
    clear_resident_codes_handoff(host)
    assert got is host
    np.testing.assert_array_equal(host, cp.asnumpy(dev))
