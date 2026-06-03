"""Regression coverage for the leak-corr adaptive sampler (2026-06-03).

User TVT prod hit numpy MemoryError 6.41 GiB on a 128 GB host with ~20 GB free
physical RAM -- not classical OOM but virtual-address-space fragmentation:
``np.column_stack`` requires ONE contiguous virtual block of the matrix size,
and on a long-running Jupyter kernel committed-but-paged-out buffers fragment
the address space so the single 6.4 GB request fails. The adaptive sampler
falls back to stride-subsampled rows when the would-be allocation exceeds a
fraction of available RAM, keeping the leak-corr estimate within ~1e-3 of
full-frame precision.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


def test_no_sample_when_ram_is_plentiful():
    """Small arrays + tons of free RAM -> return inputs untouched (bit-identical
    legacy path). Catches an accidental regression where the sampler triggers
    on every call regardless of headroom."""
    from mlframe.training._composite_discovery_filter import _maybe_sample_for_leak_corr

    arrs = [np.arange(100, dtype=np.float32) for _ in range(5)]
    y = np.arange(100, dtype=np.float32)
    # Mock virtual_memory to report 100 GB available (vastly more than needed).
    fake_vm = SimpleNamespace(available=int(100 * 1024 ** 3))
    with patch("psutil.virtual_memory", return_value=fake_vm):
        out_arrs, out_y = _maybe_sample_for_leak_corr(
            ["c"] * 5, arrs, y,
        )
    # Same objects returned (no allocation churn).
    assert out_arrs is arrs
    assert out_y is y


def test_sample_when_alloc_exceeds_available_headroom():
    """When the would-be allocation exceeds the headroom fraction the sampler
    must trim rows. Simulates the prod scenario at miniaturised scale -- a tiny
    available-RAM mock forces the sampler to trip on a small fixture array,
    keeping the test memory budget under a few MB while exercising the actual
    branch."""
    from mlframe.training._composite_discovery_filter import _maybe_sample_for_leak_corr

    # 1 M rows x 5 cols x 4 B = 20 MB matrix.
    n_rows = 1_000_000
    n_cols = 5
    arrs = [np.zeros(n_rows, dtype=np.float32) for _ in range(n_cols)]
    y = np.zeros(n_rows, dtype=np.float32)
    # Available 10 MB -> 30% headroom = 3 MB. Matrix needs 20 MB -> sample.
    fake_vm = SimpleNamespace(available=int(10 * 1024 ** 2))
    with patch("psutil.virtual_memory", return_value=fake_vm):
        out_arrs, out_y = _maybe_sample_for_leak_corr(
            ["c"] * n_cols, arrs, y,
        )
    # Sampler must reduce rows.
    assert out_arrs[0].shape[0] < n_rows
    # y_train sampled to the same row set.
    assert out_y.shape[0] == out_arrs[0].shape[0]
    # All columns shrink to the same row count.
    for a in out_arrs:
        assert a.shape[0] == out_arrs[0].shape[0]


def test_psutil_unavailable_falls_back_to_legacy_path():
    """When psutil raises we can't measure headroom; fall back to the
    full-frame path so callers' existing MemoryError try/except continues to
    work the way it did before the sampler was added."""
    from mlframe.training._composite_discovery_filter import _maybe_sample_for_leak_corr

    arrs = [np.zeros(1000, dtype=np.float32) for _ in range(5)]
    y = np.zeros(1000, dtype=np.float32)
    with patch("psutil.virtual_memory", side_effect=RuntimeError("psutil down")):
        out_arrs, out_y = _maybe_sample_for_leak_corr(
            ["c"] * 5, arrs, y,
        )
    # No-op path: same objects, no exception.
    assert out_arrs is arrs
    assert out_y is y


def test_empty_candidates_no_crash():
    """The sampler must handle the no-candidates degenerate case gracefully."""
    from mlframe.training._composite_discovery_filter import _maybe_sample_for_leak_corr

    out_arrs, out_y = _maybe_sample_for_leak_corr([], [], None)
    assert out_arrs == []
    assert out_y is None


def test_sample_emits_info_log(caplog):
    """When sampling fires the operator must see a log line explaining why
    (alloc size, available, sampled size) -- silently shrinking the corr
    estimate without telling them would be opaque."""
    import logging
    from mlframe.training._composite_discovery_filter import _maybe_sample_for_leak_corr

    n_rows = 1_000_000
    n_cols = 5
    arrs = [np.zeros(n_rows, dtype=np.float32) for _ in range(n_cols)]
    y = np.zeros(n_rows, dtype=np.float32)
    fake_vm = SimpleNamespace(available=int(10 * 1024 ** 2))
    with patch("psutil.virtual_memory", return_value=fake_vm):
        with caplog.at_level(logging.INFO, logger="mlframe.training._composite_discovery_filter"):
            _maybe_sample_for_leak_corr(["c"] * n_cols, arrs, y)
    sample_lines = [
        r for r in caplog.records
        if "leak-corr matrix sampled" in r.getMessage()
    ]
    assert sample_lines, "operator must see a single INFO line when sampling fires"
    msg = sample_lines[0].getMessage()
    # Numbers reported so the operator can sanity-check the decision.
    assert "GB" in msg
    assert "stride=" in msg
