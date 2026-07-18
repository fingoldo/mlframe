"""Regression sensor for S08: ``PipelineCache`` must enforce an LRU byte-size cap.

Pre-fix: ``PipelineCache._cache`` was a plain ``dict`` with no size accounting; on a 100+ GB train frame each cached ``(train, val, test)`` triple pinned a full preprocessing-output frame and could OOM the host across a long-running suite (per CLAUDE.md 2GB streaming threshold).

Post-fix: ``PipelineCache`` is backed by an ``OrderedDict`` with LRU promotion on access, per-entry ``nbytes`` accounting (pandas ``memory_usage(deep=False)``, polars ``estimated_size()``, numpy ``nbytes``, fallback ``sys.getsizeof``), and an env-tunable byte cap (``MLFRAME_PIPELINE_CACHE_BYTES_LIMIT``, default 2_000_000_000). On insert overflow, LRU entries are evicted until under the cap.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from mlframe.training.strategies import PipelineCache


class _SizedBlob:
    """A trivial carrier with a known ``nbytes`` attribute the cache must recognise. Mirrors numpy/pandas/polars frame interfaces just enough for the size estimator."""

    def __init__(self, n_bytes: int) -> None:
        self._buf = np.zeros(max(n_bytes, 1), dtype=np.uint8)

    @property
    def nbytes(self) -> int:
        """Nbytes."""
        return int(self._buf.nbytes)


def test_pipeline_cache_uses_ordered_dict_for_lru() -> None:
    """LRU eviction requires insertion-order tracking. Plain ``dict`` semantics happen to preserve order in CPython 3.7+, but the cache must EXPLICITLY use OrderedDict so ``move_to_end`` works and the LRU contract survives any future restructuring."""
    cache = PipelineCache(verbose=False)
    assert isinstance(cache._cache, OrderedDict), f"PipelineCache._cache must be an OrderedDict for LRU semantics; got {type(cache._cache).__name__}"


def test_pipeline_cache_evicts_lru_when_byte_limit_exceeded(monkeypatch) -> None:
    """Insert 3 entries each ~10MB with a 25MB byte cap. The third insert must evict the oldest (LRU); the second + third must survive."""
    monkeypatch.setenv("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", str(25 * 1024 * 1024))
    cache = PipelineCache(verbose=False)
    train_a = _SizedBlob(10 * 1024 * 1024)
    train_b = _SizedBlob(10 * 1024 * 1024)
    train_c = _SizedBlob(10 * 1024 * 1024)

    cache.set("key_a", train_a, None, None)
    cache.set("key_b", train_b, None, None)
    cache.set("key_c", train_c, None, None)

    assert cache.get("key_a") is None, "oldest entry must have been evicted under the 25MB cap"
    assert cache.get("key_b") is not None, "second entry must survive"
    assert cache.get("key_c") is not None, "newest entry must survive"


def test_pipeline_cache_get_promotes_to_mru(monkeypatch) -> None:
    """Accessing ``key_a`` must move it to most-recently-used; subsequent overflow then evicts ``key_b`` (now LRU), not ``key_a``."""
    monkeypatch.setenv("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", str(25 * 1024 * 1024))
    cache = PipelineCache(verbose=False)
    blob_a = _SizedBlob(10 * 1024 * 1024)
    blob_b = _SizedBlob(10 * 1024 * 1024)
    blob_c = _SizedBlob(10 * 1024 * 1024)

    cache.set("key_a", blob_a, None, None)
    cache.set("key_b", blob_b, None, None)
    # Touch key_a -> now MRU; key_b becomes LRU.
    _ = cache.get("key_a")
    cache.set("key_c", blob_c, None, None)

    assert cache.get("key_b") is None, "key_b should have been evicted after key_a was promoted to MRU"
    assert cache.get("key_a") is not None, "key_a was MRU and must survive"
    assert cache.get("key_c") is not None, "key_c just inserted, must survive"


def test_pipeline_cache_pandas_frame_size_accounted(monkeypatch) -> None:
    """pandas frames must contribute their ``memory_usage(deep=False).sum()`` to the cache budget; the size estimator must not fall back to ``sys.getsizeof`` (which returns ~100B regardless of buffer size). Two ~8MB frames under a 12MB cap means the second insert evicts the first."""
    monkeypatch.setenv("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", str(12 * 1024 * 1024))
    cache = PipelineCache(verbose=False)
    # 500k * 8 bytes * 2 cols = 8_000_000 bytes per frame (RangeIndex carries near-zero overhead).
    big_a = pd.DataFrame({"x": np.zeros(500_000, dtype=np.float64), "y": np.zeros(500_000, dtype=np.float64)})
    big_b = pd.DataFrame({"x": np.zeros(500_000, dtype=np.float64), "y": np.zeros(500_000, dtype=np.float64)})

    cache.set("big_a", big_a, None, None)
    # Sanity check: the estimator must report the actual buffer size, not ``sys.getsizeof``'s ~100 bytes container overhead.
    assert cache.cache_size_bytes() >= 7_900_000, (
        f"pandas size estimator returned only {cache.cache_size_bytes()} bytes; "
        "expected ~8MB for the 500kx2 float64 frame. The estimator likely fell back to sys.getsizeof."
    )
    cache.set("big_b", big_b, None, None)
    # 8MB + 8MB = 16MB > 12MB cap; big_a (LRU) must evict so big_b fits.
    assert cache.get("big_a") is None, "big_a should have been evicted (LRU under 12MB cap)"
    assert cache.get("big_b") is not None, "big_b just inserted; must survive"


def test_pipeline_cache_default_byte_limit_is_dynamic() -> None:
    """2026-05-25: hardcoded 2 GB default replaced with psutil-driven
    (available RAM - 8 GB reserve), clamped to [2 GB, 64 GB]. The env
    var override knob still wins. On any dev/CI box with > 10 GB free
    we expect at least 2 GB and at most 64 GB."""
    os.environ.pop("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", None)
    cache = PipelineCache(verbose=False)
    assert cache._bytes_limit >= 2 * 1024 * 1024 * 1024
    assert cache._bytes_limit <= 64 * 1024 * 1024 * 1024


def test_pipeline_cache_eviction_logs_at_info(monkeypatch, caplog) -> None:
    """On overflow, the cache must emit an INFO line with the eviction stats (entries evicted + bytes freed) so operators can triage cache thrash."""
    import logging

    monkeypatch.setenv("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT", str(15 * 1024 * 1024))
    cache = PipelineCache(verbose=False)
    cache.set("a", _SizedBlob(10 * 1024 * 1024), None, None)
    with caplog.at_level(logging.INFO, logger="mlframe.training.strategies"):
        cache.set("b", _SizedBlob(10 * 1024 * 1024), None, None)
    # The eviction must be visible in the log so operators can see when the cap kicks in.
    msgs = [rec.message for rec in caplog.records]
    assert any("evicted" in m.lower() for m in msgs), f"expected an eviction INFO log line; got: {msgs}"
