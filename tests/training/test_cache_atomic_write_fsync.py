"""Regression net: atomic-write helpers must ``fsync`` BEFORE ``os.replace``.

``pickle.dump`` / ``dill.dump`` / numpy save routines flush only their own
buffers, not the OS page cache. A power loss between ``os.replace`` (which
makes the name visible) and the kernel's eventual writeback can publish a
zero-byte file under the cache key — silently corrupting the cache without
crashing.

These tests assert that both atomic-write paths (``DiscoveryCache.set`` and
``atomic_write_bytes``) call ``os.fsync`` BEFORE ``os.replace``.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from unittest import mock

import pytest


def test_discovery_cache_set_fsyncs_before_replace(tmp_path):
    """``DiscoveryCache.set`` must fsync the tmp fd before the atomic rename."""
    from mlframe.training.composite_cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path))
    call_order: list[str] = []

    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd):
        call_order.append("fsync")
        return real_fsync(fd)

    def tracking_replace(src, dst):
        call_order.append("replace")
        return real_replace(src, dst)

    with mock.patch("os.fsync", side_effect=tracking_fsync) as m_fsync, \
         mock.patch("os.replace", side_effect=tracking_replace):
        cache.set("deadbeef", {"hello": "world"})

    assert m_fsync.called, "os.fsync was never called - cache.set is not durable"
    # The fsync must precede the replace; otherwise rename can publish a
    # name whose contents are still dirty pages.
    assert call_order.index("fsync") < call_order.index("replace"), (
        f"fsync must precede replace, got order: {call_order}"
    )

    # Sanity: the value round-trips so we did not break the happy path.
    assert cache.get("deadbeef") == {"hello": "world"}


def test_atomic_write_bytes_fsyncs_before_replace(tmp_path):
    """``atomic_write_bytes`` must fsync before the atomic rename WHEN
    ``fsync=True`` is passed.

    Note: the default was flipped from True to False on 2026-05-20
    (atomicity guarantee is independent of fsync; durability is the only
    thing fsync adds and the worst case is "re-train the model"). Tests
    that pin the fsync-before-replace order must therefore opt-in.
    """
    from mlframe.training.io import atomic_write_bytes

    target = str(tmp_path / "payload.bin")
    call_order: list[str] = []

    real_fsync = os.fsync
    real_replace = os.replace

    def tracking_fsync(fd):
        call_order.append("fsync")
        return real_fsync(fd)

    def tracking_replace(src, dst):
        call_order.append("replace")
        return real_replace(src, dst)

    def writer(f):
        f.write(b"durable bytes")

    with mock.patch("os.fsync", side_effect=tracking_fsync) as m_fsync, \
         mock.patch("os.replace", side_effect=tracking_replace):
        atomic_write_bytes(target, writer, fsync=True)

    assert m_fsync.called, "os.fsync was never called inside atomic_write_bytes"
    assert call_order.index("fsync") < call_order.index("replace"), (
        f"fsync must precede replace, got order: {call_order}"
    )

    with open(target, "rb") as f:
        assert f.read() == b"durable bytes"
