"""Cluster 29 regression: the eviction protect set names only real sidecars (audit 2026-11-20, finding 29-F40).

An eviction pass triggered by a ``put`` must spare the entry that put just wrote, its payload and its ``<key>.pkl.sha256``
sidecar together. If the sidecar were evicted, the next ``get`` would fail closed, delete both files and drop the value.
"""

import os
import time

import pytest

from mlframe.utils import disk_cache
from mlframe.utils.disk_cache import DiskCache

_PAYLOAD_BYTES = 1000
_CAP_BYTES = 1500
_HOUR = 3600


@pytest.fixture
def sidecar_oldest(monkeypatch):
    """Age each freshly written sidecar past every other file, so an unprotected eviction pass would take it first."""
    real_write_sidecar = disk_cache.write_sidecar

    def _write_and_age(path: str) -> None:
        """Write the sidecar as usual, then backdate it."""
        real_write_sidecar(path)
        sidecar = path + ".sha256"
        long_ago = time.time() - 2 * _HOUR
        os.utime(sidecar, (long_ago, long_ago))

    monkeypatch.setattr(disk_cache, "write_sidecar", _write_and_age)


def _seed_older_entry(cache: DiskCache, tmp_path) -> None:
    """Write an older entry whose payload is backdated, so it is the one eviction should drop."""
    cache.put("old", b"o" * _PAYLOAD_BYTES)
    old_payload = next(p for p in tmp_path.iterdir() if p.name.startswith("old") and p.suffix == ".pkl")
    past = time.time() - _HOUR
    os.utime(old_payload, (past, past))


def test_put_that_triggers_eviction_keeps_the_new_entry_readable(tmp_path, sidecar_oldest):
    """The entry the put just wrote still round-trips through get after the eviction pass it triggered."""
    cache = DiskCache(str(tmp_path), max_size_bytes=_CAP_BYTES)
    _seed_older_entry(cache, tmp_path)

    cache.put("new", b"n" * _PAYLOAD_BYTES)

    assert cache.get("new") == b"n" * _PAYLOAD_BYTES


def test_real_sidecar_of_the_new_entry_is_spared(tmp_path, sidecar_oldest):
    """The ``<key>.pkl.sha256`` sidecar of the entry being written survives the eviction pass that follows its write."""
    cache = DiskCache(str(tmp_path), max_size_bytes=_CAP_BYTES)
    _seed_older_entry(cache, tmp_path)

    cache.put("new", b"n" * _PAYLOAD_BYTES)

    new_sidecars = [p for p in tmp_path.iterdir() if p.name.startswith("new") and p.name.endswith(".sha256")]
    assert len(new_sidecars) == 1
