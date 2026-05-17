"""
``CacheBackend`` Protocol + ``LocalDiskBackend`` implementation.

Why an abstraction layer for one local backend: ``cache_dir: str`` and
direct ``np.load``/``.npz`` would hard-code "local disk forever".
Cloud-native (S3, GCS, Azure Blob), shared-FS, distributed (Ray, Dask),
federated -- all need a different backend. Retrofitting after rollout
is a many-hundred-line change. The Protocol here is the seam;
``LocalDiskBackend`` is the only impl in v1.

Atomic writes route through :func:`mlframe.training.io.atomic_write_bytes`,
which does the tempfile then ``os.replace`` dance and cleans up on
exception.

Per-key locking goes through :class:`mlframe.training.feature_handling.locking.PIDAwareFileLock`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import ContextManager, List, Protocol, runtime_checkable

from mlframe.training.feature_handling.locking import PIDAwareFileLock
from mlframe.training.feature_handling.system import long_path_safe
from mlframe.training.io import atomic_write_bytes

logger = logging.getLogger(__name__)


@runtime_checkable
class CacheBackend(Protocol):
    """Pluggable backend for the feature-handling cache.

    Methods are intentionally bytes-oriented so the layer above (which
    handles serialisation, fp16 storage, memmap, etc.) is backend-agnostic.
    """

    def read(self, key: str) -> bytes:
        """Return the bytes stored under ``key``. Raise ``KeyError`` if absent."""

    def write(self, key: str, data: bytes) -> None:
        """Store ``data`` under ``key`` atomically."""

    def exists(self, key: str) -> bool:
        """Cheap presence check."""

    def delete(self, key: str) -> None:
        """Remove ``key``. No-op if absent."""

    def lock(self, key: str) -> ContextManager:
        """Acquire an exclusive cross-process lock for ``key``."""

    def list_keys(self, prefix: str = "") -> List[str]:
        """Enumerate keys under ``prefix``. Used by eviction."""


class LocalDiskBackend:
    """Local-filesystem backend. The only impl in phase M.

    Layout:
        ``{root}/{key}.bin`` -- the value
        ``{root}/.locks/{key}.lock`` -- per-key cross-process lock

    ``key`` is treated as an opaque string (the cache layer hashes
    column names + provider signatures into safe filenames before
    handing them in). Path traversal defence sits at the cache layer,
    not here -- this class is a primitive.
    """

    def __init__(
        self,
        root: str,
        *,
        max_entries: "int | None" = None,
        max_size_mb: "float | None" = None,
    ):
        """Construct the local-disk backend.

        ``max_entries`` / ``max_size_mb`` are optional LRU caps. When set,
        a successful ``write()`` evicts the least-recently-accessed
        ``.bin`` entries (tracked via the sidecar ``<root>/.lru``) to fit
        the cap. ``None`` (default) preserves the pre-cap behaviour - no
        eviction, suitable for the v1 solo-greenfield workload.
        """
        self.root = long_path_safe(os.path.abspath(root))
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        self._locks_dir = os.path.join(self.root, ".locks")
        os.makedirs(self._locks_dir, mode=0o700, exist_ok=True)
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        self._lru_path = os.path.join(self.root, ".lru")
        # In-process serialisation for the ``.lru`` sidecar: every
        # mutation reads the JSON, modifies it, and atomic-writes back.
        # Without a lock, two threads racing on ``_touch_lru`` /
        # ``_evict_to_caps`` can load the same snapshot, mutate
        # independently, then sequentially clobber each other's update.
        # Cross-process safety is layered via ``PIDAwareFileLock`` on
        # the sidecar path so concurrent writers from sibling processes
        # observe the same critical section.
        self._lru_mem_lock = threading.Lock()
        self._lru_cross_proc_lock_path = os.path.join(
            self._locks_dir, ".lru.lock"
        )

    # ---- path helpers ------------------------------------------------

    def _value_path(self, key: str) -> str:
        return os.path.join(self.root, f"{key}.bin")

    def _lock_path(self, key: str) -> str:
        return os.path.join(self._locks_dir, f"{key}.lock")

    # ---- Protocol methods -------------------------------------------

    def read(self, key: str) -> bytes:
        path = self._value_path(key)
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError as e:
            raise KeyError(key) from e

    def write(self, key: str, data: bytes) -> None:
        path = self._value_path(key)

        def _writer(fileobj) -> None:
            fileobj.write(data)

        atomic_write_bytes(path, _writer)
        # LRU tracking + cap enforcement only kick in when caps are set;
        # the unbounded path bypasses sidecar I/O entirely so the
        # zero-config workload pays nothing.
        if self.max_entries is not None or self.max_size_mb is not None:
            try:
                with self._lru_locked():
                    self._touch_lru(key)
                    self._evict_to_caps()
            except Exception:
                pass

    def exists(self, key: str) -> bool:
        return os.path.exists(self._value_path(key))

    def delete(self, key: str) -> None:
        path = self._value_path(key)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        if self.max_entries is not None or self.max_size_mb is not None:
            try:
                with self._lru_locked():
                    lru = self._load_lru()
                    if lru.pop(key, None) is not None:
                        self._save_lru(lru)
            except Exception:
                pass

    # ---- LRU sidecar -------------------------------------------------
    # Mirrors DiscoveryCache's sidecar: NTFS / noatime mounts make atime
    # unreliable, so a separate ledger gives portable access tracking.

    @contextlib.contextmanager
    def _lru_locked(self):
        """Serialise sidecar read-modify-write across threads and
        processes. The in-process ``threading.Lock`` is the fast path;
        the ``PIDAwareFileLock`` covers sibling processes. Falls back
        to the in-process lock alone if the cross-process layer cannot
        be acquired (filelock missing on minimal installs).
        """
        with self._lru_mem_lock:
            file_lock = PIDAwareFileLock(self._lru_cross_proc_lock_path, timeout=30.0)
            try:
                file_lock.__enter__()
            except Exception:
                # Cross-process layer unavailable; in-process lock
                # still protects this interpreter's threads.
                yield
                return
            try:
                yield
            finally:
                file_lock.__exit__(None, None, None)

    def _load_lru(self) -> "dict[str, float]":
        import json
        if not os.path.exists(self._lru_path):
            return {}
        try:
            with open(self._lru_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if isinstance(d, dict):
                return {str(k): float(v) for k, v in d.items()}
        except (OSError, ValueError):
            pass
        return {}

    def _save_lru(self, lru: "dict[str, float]") -> None:
        import json
        atomic_write_bytes(
            self._lru_path,
            lambda f: f.write(json.dumps(lru, sort_keys=True).encode("utf-8")),
        )

    def _touch_lru(self, key: str) -> None:
        import time
        lru = self._load_lru()
        lru[key] = time.time()
        self._save_lru(lru)

    def _evict_to_caps(self) -> int:
        """Evict least-recently-accessed entries until both caps fit.
        Returns the number of entries removed.

        Fast-path: when ``max_size_mb`` is unset and the sidecar entry count is already at-or-below
        ``max_entries``, skip the listdir + per-entry stat entirely. The pre-fix unconditional
        listdir paid O(N) on every write even when no eviction was due, which dominated bulk insert
        latency once N grew into the thousands.
        """
        if self.max_entries is None and self.max_size_mb is None:
            return 0
        lru = self._load_lru()
        if (
            self.max_size_mb is None
            and self.max_entries is not None
            and len(lru) <= self.max_entries
        ):
            return 0
        # Collect every on-disk entry; pre-cap legacy keys default to
        # timestamp 0 so they evict first.
        entries: List[tuple] = []
        try:
            names = os.listdir(self.root)
        except OSError:
            return 0
        for name in names:
            if name.startswith(".") or not name.endswith(".bin"):
                continue
            key = name[:-4]
            try:
                size = os.path.getsize(os.path.join(self.root, name))
            except OSError:
                size = 0
            entries.append((key, float(lru.get(key, 0.0)), size))
        entries.sort(key=lambda e: e[1])

        n = len(entries)
        total = sum(s for _, _, s in entries)
        max_bytes = (
            int(self.max_size_mb * 1024 * 1024)
            if self.max_size_mb is not None else None
        )

        removed = 0
        for key, _ts, size in entries:
            over_count = self.max_entries is not None and n > self.max_entries
            over_size = max_bytes is not None and total > max_bytes
            if not over_count and not over_size:
                break
            try:
                os.unlink(self._value_path(key))
                removed += 1
                n -= 1
                total -= size
                lru.pop(key, None)
            except OSError:
                pass
        if removed:
            self._save_lru(lru)
        return removed

    def lock(self, key: str) -> ContextManager:
        return PIDAwareFileLock(self._lock_path(key))

    def list_keys(self, prefix: str = "") -> List[str]:
        out: List[str] = []
        try:
            for name in os.listdir(self.root):
                if name.startswith(".") or not name.endswith(".bin"):
                    continue
                key = name[:-4]  # strip ".bin"
                if not prefix or key.startswith(prefix):
                    out.append(key)
        except FileNotFoundError:
            return []
        return out


# Sanity: the impl actually satisfies the Protocol structurally.
# We don't `assert isinstance(LocalDiskBackend(...), CacheBackend)` at
# import time because that would require constructing the backend with
# a real path. The runtime_checkable Protocol is sufficient for tests
# to pin the contract.


