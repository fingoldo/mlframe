"""
``CacheBackend`` Protocol + ``LocalDiskBackend`` implementation.

Why an abstraction layer for one local backend: round-3 future-proofing
F5 flagged that ``cache_dir: str`` and direct ``np.load``/``.npz``
hard-codes "local disk forever". Cloud-native (S3, GCS, Azure Blob),
shared-FS, distributed (Ray, Dask), federated -- all need a different
backend. Retrofitting after rollout is a many-hundred-line change.
The Protocol here is the seam; ``LocalDiskBackend`` is the only impl
in v1, which is enough for solo greenfield.

Atomic writes route through :func:`mlframe.training.io.atomic_write_bytes`
(existing helper, lines 21-66) -- no new ``_atomic_write`` was needed.
That helper already does the tempfile → ``os.replace`` dance and
cleans up on exception (round-3 chaos C1 verified satisfied).

Per-key locking goes through :class:`mlframe.training.feature_handling.locking.PIDAwareFileLock`.
"""

from __future__ import annotations

import contextlib
import logging
import os
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

    def __init__(self, root: str):
        self.root = long_path_safe(os.path.abspath(root))
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        self._locks_dir = os.path.join(self.root, ".locks")
        os.makedirs(self._locks_dir, mode=0o700, exist_ok=True)

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

    def exists(self, key: str) -> bool:
        return os.path.exists(self._value_path(key))

    def delete(self, key: str) -> None:
        path = self._value_path(key)
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

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


@contextlib.contextmanager
def _noop_lock():  # pragma: no cover -- helper for backends that don't lock
    """For backends without locking (in-memory, shared-Ray queues)."""
    yield
