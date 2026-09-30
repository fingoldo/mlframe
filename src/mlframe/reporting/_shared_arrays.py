"""Shared-memory hand-off of ndarrays between processes: a pool on the owner side, read-only zero-copy views on the worker side.

``SharedArrayPool.share`` copies an array into a ``multiprocessing.shared_memory`` segment once per key and returns a picklable
``SharedArrayDescriptor`` (name, shape, dtype); ``attach_array`` maps that descriptor in another process as a read-only ndarray view
over the same pages, ``detach_array`` drops the mapping. Segment names carry the owner's pid (``mlframe_rq_<pid>_...``) so a leak is
attributable, and the pool unlinks everything on ``close``.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Dict, NamedTuple, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SHM_PREFIX = "mlframe_rq_"


class SharedArrayDescriptor(NamedTuple):
    """Picklable handle to an ndarray living in a shared-memory segment: all a worker needs to map it."""

    name: str
    shape: Tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass
class _Entry:
    """One live segment on the owner side, with the number of in-flight tasks still referencing it."""

    shm: shared_memory.SharedMemory
    desc: SharedArrayDescriptor
    refs: int = 1
    src: Any = None


class SharedArrayPool:
    """Owner-side registry of shared-memory segments holding read-only copies of ndarrays.

    ``share`` copies an array into a segment exactly once per key; repeated calls with the same key (or, by default, the
    same buffer address + shape + strides + dtype) return the existing segment and bump its reference count, so one
    (model, split) prediction vector submitted to a dozen tasks is stored once. ``release`` drops a reference and unlinks
    the segment at zero. ``close`` unlinks everything that is left.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: Dict[Any, _Entry] = {}
        self._by_name: Dict[str, Any] = {}
        self._token = f"{_SHM_PREFIX}{os.getpid()}_{uuid.uuid4().hex[:8]}_"
        self._seq = 0
        self._closed = False

    def __getstate__(self) -> Dict[str, Any]:
        """The pool owns shared segments and a lock; it is process-local and must never be pickled (workers receive descriptors instead)."""
        raise TypeError("SharedArrayPool owns live shared-memory segments and cannot be pickled; pass SharedArrayDescriptor values instead")

    @staticmethod
    def _auto_key(arr: np.ndarray) -> Tuple[Any, ...]:
        """Identity of a buffer view: address + shape + strides + dtype (the source is pinned while the entry lives)."""
        return ("buf", arr.__array_interface__["data"][0], arr.shape, arr.strides, arr.dtype.str)

    def share(self, arr: np.ndarray, key: Any = None) -> SharedArrayDescriptor:
        """Copy ``arr`` into shared memory once per ``key`` and return its descriptor. Raises ``TypeError`` for object dtype."""
        a = np.asarray(arr)
        if a.dtype.hasobject:
            raise TypeError("object-dtype arrays cannot be placed in shared memory")
        ckey = ("key", key) if key is not None else self._auto_key(a)
        with self._lock:
            if self._closed:
                raise RuntimeError("SharedArrayPool is closed")
            hit = self._entries.get(ckey)
            if hit is not None and hit.desc.shape == a.shape and hit.desc.dtype == a.dtype.str:
                hit.refs += 1
                return hit.desc
            self._seq += 1
            name = f"{self._token}{self._seq}"
            shm = shared_memory.SharedMemory(create=True, size=max(int(a.nbytes), 1), name=name)
            try:
                np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)[...] = a
            except BaseException:
                shm.close()
                shm.unlink()
                raise
            desc = SharedArrayDescriptor(shm.name, tuple(a.shape), a.dtype.str, int(a.nbytes))
            # An auto-keyed entry pins its source so the address cannot be recycled for different content while it is live.
            self._entries[ckey] = _Entry(shm=shm, desc=desc, src=a if key is None else None)
            self._by_name[desc.name] = ckey
            return desc

    def owner_view(self, desc: SharedArrayDescriptor) -> np.ndarray:
        """Writable owner-side view of a live segment (tests and owners that refresh a shared buffer)."""
        with self._lock:
            entry = self._entries[self._by_name[desc.name]]
        return np.ndarray(desc.shape, dtype=np.dtype(desc.dtype), buffer=entry.shm.buf)

    def release(self, desc: SharedArrayDescriptor) -> None:
        """Drop one reference; unlink the segment when the last one goes."""
        with self._lock:
            ckey = self._by_name.get(desc.name)
            entry = self._entries.get(ckey) if ckey is not None else None
            if entry is None:
                return
            entry.refs -= 1
            if entry.refs > 0:
                return
            del self._entries[ckey]
            del self._by_name[desc.name]
        self._destroy(entry)

    @staticmethod
    def _destroy(entry: _Entry) -> None:
        """Close and unlink one segment, tolerating a segment that is already gone."""
        try:
            entry.shm.close()
        except (BufferError, OSError):
            logger.debug("shared segment %s still has exported views at close", entry.desc.name)
        try:
            entry.shm.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            logger.debug("shared segment %s could not be unlinked", entry.desc.name, exc_info=True)

    @property
    def live_segments(self) -> int:
        """Segments currently allocated."""
        with self._lock:
            return len(self._entries)

    @property
    def live_bytes(self) -> int:
        """Bytes currently allocated across segments."""
        with self._lock:
            return sum(e.desc.nbytes for e in self._entries.values())

    def close(self) -> None:
        """Unlink every remaining segment; safe to call repeatedly."""
        with self._lock:
            self._closed = True
            entries = list(self._entries.values())
            self._entries.clear()
            self._by_name.clear()
        for entry in entries:
            self._destroy(entry)


# Worker-side attachments, keyed by segment name, so a segment used by several tasks is mapped once per process.
_ATTACHED: Dict[str, Tuple[shared_memory.SharedMemory, int]] = {}
_ATTACH_LOCK = threading.Lock()


def attach_array(desc: SharedArrayDescriptor) -> np.ndarray:
    """Map ``desc`` as a read-only zero-copy ndarray view. Pair every call with ``detach_array``."""
    with _ATTACH_LOCK:
        hit = _ATTACHED.get(desc.name)
        if hit is None:
            # Workers are spawned children of the owner, so they share the owner's resource tracker: the attach registers the
            # same name (a set, no duplicate) and the owner's unlink unregisters it. Unregistering here would make that
            # unlink raise KeyError inside the tracker at exit.
            shm = shared_memory.SharedMemory(name=desc.name)
            _ATTACHED[desc.name] = (shm, 1)
        else:
            _ATTACHED[desc.name] = (hit[0], hit[1] + 1)
        shm = _ATTACHED[desc.name][0]
    view = np.ndarray(desc.shape, dtype=np.dtype(desc.dtype), buffer=shm.buf)
    view.flags.writeable = False
    return view


def detach_array(desc: SharedArrayDescriptor) -> None:
    """Drop one worker-side reference; the mapping closes when no task uses the segment any more."""
    with _ATTACH_LOCK:
        hit = _ATTACHED.get(desc.name)
        if hit is None:
            return
        if hit[1] > 1:
            _ATTACHED[desc.name] = (hit[0], hit[1] - 1)
            return
        del _ATTACHED[desc.name]
    try:
        hit[0].close()
    except (BufferError, OSError):
        logger.debug("worker could not close segment %s (views still exported)", desc.name)


def readonly_view(arr: np.ndarray) -> np.ndarray:
    """A read-only view of ``arr`` sharing its buffer (no copy); the caller's array keeps its own writability."""
    v: np.ndarray = np.asarray(arr).view()
    v.flags.writeable = False
    return v
