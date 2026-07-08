"""Disk-backed ``DiscoveryCache`` store.

Holds the key->pickle disk cache class + its byte-total helper. The signature / cache-key
composition primitives stay in the sibling ``cache.py``, which re-exports ``DiscoveryCache``
and ``_discovery_cache_bytes_total`` from here so ``from ...composite.cache import DiscoveryCache``
keeps resolving. Pure stdlib + ``mlframe.utils.safe_pickle``; no composite-internal deps."""

from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import logging
import os
import pickle  # nosec B403 - module used safely in this file, see call sites below (no untrusted input reaches it)
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from mlframe.utils.safe_pickle import (
    safe_load as _safe_pickle_load,
    write_sidecar as _safe_pickle_write_sidecar,
)

logger = logging.getLogger(__name__)

# Pre-compiled hex matcher for ``_safe_key`` (compile once, reuse). The regex anchors with
# ``\A`` / ``\Z`` to require the WHOLE key to be hex (otherwise ``re.match`` only anchors to the
# start and ``abc-def`` would slip through).
_HEX_KEY_RE = re.compile(r"\A[0-9a-fA-F]+\Z")

# One-shot guard so the "filelock missing -> LRU/eviction races" warning fires at most once per process instead of on every cache touch.
_FILELOCK_WARNED = False


class DiscoveryCache:
    """Disk-backed key->value cache for CompositeTargetDiscovery results.

    Values are pickled with stdlib ``pickle`` (safe: stored objects are dataclass-derived dicts). Files live under ``<cache_dir>/<key>.pkl`` with one file per key for easy invalidation / cleanup.

    Concurrency: value writes are crash-safe via atomic ``os.replace`` (tmp-file + fsync + rename), the LRU sidecar and eviction sweep are guarded by a cross-process ``filelock`` (when ``filelock`` is installed), and ``invalidate`` is idempotent under concurrent callers. ``filelock`` is optional: without it the LRU/eviction read-modify-write can race between processes sharing ``cache_dir`` (a stale-snapshot save may overwrite a fresh access timestamp), though the value files themselves stay consistent.
    """

    def __init__(
        self,
        cache_dir: Any,
        *,
        max_entries: Optional[int] = 1000,
        max_size_mb: Optional[float] = 2000.0,
    ) -> None:
        """Construct a disk-backed discovery cache.

        Parameters
        ----------
        cache_dir
            Directory hosting one ``<key>.pkl`` per entry.
        max_entries
            Hard cap on the number of cached entries. When ``set()`` would
            push the count above the cap, the least-recently-accessed
            entries are evicted to fit. Default 1000 - protects against
            unbounded R&D growth. Pass ``None`` to disable count-based
            eviction explicitly.
        max_size_mb
            Soft cap on the total cache footprint in megabytes. Evaluated
            after the count cap. Default 2000 MB. Pass ``None`` to disable.

        LRU tracking uses a sidecar ``<cache_dir>/.lru`` JSON file rather
        than ``os.path.getatime``: Windows / NTFS frequently mounts with
        noatime semantics so atime is unreliable; the sidecar gives us a
        portable monotonic-time access ledger that survives process exit.

        The cache directory is wrapped through
        :func:`mlframe.training.feature_handling.system.long_path_safe`
        on Windows so deep cache trees (>= 260 chars) survive
        ``os.replace`` in ``set()``. ``LocalDiskBackend`` already did
        this; ``DiscoveryCache`` did not, so a deep run-name + nested
        artifact path crashed on Windows even though the same directory
        worked under ``LocalDiskBackend``.
        """
        from ..feature_handling.system import long_path_safe
        self.cache_dir = long_path_safe(os.path.abspath(str(cache_dir)))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.max_entries = max_entries
        self.max_size_mb = max_size_mb
        self._lru_path = os.path.join(self.cache_dir, ".lru")
        # Both caps None means the cache grows monotonically on repeated R&D runs and silently
        # fills the disk. CI / test runs commonly suppress warnings, so a WARN-only signal
        # disappears in practice -- promote to a hard ValueError so the operator is forced to make
        # an explicit choice. Pass ``max_entries=10**9`` or ``max_size_mb=float("inf")`` if the
        # genuine intent is "no eviction".
        if max_entries is None and max_size_mb is None:
            raise ValueError(
                f"DiscoveryCache at {self.cache_dir!r} constructed with max_entries=None and max_size_mb=None: "
                "the cache would grow without bound. Pass at least one explicit cap (or float('inf') / 10**9 "
                "to opt into unbounded growth) so the choice is auditable."
            )
        # In-memory LRU ledger + incremental size accumulator (CPX28). The pre-fix store rewrote the
        # whole ``.lru`` JSON on every touch and globbed + getsize'd every ``*.pkl`` on each ``set``,
        # giving O(N^2) cost over a run. We now hold the LRU dict in memory, flush it lazily (on
        # eviction -- which already happens -- and on ``close`` / ``__del__``), and keep a running
        # byte total + per-stem size map updated incrementally on set / invalidate / evict. All three
        # are rebuilt from disk the first time they are needed (``_ensure_*``), so a process restart
        # reconstructs them correctly and eviction decisions stay identical to the disk-scan design.
        self._lru: Optional[Dict[str, float]] = None
        self._lru_dirty = False
        self._entry_sizes: Optional[Dict[str, int]] = None
        self._total_bytes = 0

    # LRU sidecar (key -> access timestamp). Plain JSON; tiny so we read / write the whole file on
    # every touch. Atime is too unreliable on NTFS to depend on.
    #
    # File-lock the sidecar so two concurrent processes hitting the same data_dir can't interleave
    # an evict + write and leave live entries marked stale. ``filelock`` is optional -- absence
    # falls back to the racy behaviour with a one-time WARN.

    def _lock_path(self) -> str:
        """Path of the cross-process filelock guarding the ``.lru`` sidecar and eviction sweep."""
        return self._lru_path + ".lock"

    @staticmethod
    def _maybe_filelock(lock_path: str):
        """Return a ``filelock.FileLock`` instance if the dep is present, else a no-op context."""
        try:
            from filelock import FileLock as _FileLock
            return _FileLock(lock_path, timeout=30)
        except ImportError:  # pragma: no cover
            global _FILELOCK_WARNED
            if not _FILELOCK_WARNED:
                _FILELOCK_WARNED = True
                logger.warning(
                    "DiscoveryCache: 'filelock' not installed; LRU/eviction read-modify-write is "
                    "unprotected across processes sharing the cache dir (value files stay consistent). "
                    "Install 'filelock' to close the race."
                )
            return contextlib.nullcontext()

    def _load_lru(self) -> Dict[str, float]:
        """Read the ``.lru`` sidecar JSON from disk into a ``{key: access_timestamp}`` dict; returns ``{}`` on a missing or corrupt file (fail-open, since LRU is a hint, not a correctness requirement)."""
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

    def _save_lru(self, lru: Dict[str, float]) -> None:
        """Persist ``lru`` to the ``.lru`` sidecar via tmp-file + fsync + atomic ``os.replace``, tracking fd ownership so a failure between ``mkstemp`` and ``os.fdopen`` doesn't leak the descriptor."""
        # Same atomic-rename + fsync discipline as the value writes - LRU
        # corruption would silently break eviction order.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, prefix=".lru.", suffix=".tmp")
        # ``fd`` ownership tracking: BufferedWriter from os.fdopen adopts on success;
        # if os.fdopen itself raises (rare: MemoryError) we manually close fd to avoid
        # a per-failure fd leak. _save_lru runs on every cache touch so leaks compound
        # quickly under sustained load (Windows 8192 default fd ceiling).
        _fd_adopted = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                _fd_adopted = True
                json.dump(lru, f, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._lru_path)
        except Exception:
            if not _fd_adopted:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _ensure_lru(self) -> Dict[str, float]:
        """Lazily load the LRU ledger from disk into memory exactly once.

        After load the in-memory dict is authoritative for this process; touches mutate it and set
        the dirty flag, and the disk file is only rewritten at flush points (eviction + close).
        Rebuilding from disk here keeps a process restart correct -- the prior run's flushed ledger
        is read back, so access ordering survives.
        """
        if self._lru is None:
            self._lru = self._load_lru()
        return self._lru

    def _ensure_sizes(self) -> Dict[str, int]:
        """Lazily build the per-stem byte map (pkl + .sha256 sidecar) and running total from disk.

        Done once per process; thereafter ``set`` / ``invalidate`` / eviction keep it in sync
        incrementally instead of globbing + getsize-ing every ``*.pkl`` on each ``set``. The total
        accumulator (``self._total_bytes``) and this map are reconstructed together so a restart
        re-derives the exact footprint the disk-scan design would have computed.
        """
        if self._entry_sizes is None:
            sizes: Dict[str, int] = {}
            total = 0
            for path in glob.glob(os.path.join(self.cache_dir, "*.pkl")):
                stem = os.path.splitext(os.path.basename(path))[0]
                size = 0
                try:
                    size = os.path.getsize(path)
                except OSError:
                    pass
                try:
                    size += os.path.getsize(path + ".sha256")
                except OSError:
                    pass
                sizes[stem] = size
                total += size
            self._entry_sizes = sizes
            self._total_bytes = total
        return self._entry_sizes

    def _entry_size_on_disk(self, stem: str) -> int:
        """Sum the byte size of ``<stem>.pkl`` plus its ``.sha256`` sidecar (0 for either that doesn't exist)."""
        path = os.path.join(self.cache_dir, f"{stem}.pkl")
        size = 0
        try:
            size = os.path.getsize(path)
        except OSError:
            return 0
        try:
            size += os.path.getsize(path + ".sha256")
        except OSError:
            pass
        return size

    def _flush_lru(self) -> None:
        """Write the in-memory LRU ledger to disk if dirty. Called on eviction and on close/del."""
        if self._lru is not None and self._lru_dirty:
            with self._maybe_filelock(self._lock_path()):
                self._save_lru(self._lru)
            self._lru_dirty = False

    def close(self) -> None:
        """Flush any pending in-memory LRU state to disk so it survives process exit.

        Idempotent. Eviction already flushes on every sweep, so close() only matters when the run
        did sets/gets that touched the ledger without triggering an eviction (e.g. caps not yet hit).
        """
        try:
            self._flush_lru()
        except Exception as _e:
            logger.debug("DiscoveryCache.close LRU flush failed: %s", _e)

    def __del__(self) -> None:
        # Best-effort flush so a cache GC'd without an explicit close() still persists access order.
        try:
            self.close()
        except Exception as e:
            logger.debug("swallowed exception in cache_store.py: %s", e)
            pass

    def _touch_lru(self, key: str) -> None:
        """Record ``key`` as accessed now in the in-memory LRU ledger and mark it dirty (disk write deferred to the next flush point)."""
        # ``time.time()`` is wall-clock (subject to NTP step-back) but we deliberately do NOT use
        # ``time.monotonic()`` here -- the LRU sidecar is shared across processes, and monotonic
        # clock values are not comparable across processes (each process's monotonic clock starts
        # from an arbitrary reference). Cross-process LRU ordering requires a shared reference
        # frame -- wall clock is the only portable option. Single-host NTP step-backs are rare and
        # only mis-order entries within the step delta.
        #
        # CPX28: mutate the in-memory ledger and mark dirty; the disk file is rewritten lazily at
        # flush points (eviction + ``close`` / ``__del__``) rather than on every touch, turning the
        # per-op O(N) JSON rewrite into an amortised O(1) dict write. Eviction already flushes on
        # every sweep, so a crash between flushes loses at most the access-timestamp bumps since the
        # last sweep -- the value files themselves stay durable via ``set``'s atomic rename + fsync.
        lru = self._ensure_lru()
        lru[key] = time.time()
        self._lru_dirty = True

    def _path(self, key: str) -> str:
        """On-disk ``.pkl`` path for ``key`` after sanitisation via ``_safe_key``."""
        safe_key = self._safe_key(key)
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")

    def __contains__(self, key: str) -> bool:
        # ``os.path.exists`` is racy by design (TOCTOU: the file can be deleted between the check
        # and a subsequent ``get``). Callers should use
        # ``get(key, default=_SENTINEL)`` and check ``is _SENTINEL`` instead of the
        # ``key in cache`` + ``cache.get(key)`` pattern. ``get`` opens the file directly and
        # treats ``FileNotFoundError`` as a miss, so the race is closed there.
        return os.path.exists(self._path(key))

    def get(self, key: str, default: Any = None) -> Any:
        """Return the cached value, or ``default`` if the key is absent / unreadable.

        We deliberately omit any ``os.path.exists`` check before opening:
        on Windows a delete-between-exists-and-open race surfaced
        ``FileNotFoundError`` after the existence check passed. The
        implementation just try-opens and treats any failure (missing
        file, locked file, corrupt pickle) as a cache miss.

        A successful read updates the LRU sidecar so subsequent eviction
        picks the least-recently-USED (not just least-recently-WRITTEN)
        entry.
        """
        path = self._path(key)
        # Size-clamp: refuse to load a cache entry that exceeds the configured byte
        # ceiling (default 1 GiB; override via MLFRAME_DISCOVERY_CACHE_MAX_BYTES).
        # Discovery cache entries SHOULD be small (kB-MB scale: spec lists + scalar
        # metadata + per-base float32 arrays at screen-sample size). A multi-GB
        # entry indicates a bug upstream (e.g. an _auto_base_pool entry sized to
        # FULL train rows leaked into the pickled discovery instance) and loading
        # it would spike RAM at the worst possible time -- right before composite
        # discovery starts its own allocations. Treat oversize as a miss so the
        # caller falls back to a fresh recompute; the stale entry is left on disk
        # so the operator can inspect / delete it.
        try:
            _max_bytes_raw = os.environ.get("MLFRAME_DISCOVERY_CACHE_MAX_BYTES")
            _max_bytes = int(_max_bytes_raw) if _max_bytes_raw else 1024 * 1024 * 1024
        except (TypeError, ValueError):
            _max_bytes = 1024 * 1024 * 1024
        try:
            _file_size = os.path.getsize(path)
        except OSError:
            _file_size = -1
        if _file_size > _max_bytes:
            logger.warning(
                "DiscoveryCache: skipping oversized entry at %s (%.2f GiB > %.2f GiB ceiling); "
                "treating as cache miss. Set MLFRAME_DISCOVERY_CACHE_MAX_BYTES to raise the cap "
                "or delete the file to recompute. An oversized entry usually means an unintended "
                "full-train ndarray got pickled into the discovery instance upstream.",
                path,
                _file_size / 1024**3,
                _max_bytes / 1024**3,
            )
            return default
        # Route the load through safe_pickle so a corrupt-sidecar finding (digest mismatch from
        # tampering or partial write) raises PickleVerificationError instead of silently returning
        # stale data. allow_unverified=True keeps the migration story: legacy entries written before
        # the sidecar landed remain readable (cache miss is the safe fallback if they're broken).
        # Operators who want strict-only behaviour set MLFRAME_DISCOVERY_CACHE_STRICT=1.
        _strict = os.environ.get("MLFRAME_DISCOVERY_CACHE_STRICT", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        try:
            if _strict:
                value = _safe_pickle_load(path)
            else:
                # allow_unverified=True: missing sidecar is OK (WARN-logged once by verify_sidecar);
                # digest mismatch still raises so a tampered file does not slip into the cache hit
                # path. The broad except below converts the mismatch to a cache miss so callers see
                # consistent semantics.
                value = _safe_pickle_load(path, allow_unverified=True)
        except FileNotFoundError:
            return default
        except Exception as _e:
            # A persistent corrupt / unverifiable entry would otherwise return a silent miss every run, triggering unbounded multi-minute recomputes with no operator signal. Surface it once per read, then treat as a miss.
            logger.warning(
                "DiscoveryCache: unreadable/unverifiable entry %s (%s: %s); treating as miss",
                path, type(_e).__name__, _e,
            )
            return default
        # Successful read: bump LRU. Done outside the read try/except so
        # an LRU file failure doesn't break the read path.
        try:
            self._touch_lru(self._safe_key(key))
        except Exception as e:
            logger.debug("swallowed exception in cache_store.py: %s", e)
            pass
        return value

    def _safe_key(self, key: str) -> str:
        """Sanitised key (matches the on-disk filename stem).

        Collision-proof: pure-hex keys (the format ``make_discovery_cache_key`` emits)
        pass through unchanged. Any other key is hashed via blake2b and tagged with
        ``__h`` plus the BYTE length of the original; this prevents the old
        "strip non-alnum" sanitiser collapsing ``abc-def`` and ``abcdef`` (or
        ``abc/../def`` and ``abcdef``) to the same filename.
        """
        if not key:
            raise ValueError(f"DiscoveryCache: empty key {key!r}")
        # Pure-hex (the format make_discovery_cache_key emits) passes through unchanged.
        if _HEX_KEY_RE.match(key):
            return key.lower()
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        return f"{digest}__h{len(key.encode('utf-8'))}"

    def _entry_size_bytes(self, safe_key: str) -> int:
        """Byte size of the ``.pkl`` file for an already-sanitised ``safe_key`` (0 if missing); unlike ``_entry_size_on_disk`` this does NOT include the ``.sha256`` sidecar."""
        path = os.path.join(self.cache_dir, f"{safe_key}.pkl")
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def _evict_to_caps(self) -> int:
        """Evict least-recently-accessed entries to satisfy the configured
        ``max_entries`` / ``max_size_mb`` caps. Returns the number of
        entries removed. Called from ``set()`` after the new entry has
        been written so the new key participates in the LRU ordering.

        Entries missing from the LRU sidecar (legacy / external writes)
        are treated as least-recently-accessed (timestamp 0) so they
        evict first - keeping pre-existing-without-LRU entries pinned
        forever would defeat the cap.
        """
        if self.max_entries is None and self.max_size_mb is None:
            return 0
        # Same lock as _touch_lru: eviction reads + writes the sidecar AND removes files; another
        # process eviction sweep racing here could double-delete or leave the sidecar inconsistent.
        # Capture sys.exc_info() so the in-flight exception is forwarded to the lock manager's
        # __exit__ (preserves CM contract), AND wrap __exit__ itself in try/except so a filelock
        # cleanup error doesn't mask the eviction-body exception.
        import sys as _sys
        _lock_ctx = self._maybe_filelock(self._lock_path())
        _lock_ctx.__enter__()
        try:
            return self._evict_to_caps_locked()
        finally:
            _exc = _sys.exc_info()
            try:
                _lock_ctx.__exit__(*_exc)
            except Exception as _exit_err:
                logger.warning("DiscoveryCache eviction filelock __exit__ failed: %s", _exit_err)

    # Age (seconds) below which an orphan ``*.tmp`` is left alone -- a write in a sibling process
    # could be mid-rename. 1 h is far longer than any legitimate ``set`` / ``_save_lru`` write
    # window (sub-second) yet short enough that a leaked tmp from a crashed run is reclaimed
    # within the same R&D session. Override via MLFRAME_DISCOVERY_CACHE_TMP_AGE_S.
    _ORPHAN_TMP_MIN_AGE_S: float = 3600.0

    def _sweep_orphan_tmp_files(self) -> int:
        """Remove stale ``*.tmp`` files left by interrupted ``set`` / ``_save_lru`` writes.

        ``set`` and ``_save_lru`` write to ``tempfile.mkstemp(..., suffix=".tmp")`` then
        ``os.replace`` onto the visible name; a crash / kill between mkstemp and replace orphans
        the tmp. ``glob("*.pkl")`` never sees it, so the eviction byte cap under-counts the real
        footprint and the file accumulates forever (only a manual ``clear()`` reclaimed it, and
        pre-fix ``clear()`` did not sweep tmp either). We only delete tmps older than
        ``_ORPHAN_TMP_MIN_AGE_S`` so an in-flight write in a sibling process is never yanked.
        Returns the number of files removed.
        """
        try:
            _age_raw = os.environ.get("MLFRAME_DISCOVERY_CACHE_TMP_AGE_S")
            min_age = float(_age_raw) if _age_raw else self._ORPHAN_TMP_MIN_AGE_S
        except (TypeError, ValueError):
            min_age = self._ORPHAN_TMP_MIN_AGE_S
        now = time.time()
        removed = 0
        for path in glob.glob(os.path.join(self.cache_dir, "*.tmp")):
            try:
                age = now - os.path.getmtime(path)
            except OSError:
                continue
            if age < min_age:
                continue
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
        return removed

    def _evict_to_caps_locked(self) -> int:
        """Body of ``_evict_to_caps`` run while already holding the cross-process filelock: sweeps orphan tmp files, then removes entries oldest-access-first until both the entry-count and byte-size caps are satisfied. Returns the number of entries removed."""
        # Best-effort orphan sweep before sizing: a crashed/interrupted ``set`` or ``_save_lru``
        # can leave a ``*.tmp`` in cache_dir that ``glob("*.pkl")`` never sees, so eviction's
        # byte cap silently under-counts the true footprint. The filelock around eviction
        # makes this safe -- no concurrent ``set`` is mid-rename on a tmp we'd delete -- but we
        # still age-gate so a tmp from a write that is genuinely in flight in THIS process (none,
        # since we hold the lock) or a clock-skewed sibling is never yanked. Lock files
        # (``.lru.lock``) are owned by filelock and left in place.
        self._sweep_orphan_tmp_files()
        lru = self._ensure_lru()
        sizes = self._ensure_sizes()
        # Enumerate every on-disk entry from the in-memory size map, defaulting unseen-in-LRU ones to
        # ts=0 (legacy / external writes evict first). Identical to the prior glob+getsize scan: the
        # size map is built from the same ``glob("*.pkl")`` once per process then kept in sync.
        entries: List[Tuple[str, float, int]] = [(stem, float(lru.get(stem, 0.0)), size) for stem, size in sizes.items()]
        # Oldest first - that's the eviction order.
        entries.sort(key=lambda e: e[1])

        n = len(entries)
        total_bytes = self._total_bytes
        max_bytes = int(self.max_size_mb * 1024 * 1024) if self.max_size_mb is not None else None

        removed = 0
        i = 0
        while i < len(entries):
            over_count = self.max_entries is not None and n > self.max_entries
            over_size = max_bytes is not None and total_bytes > max_bytes
            if not over_count and not over_size:
                break
            stem, _ts, size = entries[i]
            path = os.path.join(self.cache_dir, f"{stem}.pkl")
            try:
                os.remove(path)
                removed += 1
                n -= 1
                total_bytes -= size
                lru.pop(stem, None)
                sizes.pop(stem, None)
                self._total_bytes -= size
                # Drop the sidecar only after the value file is gone: if the value remove raised (e.g. Windows lock) the entry survives and must keep its sidecar, else strict load refuses the still-present entry forever.
                try:
                    os.remove(path + ".sha256")
                except OSError:
                    pass
            except OSError:
                pass
            i += 1
        # Already holding the eviction filelock (see ``_evict_to_caps``); save directly rather than
        # via ``_flush_lru`` which would acquire a second, non-re-entrant FileLock on the same path.
        if removed:
            self._save_lru(lru)
            self._lru_dirty = False
        return removed

    def set(self, key: str, value: Any) -> None:
        """Write ``value`` to ``<cache_dir>/<key>.pkl``. Atomic via tmp-file rename so a crash mid-write doesn't leave corrupt cache files. ``f.flush()`` + ``os.fsync()`` run BEFORE ``os.replace`` so a power loss between pickle.dump returning and the OS flushing dirty pages cannot leave a zero-byte file under the visible name.

        After a successful write, the LRU sidecar is bumped and, if
        ``max_entries`` / ``max_size_mb`` are configured, least-recently-
        accessed entries are evicted to fit the caps.
        """
        path = self._path(key)
        # Write to a temp file in the same directory, then rename atomically.
        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        # ``fd`` ownership tracking (see io.py:atomic_write_bytes for the canonical pattern).
        # On the rare path where os.fdopen itself raises BEFORE the BufferedWriter adopts fd,
        # the raw fd would otherwise leak. set() runs on every cache write so leaks compound
        # quickly. Track adoption + manually close on failure.
        _fd_adopted = False
        try:
            with os.fdopen(fd, "wb") as f:
                _fd_adopted = True
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                # fsync inside the with-block so the data is on stable storage
                # BEFORE rename makes the path visible to readers. Without this,
                # rename can publish a name whose contents are still dirty pages
                # in the OS cache; a crash between rename and writeback leaves
                # a zero-byte file under the cache key.
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            # Write the sha256 sidecar AFTER the rename so the digest matches the on-disk bytes.
            # Sidecar write failures are logged at DEBUG (the value file is already durable); a
            # subsequent strict load will refuse the entry until the sidecar is regenerated, which
            # is the correct fail-closed behaviour.
            try:
                _safe_pickle_write_sidecar(path)
            except OSError as _sc_err:
                logger.debug("DiscoveryCache.set sidecar write failed (value written OK): %s", _sc_err)
            # POSIX requires ``fsync(dirfd)`` to make the new entry's directory metadata durable
            # across a power loss; without it the rename is visible
            # to readers but may revert after a crash on journaled-data-mode-off filesystems.
            # Windows NTFS does NOT expose directory fsync via ``os.fsync(dirfd)``
            # (``OSError: [Errno 13] Permission denied`` opening a dir for fsync), so we skip
            # the dir fsync on Windows and rely on the journaled-metadata guarantee NTFS already
            # provides for renames. On POSIX we attempt the dir fsync but treat failure as
            # non-fatal: the file fsync already happened, only metadata durability is at risk.
            if os.name == "posix":
                try:
                    _dir_fd = os.open(self.cache_dir, os.O_RDONLY)
                    try:
                        os.fsync(_dir_fd)
                    finally:
                        os.close(_dir_fd)
                except OSError:
                    pass
        except Exception:
            if not _fd_adopted:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        # Touch LRU AFTER the rename so the timestamp reflects the new
        # entry; then evict if caps are configured.
        #
        # A bare ``except Exception: pass`` here would silently swallow disk-full / lock-timeout /
        # corrupt-LRU errors during eviction. Log at DEBUG so an operator running with
        # ``logging.DEBUG`` sees the underlying cause, while normal runs are unaffected (the write
        # itself already succeeded before this block).
        try:
            _safe = self._safe_key(key)
            # Incrementally update the size accumulator with this entry's on-disk footprint
            # (replacing any prior size if the key was overwritten) instead of re-globbing every
            # ``*.pkl`` inside eviction. ``_ensure_sizes`` rebuilds from disk once if not yet primed.
            sizes = self._ensure_sizes()
            new_size = self._entry_size_on_disk(_safe)
            self._total_bytes += new_size - sizes.get(_safe, 0)
            sizes[_safe] = new_size
            self._touch_lru(_safe)
            self._evict_to_caps()
        except Exception as _evict_err:
            logger.debug("DiscoveryCache.set LRU/eviction failed (entry written OK): %s", _evict_err)

    def invalidate(self, key: str) -> bool:
        """Remove a cached entry. Returns True if the entry existed, False otherwise."""
        path = self._path(key)
        # TOCTOU race -- parallel hyperopt suites sharing cache_dir can both call
        # invalidate(same_key); an exists+remove pattern raises uncaught FileNotFoundError on the
        # loser. try/except makes concurrent invalidations of the same key idempotent.
        try:
            os.remove(path)
        except FileNotFoundError:
            return False
        # Also drop the sha256 sidecar so a future write of the same key starts fresh.
        try:
            os.remove(path + ".sha256")
        except OSError:
            pass
        # Mirror the deletion in the in-memory LRU ledger + size accumulator so a stale ledger
        # doesn't keep ghost keys pinning the count, and flush so the on-disk ledger drops the key.
        try:
            _safe = self._safe_key(key)
            sizes = self._ensure_sizes()
            self._total_bytes -= sizes.pop(_safe, 0)
            lru = self._ensure_lru()
            if lru.pop(_safe, None) is not None:
                self._lru_dirty = True
                self._flush_lru()
        except Exception as e:
            logger.debug("swallowed exception in cache_store.py: %s", e)
            pass
        return True

    def clear(self) -> int:
        """Remove all cached entries (and their sidecars). Returns the number of ``.pkl`` entries removed.

        Also sweeps orphan ``*.tmp`` files (interrupted ``set`` / ``_save_lru`` writes) and the
        ``.lru.lock`` filelock marker, which eviction never reclaims -- otherwise a long-lived
        cache_dir accumulates these forever. The returned count is still the number of cached
        ``.pkl`` ENTRIES removed (the contract callers rely on); swept tmp/lock files are logged
        at DEBUG, not counted, so existing assertions on the entry count are unaffected.
        """
        files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass
            try:
                os.remove(f + ".sha256")
            except OSError:
                pass
        try:
            if os.path.exists(self._lru_path):
                os.remove(self._lru_path)
        except OSError:
            pass
        # Reset the in-memory ledger + size accumulator to empty/clean so a later ``close`` /
        # ``__del__`` flush does not recreate a stale ``.lru`` for a directory we just wiped.
        self._lru = {}
        self._lru_dirty = False
        self._entry_sizes = {}
        self._total_bytes = 0
        # Sweep orphan tmp files (any age -- clear() is an explicit wipe, no in-flight write to
        # protect) and the filelock marker so a cleared dir is genuinely empty of cache cruft.
        _swept = 0
        for f in glob.glob(os.path.join(self.cache_dir, "*.tmp")):
            try:
                os.remove(f)
                _swept += 1
            except OSError:
                pass
        try:
            _lock = self._lock_path()
            if os.path.exists(_lock):
                os.remove(_lock)
                _swept += 1
        except OSError:
            pass
        if _swept:
            logger.debug("DiscoveryCache.clear: swept %d orphan tmp/lock file(s)", _swept)
        return len(files)


def _discovery_cache_bytes_total(cache: DiscoveryCache) -> int:
    """Best-effort on-disk byte total for a :class:`DiscoveryCache`. Mirrors ``_mrmr_cache_bytes_total`` / ``SuiteArtefactCache._total_bytes_locked`` so callers comparing against ``max_size_mb`` don't inline a per-call directory walk. Counts the .pkl plus its .pkl.sha256 sidecar -- both contribute to the on-disk budget."""
    total = 0
    try:
        with os.scandir(cache.cache_dir) as it:
            for de in it:
                if not de.is_file():
                    continue
                if not (de.name.endswith(".pkl") or de.name.endswith(".pkl.sha256")):
                    continue
                try:
                    total += de.stat().st_size
                except OSError:
                    pass
    except FileNotFoundError:
        return 0
    return total
