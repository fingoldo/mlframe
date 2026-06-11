"""Disk-backed ``DiscoveryCache`` store carved out of ``cache.py`` (monolith-split, 2026-06-11).

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
import pickle
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from mlframe.utils.safe_pickle import (
    PickleVerificationError,
    safe_load as _safe_pickle_load,
    verify_sidecar as _safe_pickle_verify_sidecar,
    write_sidecar as _safe_pickle_write_sidecar,
)

logger = logging.getLogger(__name__)

# Audit D L-4 (2026-05-18): pre-compiled hex matcher replaces the
# ``all(c in "0123456789abcdefABCDEF" for c in key)`` generator-expression in ``_safe_key`` per
# the user's ``feedback_orjson_compile_regex`` rule (compile once, reuse). The regex anchors with
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

    # ------------------------------------------------------------------
    # LRU sidecar (key -> access timestamp). Plain JSON; tiny so we read
    # / write the whole file on every touch. Atime is too unreliable on
    # NTFS to depend on.
    #
    # DISC-LRU-RACE / DISC-RACE-UNPROT: file-lock the sidecar so two
    # concurrent processes hitting the same data_dir can't interleave
    # an evict + write and leave live entries marked stale. ``filelock``
    # is optional -- absence falls back to the pre-fix racy behaviour
    # with a one-time WARN.
    # ------------------------------------------------------------------

    def _lock_path(self) -> str:
        return self._lru_path + ".lock"

    @staticmethod
    def _maybe_filelock(lock_path: str):
        """Return a ``filelock.FileLock`` instance if the dep is present, else a no-op context."""
        try:
            from filelock import FileLock as _FileLock  # type: ignore[import-untyped]
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

    def _touch_lru(self, key: str) -> None:
        # Cross-process file lock around read-modify-write so a concurrent process can't replay a
        # stale-snapshot save and overwrite a fresh access timestamp. filelock is optional.
        #
        # Audit D L-1 (2026-05-18): ``time.time()`` is wall-clock (subject to NTP step-back) but
        # we deliberately do NOT use ``time.monotonic()`` here -- the LRU sidecar is shared across
        # processes, and monotonic clock values are not comparable across processes (each
        # process's monotonic clock starts from an arbitrary reference). Cross-process LRU
        # ordering requires a shared reference frame -- wall clock is the only portable option.
        # Single-host NTP step-backs are rare and only mis-order entries within the step delta.
        #
        # Audit D P2-2 (2026-05-18): the full JSON rewrite per touch is O(N entries). For caches
        # >10,000 entries the rewrite cost dominates the read; we keep the simple
        # write-everything-each-touch design because:
        #   (a) the rewrite happens under the cross-process filelock, so a "mark dirty, flush
        #       later" batching strategy would require additional cross-process flush
        #       synchronisation;
        #   (b) the atomic-rename + fsync guarantee survives a mid-touch crash;
        #   (c) typical R&D cache sizes are <500 entries where the rewrite is sub-millisecond.
        # Operators hitting the >10K-entry regime should pass ``max_entries`` to keep the file
        # bounded.
        with self._maybe_filelock(self._lock_path()):
            lru = self._load_lru()
            lru[key] = time.time()
            self._save_lru(lru)

    def _path(self, key: str) -> str:
        safe_key = self._safe_key(key)
        return os.path.join(self.cache_dir, f"{safe_key}.pkl")

    def __contains__(self, key: str) -> bool:
        # Audit D L-5 (2026-05-18): ``os.path.exists`` is racy by design (TOCTOU: the file
        # can be deleted between the check and a subsequent ``get``). Callers should use
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
                _file_size / 1024 ** 3,
                _max_bytes / 1024 ** 3,
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
        except Exception:
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
        # Wave 52 (2026-05-20): capture sys.exc_info() so the in-flight exception is
        # forwarded to the lock manager's __exit__ (preserves CM contract), AND wrap
        # __exit__ itself in try/except so a filelock cleanup error doesn't mask the
        # eviction-body exception.
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
        """Remove stale ``*.tmp`` files left by interrupted ``set`` / ``_save_lru`` writes (S18).

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
        # Best-effort orphan sweep before sizing: a crashed/interrupted ``set`` or ``_save_lru``
        # can leave a ``*.tmp`` in cache_dir that ``glob("*.pkl")`` never sees, so eviction's
        # byte cap silently under-counts the true footprint (S18). The filelock around eviction
        # makes this safe -- no concurrent ``set`` is mid-rename on a tmp we'd delete -- but we
        # still age-gate so a tmp from a write that is genuinely in flight in THIS process (none,
        # since we hold the lock) or a clock-skewed sibling is never yanked. Lock files
        # (``.lru.lock``) are owned by filelock and left in place.
        self._sweep_orphan_tmp_files()
        lru = self._load_lru()
        # Enumerate every on-disk entry, defaulting unseen ones to ts=0.
        files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        entries: List[Tuple[str, float, int]] = []
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0]
            ts = float(lru.get(stem, 0.0))
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            # S18: count the ``.pkl.sha256`` sidecar in the entry's footprint so the eviction
            # byte cap agrees with ``_discovery_cache_bytes_total`` (which counts .pkl + sidecar).
            # Pre-fix the cap measured only the .pkl, so a cache reported as over ``max_size_mb``
            # by the helper could still refuse to evict (cap thought it was under budget).
            try:
                size += os.path.getsize(path + ".sha256")
            except OSError:
                pass
            entries.append((stem, ts, size))
        # Oldest first - that's the eviction order.
        entries.sort(key=lambda e: e[1])

        n = len(entries)
        total_bytes = sum(s for _, _, s in entries)
        max_bytes = (
            int(self.max_size_mb * 1024 * 1024)
            if self.max_size_mb is not None else None
        )

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
                # Drop the sidecar only after the value file is gone: if the value remove raised (e.g. Windows lock) the entry survives and must keep its sidecar, else strict load refuses the still-present entry forever.
                try:
                    os.remove(path + ".sha256")
                except OSError:
                    pass
            except OSError:
                pass
            i += 1
        if removed:
            self._save_lru(lru)
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
            # Audit D L-10 (2026-05-18): POSIX requires ``fsync(dirfd)`` to make the new entry's
            # directory metadata durable across a power loss; without it the rename is visible
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
        # Audit D L-6 (2026-05-18): the previous ``except Exception: pass`` silently swallowed
        # disk-full / lock-timeout / corrupt-LRU errors during eviction. Log at DEBUG so an
        # operator running with ``logging.DEBUG`` sees the underlying cause, while normal runs
        # are unaffected (the write itself already succeeded before this block).
        try:
            self._touch_lru(self._safe_key(key))
            self._evict_to_caps()
        except Exception as _evict_err:
            logger.debug("DiscoveryCache.set LRU/eviction failed (entry written OK): %s", _evict_err)

    def invalidate(self, key: str) -> bool:
        """Remove a cached entry. Returns True if the entry existed, False otherwise."""
        path = self._path(key)
        # Wave 48 (2026-05-20): TOCTOU race -- parallel hyperopt suites sharing
        # cache_dir can both call invalidate(same_key); the prior exists+remove
        # pattern raised uncaught FileNotFoundError on the loser. Replace with
        # try/except so concurrent invalidations of the same key are idempotent.
        try:
            os.remove(path)
        except FileNotFoundError:
            return False
        # Also drop the sha256 sidecar so a future write of the same key starts fresh.
        try:
            os.remove(path + ".sha256")
        except OSError:
            pass
        # Mirror the deletion in the LRU sidecar so a stale ledger
        # doesn't keep ghost keys pinning the count.
        try:
            lru = self._load_lru()
            if lru.pop(self._safe_key(key), None) is not None:
                self._save_lru(lru)
        except Exception:
            pass
        return True

    def clear(self) -> int:
        """Remove all cached entries (and their sidecars). Returns the number of ``.pkl`` entries removed.

        S18: also sweeps orphan ``*.tmp`` files (interrupted ``set`` / ``_save_lru`` writes) and
        the ``.lru.lock`` filelock marker, which eviction never reclaims -- pre-fix a long-lived
        cache_dir accumulated these forever. The returned count is still the number of cached
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
