"""
PID-aware filelock with stale-lock reclaim.

The standard ``filelock`` package does NOT
detect when the process holding a lock has died (OOM-killed,
SIGSEGV-crashed, kernel-panic'd). The next acquirer blocks
indefinitely on a ghost lock. For mlframe's cross-process cache,
that means one OOM-kill bricks the cache directory until manual
``rm cache_dir/v3/.locks/*.lock``.

This module wraps ``filelock.FileLock`` with two extras:

1. On acquire, write the current PID into the lock file alongside
   the OS-level exclusive lock (POSIX ``fcntl`` / Windows ``msvcrt``).
2. On acquisition timeout, read the PID from the lock file and
   call ``psutil.pid_exists()``; if dead, remove the lock and retry
   ONCE. Surface the reclaim via ``StaleLockReclaimed`` log event
   so users know an orphan was cleaned up.

Pin requirement: ``filelock>=3.15.0`` (3.13.x had a Windows bug that
left .lock files behind on crash; 3.15+ cleans up on context-manager
exit, our PID layer covers the kill-9 case).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import filelock as _filelock_mod  # noqa: F401
    from filelock import FileLock as _BaseFileLock
    _FILELOCK_AVAILABLE = True
except ImportError:  # pragma: no cover -- optional dep until D phase wires cache layer
    _FILELOCK_AVAILABLE = False
    _BaseFileLock = object  # type: ignore[assignment, misc]


class StaleLockReclaimed(UserWarning):
    """Emitted when a lock left by a dead process was reclaimed.

    Surfaced as both a logged warning AND a Python ``warnings.warn``
    so test fixtures can use ``pytest.warns`` to assert reclaim
    happened (behavioural check beats source-grep).
    """


class PIDAwareFileLock:
    """Filelock that records the holder's PID next to the lock.

    Use as context manager:

        with PIDAwareFileLock(path, timeout=300):
            ...

    On timeout the caller sees ``Timeout``: a lock that is held is never taken over, because the PID in the sidecar file can lag behind the real
    holder (a new holder writes its PID only after acquiring), and removing the lock file on that basis lets a second process in.

    Pin requirement: ``filelock>=3.15.0``. If the package is missing
    (it's an optional dep until phase D wires the cache layer), this
    class still constructs and the context manager becomes a no-op
    so importing mlframe doesn't blow up on minimal installs -- the
    consumer (cache layer) gates use behind an availability check.
    """

    def __init__(
        self,
        path: str,
        timeout: float = 300.0,
    ):
        self.path = path
        self.timeout = timeout
        self._lock: Optional[_BaseFileLock] = None
        self._held = False

    def __enter__(self) -> PIDAwareFileLock:
        if not _FILELOCK_AVAILABLE:
            # Optional-dep absent path: act as a no-op so the cache layer
            # can still degrade gracefully. The cache layer also has its
            # own filesystem-capability self-test before relying on
            # cross-process semantics.
            self._held = True
            return self

        self._lock = _BaseFileLock(self.path)

        # A Timeout here means a live process owns the OS lock; it propagates to the caller and the lock file is left alone.
        self._lock.acquire(timeout=self.timeout)

        # Write our PID alongside the exclusive lock so a future
        # contender can detect us if WE die holding it.
        self._write_holder_pid()
        self._held = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._held:
            return
        if not _FILELOCK_AVAILABLE or self._lock is None:
            self._held = False
            return
        try:
            # Best effort: unlink first so a SIGKILL between unlink and
            # release doesn't strand a PID file (rare race).
            try:
                os.unlink(self._meta_path())
            except FileNotFoundError:
                pass
            # Wrap release() in its own try/except so a filelock Timeout / OSError on lock-file unlink
            # (Windows paging stress) doesn't mask the in-flight exception from the `with` body AND
            # doesn't propagate a release-side error past the user.
            try:
                self._lock.release()
            except Exception as _rel_err:  # best-effort: the OS reclaims the lock on process exit regardless
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "PIDAwareFileLock.release() failed for %s: %s", self.path, _rel_err,
                )
        finally:
            self._held = False

    def _meta_path(self) -> str:
        """Path of the sidecar PID file."""
        return self.path + ".pid"

    def _write_holder_pid(self) -> None:
        """Record this process's PID in the sidecar file so a future contender can detect a stale (dead-process) lock and reclaim it. Best-effort: an OSError here degrades to "future contenders wait out the full timeout" rather than crashing the lock holder."""
        # Atomic via temp-file + os.replace so a SIGKILL between create
        # and write cannot leave an empty PID file (which would make
        # ``_read_holder_pid`` return None, breaking stale-lock reclaim).
        try:
            target = self._meta_path()
            tmp = target + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(str(os.getpid()))
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, target)
        except OSError:
            # If we can't write the meta, accept that future contenders
            # won't be able to detect our liveness -- they'll just wait
            # the full timeout and never reclaim. Better than crashing.
            try:
                os.unlink(target + ".tmp")
            except OSError:
                pass

    def _read_holder_pid(self) -> Optional[int]:
        """Read the current lock holder's PID from the sidecar file; None if the file is missing, empty, or unreadable (treated as "liveness unknown", not an error)."""
        try:
            with open(self._meta_path(), encoding="utf-8") as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError) as _exc:
            logger.debug("_read_holder_pid fell back to its default after %s", type(_exc).__name__)
            return None
        except OSError as _exc:  # pragma: no cover
            logger.debug("_read_holder_pid fell back to its default after %s", type(_exc).__name__)
            return None
