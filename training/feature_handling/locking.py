"""
PID-aware filelock with stale-lock reclaim.

Round-3 chaos C5 + C16: the standard ``filelock`` package does NOT
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

Pin requirement: ``filelock>=3.15.0`` (round-3 chaos C16: 3.13.x had
a Windows bug that left .lock files behind on crash; 3.15+ cleans up
on context-manager exit, our PID layer covers the kill-9 case).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import psutil

logger = logging.getLogger(__name__)

try:
    import filelock as _filelock_mod
    from filelock import FileLock as _BaseFileLock, Timeout
    _FILELOCK_AVAILABLE = True
except ImportError:  # pragma: no cover -- optional dep until D phase wires cache layer
    _FILELOCK_AVAILABLE = False
    Timeout = TimeoutError  # type: ignore
    _BaseFileLock = object  # type: ignore


class StaleLockReclaimed(UserWarning):
    """Emitted when a lock left by a dead process was reclaimed.

    Surfaced as both a logged warning AND a Python ``warnings.warn``
    so test fixtures can use ``pytest.warns`` to assert reclaim
    happened (behavioural check beats source-grep).
    """


class PIDAwareFileLock:
    """Filelock with PID-tracking + stale-lock reclaim.

    Use as context manager:

        with PIDAwareFileLock(path, timeout=300):
            ...

    On timeout, checks the PID stored in the lock file. If the
    holder is dead (`psutil.pid_exists` returns False), removes
    the lock and retries once with a small grace timeout. Surfaces
    via ``StaleLockReclaimed``.

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
        reclaim_grace_timeout: float = 5.0,
    ):
        self.path = path
        self.timeout = timeout
        self.reclaim_grace_timeout = reclaim_grace_timeout
        self._lock: Optional[_BaseFileLock] = None
        self._held = False

    def __enter__(self) -> "PIDAwareFileLock":
        if not _FILELOCK_AVAILABLE:
            # Optional-dep absent path: act as a no-op so the cache layer
            # can still degrade gracefully. The cache layer also has its
            # own filesystem-capability self-test before relying on
            # cross-process semantics.
            self._held = True
            return self

        self._lock = _BaseFileLock(self.path)

        try:
            self._lock.acquire(timeout=self.timeout)
        except Timeout:
            # Stale-lock detection: read PID from lockfile, check if the
            # holder is alive. If dead, remove and retry once.
            holder_pid = self._read_holder_pid()
            if holder_pid is not None and not psutil.pid_exists(holder_pid):
                logger.warning(
                    "PIDAwareFileLock: lock %s held by dead PID %d; reclaiming",
                    self.path, holder_pid,
                )
                # NOTE: this race is acceptable -- if a third process
                # reclaims simultaneously we'd just retry-acquire below
                # and get a fresh lock. The danger would be reclaiming
                # a LIVE PID's lock; the pid_exists check guards against it.
                try:
                    os.unlink(self.path)
                except FileNotFoundError:
                    pass
                # Surface to callers via Python warnings so test fixtures
                # can `pytest.warns(StaleLockReclaimed)`.
                import warnings
                warnings.warn(
                    f"reclaimed lock {self.path} from dead PID {holder_pid}",
                    StaleLockReclaimed,
                    stacklevel=2,
                )
                # Retry once with the grace timeout.
                self._lock.acquire(timeout=self.reclaim_grace_timeout)
            else:
                raise  # holder is alive (or no PID written) -- propagate timeout

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
            self._lock.release()
        finally:
            self._held = False

    def _meta_path(self) -> str:
        """Path of the sidecar PID file."""
        return self.path + ".pid"

    def _write_holder_pid(self) -> None:
        try:
            with open(self._meta_path(), "w") as f:
                f.write(str(os.getpid()))
        except OSError:
            # If we can't write the meta, accept that future contenders
            # won't be able to detect our liveness -- they'll just wait
            # the full timeout and never reclaim. Better than crashing.
            pass

    def _read_holder_pid(self) -> Optional[int]:
        try:
            with open(self._meta_path(), "r") as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return None
        except OSError:  # pragma: no cover
            return None
