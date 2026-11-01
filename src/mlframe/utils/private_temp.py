"""A per-user temp directory that another local user cannot have pre-created or opened up.

A fixed name under the shared temp root (``/tmp/mlframe_caches``) can be created by any local user before the first run, with a mode and an owner of their
choosing, and then read or poisoned. ``private_temp_dir`` names the directory per user, creates it ``0o700`` and, on POSIX, checks that what it found is a
real directory owned by the caller with no group or other access; anything else is replaced by a fresh ``mkdtemp`` directory.
"""

from __future__ import annotations

import logging
import os
import stat
import tempfile

logger = logging.getLogger(__name__)

__all__ = ["private_temp_dir"]


def _user_tag() -> str:
    """A filesystem-safe tag for the current user."""
    uid = os.geteuid() if hasattr(os, "geteuid") else None
    raw = str(uid) if uid is not None else os.environ.get("USERNAME") or os.environ.get("USER") or "user"
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)


def _is_private(st: os.stat_result, euid: int | None) -> bool:
    """Whether ``st`` describes a real directory owned by ``euid`` that group and other cannot touch (``euid`` None skips the POSIX checks)."""
    if not stat.S_ISDIR(st.st_mode):
        return False
    if euid is None:
        return True
    return st.st_uid == euid and (st.st_mode & 0o077) == 0


def private_temp_dir(name: str) -> str:
    """Path of a per-user ``0o700`` directory ``<tempdir>/<name>-<user>``, created if needed; a fresh ``mkdtemp`` directory if the existing one is not private."""
    path = os.path.join(tempfile.gettempdir(), f"{name}-{_user_tag()}")
    os.makedirs(path, mode=0o700, exist_ok=True)
    euid = os.geteuid() if hasattr(os, "geteuid") else None
    if not _is_private(os.lstat(path), euid):
        fresh = tempfile.mkdtemp(prefix=f"{name}-")
        logger.warning("private_temp_dir: %s is not a private directory owned by this user; using %s instead.", path, fresh)
        return fresh
    return path
