"""
System-level helpers for the feature-handling subsystem.

Bundles three small but load-bearing utilities:

* :func:`detect_memory_limit_bytes` -- cgroup-aware memory probe so
  containerised runs (Docker / K8s) honour their cgroup limit instead
  of the host RAM (container-blind budget on a 4 GB container of a
  256 GB host derives ``budget=179 GB`` and OOM-kills instantly).

* :func:`classify_cuda_error` + :class:`CudaErrorClass` -- splits the
  retryable ``OutOfMemoryError`` from the "context-lost, restart
  Python" generic CUDA error so the halve-batch retry loop doesn't
  spin forever after a driver crash.

* :func:`long_path_safe` -- prepends the ``\\?\\`` UNC marker on
  Windows so cache paths longer than 260 chars don't ``FileNotFoundError``
  through ``os.replace``.
"""

from __future__ import annotations

import logging
import os
import sys
from enum import Enum
from typing import Optional

import psutil

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# cgroup-aware memory probe
# ---------------------------------------------------------------------

_CGROUP_V2_MAX = "/sys/fs/cgroup/memory.max"
_CGROUP_V1_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_CGROUP_V1_UNLIMITED_SENTINEL = 9223372036854771712  # 2^63 page-aligned


def _read_cgroup_memory_limit_bytes() -> Optional[int]:
    """Return the cgroup memory limit in bytes, or ``None`` if not in
    a cgroup or limit is unset.

    Handles both cgroup v2 (``memory.max`` contains literal ``"max"``
    when unlimited) and cgroup v1 (a 9.2e18 sentinel when unlimited).
    """
    if sys.platform != "linux":
        return None
    # cgroup v2
    if os.path.exists(_CGROUP_V2_MAX):
        try:
            with open(_CGROUP_V2_MAX) as f:
                value = f.read().strip()
            if value == "max":
                return None
            return int(value)
        except (OSError, ValueError):
            return None
    # cgroup v1
    if os.path.exists(_CGROUP_V1_LIMIT):
        try:
            with open(_CGROUP_V1_LIMIT) as f:
                limit = int(f.read().strip())
            if limit >= _CGROUP_V1_UNLIMITED_SENTINEL:
                return None
            return limit
        except (OSError, ValueError):
            return None
    return None


def detect_memory_limit_bytes() -> int:
    """Return the effective memory limit in bytes, honouring cgroup
    constraints inside containers.

    Resolution order:
      1. ``MLFRAME_MEMORY_BUDGET_GB`` env override (explicit operator
         intent always wins);
      2. ``min(psutil.virtual_memory().total, cgroup_limit)`` when in
         a container with a memory limit set;
      3. ``psutil.virtual_memory().total`` otherwise.

    Falls back gracefully to ``psutil`` on any read error so a weird
    cgroup layout never crashes FHC construction.
    """
    env = os.environ.get("MLFRAME_MEMORY_BUDGET_GB")
    if env:
        try:
            return int(float(env) * 1e9)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid MLFRAME_MEMORY_BUDGET_GB=%r; ignoring and falling back to psutil/cgroup",
                env,
            )

    try:
        psutil_total = psutil.virtual_memory().total
    except Exception as e:  # pragma: no cover -- exotic Linux configs
        logger.warning("psutil.virtual_memory failed (%s); assuming 8 GB", e)
        return int(8 * 1e9)

    cgroup_limit = _read_cgroup_memory_limit_bytes()
    if cgroup_limit is not None and cgroup_limit < psutil_total:
        return cgroup_limit
    return psutil_total


# ---------------------------------------------------------------------
# CUDA error taxonomy
# ---------------------------------------------------------------------

class CudaErrorClass(str, Enum):
    """How to react to a CUDA exception.

    ``OUT_OF_MEMORY`` -- retryable; halve batch size and try again.
    ``CONTEXT_LOST`` -- driver crash or unknown CUDA error; the CUDA
    context in this process is unrecoverable, abort and ask the user
    to restart Python (no clean recovery exists).
    ``OTHER`` -- anything we don't have a recipe for; treat conservatively
    (don't retry, surface as is).
    """

    OUT_OF_MEMORY = "out_of_memory"
    CONTEXT_LOST = "context_lost"
    OTHER = "other"


def classify_cuda_error(exc: BaseException) -> CudaErrorClass:
    """Classify a CUDA-flavoured exception into one of the three buckets.

    The discriminator runs in two stages:
      1. ``isinstance`` check against ``torch.cuda.OutOfMemoryError``
         when torch is present (most reliable signal).
      2. Substring sniff on ``str(exc)`` for ``"out of memory"``
         (covers the case where torch isn't imported or upstream
         renamed the exception class).

    Generic CUDA errors with ``"unknown error"`` / ``"context"`` /
    ``"misaligned"`` substrings → ``CONTEXT_LOST`` (caller should
    abort, not retry). Anything else → ``OTHER``.
    """
    msg = str(exc).lower()

    # Stage 1: typed check (most reliable)
    try:
        import torch  # type: ignore
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return CudaErrorClass.OUT_OF_MEMORY
    except ImportError:  # pragma: no cover
        pass

    # Stage 2: substring sniff
    if "out of memory" in msg or "cuda out of memory" in msg:
        return CudaErrorClass.OUT_OF_MEMORY

    if any(needle in msg for needle in (
        "unknown error",
        "context is destroyed",
        "context was destroyed",
        "an illegal memory access",
        "misaligned address",
        "device-side assert",
    )):
        return CudaErrorClass.CONTEXT_LOST

    if "cuda error" in msg:
        # Unknown flavor of CUDA error -- treat as context-lost rather
        # than retrying. Better to surface a clear "restart Python"
        # message than to spin forever.
        return CudaErrorClass.CONTEXT_LOST

    return CudaErrorClass.OTHER


# ---------------------------------------------------------------------
# Windows long-path helper
# ---------------------------------------------------------------------

_LONG_PATH_PREFIX = "\\\\?\\"


def long_path_safe(path: str) -> str:
    """On Windows, prepend the ``\\?\\`` UNC marker to bypass the
    260-char ``MAX_PATH`` ceiling so deep cache directories don't
    blow up at ``os.replace`` time. No-op on POSIX.

    Idempotent: re-applying to an already-prefixed path returns it
    unchanged.
    """
    if sys.platform != "win32":
        return path
    if path.startswith(_LONG_PATH_PREFIX):
        return path
    abspath = os.path.abspath(path)
    return _LONG_PATH_PREFIX + abspath
