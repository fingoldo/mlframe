"""One psutil available-RAM probe, shared by the callers that size work against free memory.

Two modules carried their own ``_available_ram_bytes`` -- a filters one and a shap-proxy one -- with the
same body and DIFFERENT failure sentinels: ``-1`` meaning "no cap" and ``None`` meaning "drop the RAM term
from the auto-size". Merging them into one sentinel would have changed both contracts, so the probe is
shared and each caller keeps the sentinel its own callers already branch on.

The part worth sharing is the probe itself: the psutil import, the exception envelope, and the decision
that a failed probe is never fatal to the work it sizes. That is what drifts when a copy is edited.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger(__name__)


def available_ram_bytes(caller: str = "caller") -> Optional[int]:
    """Host available RAM in bytes, or ``None`` when psutil is absent or the probe fails.

    Never raises: a sizing hint that cannot be read must leave the caller free to proceed uncapped rather
    than take down the work it was meant to bound.
    """
    try:
        return int(host_memory()[0])
    except Exception as exc:
        # fallback: psutil is optional; without it the RAM term is dropped and the caller proceeds on its own budget.
        logger.debug("available_ram_bytes: psutil unavailable or failed (%s); %s proceeds without the RAM term", exc, caller)
        return None


_STATUS_TYPE: Any = None
_PSUTIL_VIRTUAL_MEMORY: Any = None  # the function psutil shipped, kept to notice a test double installed over it


def _status_type() -> Any:
    """The ``MEMORYSTATUSEX`` structure, built on first use (``ctypes`` is imported on Windows only)."""
    global _STATUS_TYPE
    if _STATUS_TYPE is None:
        import ctypes

        class _Status(ctypes.Structure):
            """``MEMORYSTATUSEX``."""

            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")
            ]

        _STATUS_TYPE = (_Status, ctypes.sizeof(_Status), ctypes.byref, ctypes.windll.kernel32.GlobalMemoryStatusEx)
    return _STATUS_TYPE


def _windows_memory() -> "tuple[int, int] | None":
    """``(available, total)`` physical bytes from ``GlobalMemoryStatusEx``, or ``None`` off Windows or when the call fails."""
    if sys.platform != "win32":
        return None
    status_type, size, byref, probe = _status_type()
    status = status_type()
    status.dwLength = size
    if not probe(byref(status)):
        return None
    return int(status.ullAvailPhys), int(status.ullTotalPhys)


def host_memory() -> "tuple[int, int]":
    """``(available, total)`` bytes of host RAM, the numbers ``psutil.virtual_memory()`` reports. Raises when neither probe works.

    ``psutil.virtual_memory()`` takes 2.1 ms on Windows (it also reads the commit and page-file counters); the FE cascades probe free memory at every family start and budget check, about
    0.5 s of a 100000 x 100 fit. ``GlobalMemoryStatusEx`` returns the same two numbers in 3 us. A ``psutil.virtual_memory`` replaced by a test double is honoured."""
    global _PSUTIL_VIRTUAL_MEMORY
    try:
        import psutil
    except ImportError:
        psutil = None  # type: ignore[assignment]
    if psutil is not None:
        if _PSUTIL_VIRTUAL_MEMORY is None:
            _PSUTIL_VIRTUAL_MEMORY = psutil.virtual_memory
        if psutil.virtual_memory is not _PSUTIL_VIRTUAL_MEMORY:
            vm = psutil.virtual_memory()
            return int(vm.available), int(getattr(vm, "total", 0) or 0)
    fast = _windows_memory()
    if fast is not None:
        return fast
    if psutil is None:
        raise RuntimeError("no host memory probe: psutil is not installed")
    vm = psutil.virtual_memory()
    return int(vm.available), int(vm.total)
