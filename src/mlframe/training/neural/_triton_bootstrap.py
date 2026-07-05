"""F-42 (2026-05-31): Triton bootstrap workaround for Windows.

triton-windows 3.7.0 fails to import on Windows when Python's standard
``LoadLibraryExW`` flags are used (``WinError 1114: A dynamic link
library (DLL) initialization routine failed``). The libtriton.pyd
extension has a delay-loaded dependency that's only findable when
the loader uses ``LOAD_WITH_ALTERED_SEARCH_PATH`` (winmode=0x8) --
which Python's import machinery does NOT pass by default.

Workaround: preload libtriton.pyd via ``ctypes.WinDLL(..., winmode=0x8)``
BEFORE any other imports that touch CUDA / Triton. The DLL stays
resident in process memory, and subsequent ``from triton._C.libtriton
import ...`` calls reuse the loaded module successfully.

This module is a no-op on:
  * Non-Windows hosts (the DLL workaround is Windows-specific)
  * Python sessions where Triton isn't installed
  * Sessions where libtriton.pyd is missing from the expected path

To use it: ``from mlframe.training.neural._triton_bootstrap import
ensure_triton_loaded; ensure_triton_loaded()`` BEFORE the first
``import triton`` in any module that needs Triton kernels.

Once Triton's upstream packaging fixes the Windows DLL search path
(tracked at https://github.com/woct0rdho/triton-windows), this
bootstrap can be removed.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_triton_loaded: Optional[bool] = None  # tri-state: None=untried, True=ok, False=failed


def ensure_triton_loaded() -> bool:
    """Idempotently preload libtriton.pyd on Windows. Returns True if
    Triton is now importable (or was already importable on the host
    platform), False if the bootstrap couldn't help."""
    global _triton_loaded
    if _triton_loaded is not None:
        return _triton_loaded

    if sys.platform != "win32":
        # Non-Windows: Triton's stock packaging works, no bootstrap needed.
        _triton_loaded = True
        return True

    try:
        import triton  # noqa: F401
        _triton_loaded = True
        return True  # already importable, no bootstrap needed
    except ImportError:
        pass  # fall through to the WinDLL preload

    try:
        import ctypes
        import site

        # Hunt libtriton.pyd in the active site-packages.
        candidates = []
        for sp in site.getsitepackages():
            candidates.append(os.path.join(sp, "triton", "_C", "libtriton.pyd"))
        # User-site fallback.
        try:
            candidates.append(os.path.join(site.getusersitepackages(), "triton", "_C", "libtriton.pyd"))
        except Exception:
            pass

        for pyd in candidates:
            if not os.path.exists(pyd):
                continue
            # winmode=0x8 -> LOAD_WITH_ALTERED_SEARCH_PATH; lets Windows
            # find delay-loaded deps that the stock import flags miss.
            try:
                ctypes.WinDLL(pyd, winmode=0x8)
                logger.info(
                    "F-42: Triton bootstrap preloaded libtriton.pyd via " "WinDLL (winmode=0x8). Source: %s",
                    pyd,
                )
                # Verify the import actually works now.
                import triton  # noqa: F401
                _triton_loaded = True
                return True
            except Exception as _preload_err:
                logger.debug(
                    "F-42: WinDLL preload failed for %s (%s); trying next candidate.",
                    pyd, _preload_err,
                )
                continue

        _triton_loaded = False
        logger.info(
            "F-42: Triton bootstrap could not preload libtriton.pyd; "
            "Triton-dependent code paths will use eager / non-Triton "
            "fallbacks. (Expected if Triton is not installed on this host.)"
        )
        return False
    except Exception as _bootstrap_err:
        _triton_loaded = False
        logger.warning(
            "F-42: Triton bootstrap raised unexpectedly (%s); " "Triton paths disabled.",
            _bootstrap_err,
        )
        return False


def is_triton_available() -> bool:
    """Check if Triton is importable + functional (after bootstrap if
    needed). Caches the result -- safe to call from hot paths."""
    return ensure_triton_loaded()
