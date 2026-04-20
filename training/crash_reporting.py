"""Process-level crash reporting for long training runs.

On Windows, a native C++ ``std::bad_alloc`` / access violation / abort
inside XGBoost, CatBoost, LightGBM, or numba can bypass Python's
exception chain and surface as a "Python has stopped working" Windows
Error Reporting (WER) dialog. That dialog is modal: the Jupyter kernel
freezes until a human clicks it, and nothing useful reaches the cell
output.

This module provides a single-call toggle that:

  1. enables ``faulthandler`` so fatal signals (SIGSEGV, SIGFPE,
     SIGABRT, SIGBUS, SIGILL) print a Python traceback to stderr
     *before* the process dies, and
  2. on Windows only, calls ``SetErrorMode`` to suppress the WER
     dialog so the kernel exits cleanly with a non-zero exit code
     instead of hanging on a modal popup.

``XGBoostError`` / ``CatBoostError`` / etc. that are already thrown as
Python exceptions are unaffected — they still propagate normally and
can be caught by ``try/except``. This toggle only fixes the case where
the native runtime converts the error into an OS-level signal.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# Guard so repeated calls are cheap and idempotent (users frequently
# re-invoke train_mlframe_models_suite in the same process).
_ENABLED: bool = False


def enable_crash_reporting(file=None, all_threads: bool = True) -> bool:
    """Enable faulthandler + suppress Windows Error Reporting popup.

    Parameters
    ----------
    file : file-like, optional
        Where faulthandler writes the traceback. Defaults to
        ``sys.stderr`` which is what Jupyter shows in cell output.
    all_threads : bool, default True
        If True, dump tracebacks of all threads on a fatal signal —
        useful when the crash originates in an OpenMP worker thread
        (XGBoost, CatBoost, numba all use OMP).

    Returns
    -------
    bool
        True if fully enabled on this platform, False if a step failed
        (the call never raises — a failure to enable crash reporting
        must not break training).
    """
    global _ENABLED
    if _ENABLED:
        return True

    ok = True

    # Step 1: faulthandler — works on all platforms.
    try:
        import faulthandler
        stream = file if file is not None else sys.stderr
        faulthandler.enable(file=stream, all_threads=all_threads)
    except Exception as e:
        logger.warning(f"faulthandler.enable() failed: {e}")
        ok = False

    # Step 2: Windows-only — suppress "Python has stopped working" modal.
    if sys.platform == "win32":
        try:
            import ctypes
            # SEM_FAILCRITICALERRORS = 0x0001: suppresses the
            #   system-modal dialog asking the user to insert a disk
            #   (irrelevant for us but part of the standard combo).
            # SEM_NOGPFAULTERRORBOX  = 0x0002: suppresses the actual
            #   "Program has stopped working" WER dialog. Without this,
            #   Jupyter freezes indefinitely on a bad_alloc.
            SEM_FAILCRITICALERRORS = 0x0001
            SEM_NOGPFAULTERRORBOX = 0x0002
            prev = ctypes.windll.kernel32.SetErrorMode(0)  # read current
            ctypes.windll.kernel32.SetErrorMode(
                prev | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
            )
        except Exception as e:
            logger.warning(f"SetErrorMode() failed: {e}")
            ok = False

    _ENABLED = ok
    if ok:
        logger.info("Crash reporting enabled (faulthandler + WER suppression on Windows)")
    return ok
