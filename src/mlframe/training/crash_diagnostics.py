"""Make an abrupt death of a long training process diagnosable from the log file alone.

faulthandler writes to a file descriptor, not to ``logging``, so a native crash / OOM kill / paging-file exhaustion left
nothing in the run log: normal INFO lines, then silence, then the next process starting. This module adds, all best-effort
(nothing here may raise into training):

* a persistent faulthandler file next to the log (path logged at startup, file object kept alive for the process lifetime);
* ``sys.excepthook`` / ``threading.excepthook`` that route uncaught exceptions to the logger with the full traceback;
* an ``atexit`` line ("suite process exiting ..."), so its ABSENCE in a log proves the process was killed abruptly;
* a periodic heartbeat thread logging the active phase(s), process memory, system commit headroom and GPU memory, so the
  last heartbeat before a death shows the memory pressure it happened under;
* on Windows, the commit limit / available page file at startup (WinError 1455 is a known failure mode on big runs).

Knobs: ``MLFRAME_CRASH_LOG_DIR`` (faulthandler file directory), ``MLFRAME_CRASH_HEARTBEAT_S`` (heartbeat period, 0 disables,
default 300).
"""
from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_S = 300.0

_FAULT_FILE: Any = None  # kept referenced for the process lifetime: faulthandler holds only the fd
_FAULT_PATH: Optional[str] = None
_HOOKS_INSTALLED = False
_ATEXIT_REGISTERED = False
_UNCAUGHT: Dict[str, Any] = {"exc": None}
_HEARTBEAT: Optional["Heartbeat"] = None
_START_TIME = time.time()


# ----------------------------------------------------------------------------------------------------------------------
# Memory probes
# ----------------------------------------------------------------------------------------------------------------------


def windows_commit_status() -> Optional[Dict[str, float]]:
    """GlobalMemoryStatusEx in GB: physical total/available and commit limit/available (page-file backed). Windows only."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return None
        g = 1024.0**3
        return {
            "phys_total_gb": ms.ullTotalPhys / g, "phys_avail_gb": ms.ullAvailPhys / g,
            "commit_limit_gb": ms.ullTotalPageFile / g, "commit_avail_gb": ms.ullAvailPageFile / g,
            "pagefile_gb": max(0.0, (ms.ullTotalPageFile - ms.ullTotalPhys) / g),
        }
    except Exception:
        return None


def memory_line() -> str:
    """Process RSS / private commit plus system available RAM (and commit headroom on Windows). Never raises."""
    parts = []
    try:
        import psutil

        mi = psutil.Process().memory_info()
        parts.append(f"rss={mi.rss / 1024**3:.1f}GB")
        private = getattr(mi, "private", None)
        if private:
            parts.append(f"private_commit={private / 1024**3:.1f}GB")
        vm = psutil.virtual_memory()
        parts.append(f"sys_avail={vm.available / 1024**3:.1f}/{vm.total / 1024**3:.1f}GB")
    except Exception:
        parts.append("mem=n/a")
    cs = windows_commit_status()
    if cs:
        parts.append(f"commit_avail={cs['commit_avail_gb']:.1f}/{cs['commit_limit_gb']:.1f}GB")
    return " ".join(parts)


# ----------------------------------------------------------------------------------------------------------------------
# faulthandler file
# ----------------------------------------------------------------------------------------------------------------------


def resolve_crash_dir(crash_dir: Optional[str] = None) -> str:
    """Explicit dir > MLFRAME_CRASH_LOG_DIR > directory of the first logging FileHandler > system temp dir."""
    if crash_dir:
        return str(crash_dir)
    env = os.environ.get("MLFRAME_CRASH_LOG_DIR", "").strip()
    if env:
        return env
    try:
        for lg in (logging.getLogger(), logger):
            cur: Any = lg
            while cur is not None:
                for h in getattr(cur, "handlers", []):
                    fn = getattr(h, "baseFilename", None)
                    # pytest's logging plugin installs a FileHandler on os.devnull; that is not a real log location.
                    if fn and os.path.basename(fn).lower() not in ("nul", "null") and os.path.abspath(fn) != os.path.abspath(os.devnull):
                        return os.path.dirname(os.path.abspath(fn))
                cur = cur.parent if getattr(cur, "propagate", False) else None
    except Exception:
        pass
    import tempfile

    return tempfile.gettempdir()


def open_faulthandler_file(crash_dir: Optional[str] = None, all_threads: bool = True) -> Optional[str]:
    """Point faulthandler at a persistent per-process file and return its path (``None`` on failure). Idempotent."""
    global _FAULT_FILE, _FAULT_PATH
    if _FAULT_FILE is not None and not _FAULT_FILE.closed:
        return _FAULT_PATH
    try:
        import faulthandler

        d = resolve_crash_dir(crash_dir)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"mlframe_faulthandler_{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}.log")
        f = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - must outlive this call
        f.write(f"mlframe faulthandler file; pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M:%S')} argv={sys.argv!r}\n")
        f.flush()
        faulthandler.enable(file=f, all_threads=all_threads)
        _FAULT_FILE, _FAULT_PATH = f, path
        return path
    except Exception as e:
        logger.warning("Could not open a persistent faulthandler file: %s", e)
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Exception hooks + atexit
# ----------------------------------------------------------------------------------------------------------------------


def _sys_excepthook(exc_type, exc, tb, _prev=None):
    try:
        if not issubclass(exc_type, KeyboardInterrupt):
            _UNCAUGHT["exc"] = f"{exc_type.__name__}: {exc}"
            logger.critical("Uncaught exception in main thread:\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        else:
            _UNCAUGHT["exc"] = "KeyboardInterrupt"
    except Exception:
        pass
    prev = _prev or sys.__excepthook__
    try:
        prev(exc_type, exc, tb)
    except Exception:
        pass


def _thread_excepthook(args, _prev=None):
    try:
        if args.exc_type is not SystemExit:
            tname = getattr(args.thread, "name", "?")
            logger.critical(
                "Uncaught exception in thread %s:\n%s", tname,
                "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
            )
    except Exception:
        pass
    prev = _prev or threading.__excepthook__
    try:
        prev(args)
    except Exception:
        pass


def install_exception_hooks() -> None:
    """Chain logger-routing hooks in front of the current ``sys.excepthook`` / ``threading.excepthook``. Idempotent."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    prev_sys = sys.excepthook
    prev_thr = threading.excepthook
    sys.excepthook = lambda t, e, tb: _sys_excepthook(t, e, tb, _prev=prev_sys)
    threading.excepthook = lambda a: _thread_excepthook(a, _prev=prev_thr)
    _HOOKS_INSTALLED = True


def _atexit_handler() -> None:
    try:
        up = time.time() - _START_TIME
        if _UNCAUGHT["exc"]:
            logger.warning("mlframe suite process exiting after an uncaught exception (%s); uptime %.0fs. %s", _UNCAUGHT["exc"], up, memory_line())
        else:
            logger.info("mlframe suite process exiting normally; uptime %.0fs. %s", up, memory_line())
    except Exception:
        pass
    try:
        if _HEARTBEAT is not None:
            _HEARTBEAT.stop(join=False)
    except Exception:
        pass


def register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_handler)
        _ATEXIT_REGISTERED = True


# ----------------------------------------------------------------------------------------------------------------------
# Heartbeat
# ----------------------------------------------------------------------------------------------------------------------


def heartbeat_line() -> str:
    """Active phases + memory + GPU memory in one line. Never raises."""
    try:
        from .phases import all_active_phases

        ph = all_active_phases()
        phase_s = "; ".join(f"{k}: {v}" for k, v in ph.items()) if ph else "(no active phase)"
    except Exception:
        phase_s = "?"
    try:
        from ._gpu_state_probe import format_gpu_snapshot, gpu_snapshot

        gpu_s = format_gpu_snapshot(gpu_snapshot(), exclude_pid=None, max_procs=0)
    except Exception:
        gpu_s = "gpu=?"
    return f"[heartbeat] phase={phase_s} | {memory_line()} | {gpu_s}"


class Heartbeat:
    """Daemon thread logging :func:`heartbeat_line` every ``interval_s`` seconds until :meth:`stop`."""

    def __init__(self, interval_s: float, log: Optional[logging.Logger] = None, line_fn=heartbeat_line):
        self.interval_s = float(interval_s)
        self._log = log or logger
        self._line_fn = line_fn
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mlframe-heartbeat", daemon=True)
        self.beats = 0

    def start(self) -> "Heartbeat":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._log.info(self._line_fn())
                self.beats += 1
            except Exception:
                pass

    def stop(self, join: bool = True, timeout: float = 5.0) -> None:
        self._stop.set()
        if join and self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()


def heartbeat_interval_from_env(default: float = DEFAULT_HEARTBEAT_S) -> float:
    raw = os.environ.get("MLFRAME_CRASH_HEARTBEAT_S", "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def start_heartbeat(interval_s: Optional[float] = None) -> Optional[Heartbeat]:
    """Start the process heartbeat once (interval from arg or MLFRAME_CRASH_HEARTBEAT_S; <=0 disables)."""
    global _HEARTBEAT
    if _HEARTBEAT is not None and _HEARTBEAT.alive:
        return _HEARTBEAT
    iv = heartbeat_interval_from_env() if interval_s is None else float(interval_s)
    if iv <= 0:
        return None
    try:
        _HEARTBEAT = Heartbeat(iv).start()
        return _HEARTBEAT
    except Exception as e:
        logger.warning("Could not start heartbeat thread: %s", e)
        return None


def install_crash_diagnostics(crash_dir: Optional[str] = None, all_threads: bool = True, heartbeat_s: Optional[float] = None) -> Dict[str, Any]:
    """Install everything above; returns what got enabled. Never raises."""
    info: Dict[str, Any] = {}
    try:
        info["faulthandler_file"] = open_faulthandler_file(crash_dir, all_threads=all_threads)
        install_exception_hooks()
        register_atexit()
        hb = start_heartbeat(heartbeat_s)
        info["heartbeat_s"] = hb.interval_s if hb else 0
        cs = windows_commit_status()
        if cs:
            logger.info(
                "Memory at startup: physical %.1f GB (avail %.1f), commit limit %.1f GB (avail %.1f), page file ~%.1f GB. "
                "Paging-file exhaustion (WinError 1455) kills the process once commit reaches the limit.",
                cs["phys_total_gb"], cs["phys_avail_gb"], cs["commit_limit_gb"], cs["commit_avail_gb"], cs["pagefile_gb"],
            )
        logger.info(
            "Crash diagnostics: faulthandler file=%s; uncaught exceptions -> log; exit line on normal exit "
            "(its absence means an abrupt kill); heartbeat every %ss.",
            info["faulthandler_file"], info["heartbeat_s"] or "off",
        )
    except Exception as e:
        logger.warning("install_crash_diagnostics partially failed: %s", e)
    return info
