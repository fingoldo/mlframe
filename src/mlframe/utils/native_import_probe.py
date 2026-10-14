"""Import an optional native extension only after proving, out of process, that importing it does not crash.

A broken C extension does not raise ImportError: its DLL faults while loading and takes the whole process down, and
``try/except`` never runs. mlframe hit this with hnswlib on a conda Windows box whose ``python.exe`` directory carried
an MSVC runtime (msvcp140.dll 14.27) older than the one the wheel was built with; since VS 2022 17.10 such binaries
fault on their first ``std::mutex`` (0xC0000005) under an older runtime, so a kNN feature step killed a whole training
run instead of falling back to sklearn.

``native_module_importable(name)`` runs ``import <name>`` in a child interpreter once per process and caches the
verdict. It also exports the verdict through an environment variable, so joblib/loky workers spawned afterwards reuse
it instead of probing again. A module that is already imported is importable by definition and is never probed.
"""

from __future__ import annotations

from typing import Optional, cast

import logging
import os
import subprocess  # nosec B404 - runs sys.executable with a fixed argument list, no shell, no untrusted input
import sys
import threading

logger = logging.getLogger(__name__)

_ENV_PREFIX = "MLFRAME_NATIVE_IMPORT_OK_"
_PROBE_TIMEOUT_S = 120.0
_ACCESS_VIOLATION = {0xC0000005, -1073741819}  # 0xC0000005 is 3221225477 unsigned; -1073741819 is the signed form Windows also reports

_verdicts: dict = {}
_lock = threading.Lock()


def _env_key(name: str) -> str:
    """Environment variable that caches the probe verdict for module ``name``."""
    return _ENV_PREFIX + name.replace(".", "_").upper()


def _describe_failure(returncode: int, stderr: str) -> str:
    """Human-readable reason a probe subprocess failed to import the module."""
    # A Python-level failure always prints a traceback; an interpreter that died without one crashed in native code.
    if returncode in _ACCESS_VIOLATION or returncode in (-11, 139) or "Traceback (most recent call last)" not in (stderr or ""):
        return (
            f"the interpreter crashed while loading it (exit code {returncode}: access violation / segfault). On Windows conda installs this is "
            "usually an MSVC runtime (msvcp140.dll / vcruntime140.dll next to python.exe) older than the one the wheel "
            "was built with; update the environment's vc/vs2015_runtime to 14.40+"
        )
    last = (stderr or "").strip().splitlines()
    return f"import failed with exit code {returncode}" + (f": {last[-1][:300]}" if last else "")


def _probe_child(name: str) -> subprocess.CompletedProcess[str] | None:
    """Run ``import name`` in a child interpreter; None when the child cannot be started at all."""
    try:
        return subprocess.run(  # nosec B603 - fixed trusted executable and argument list
            [sys.executable, "-c", f"import {name}"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # A probe that cannot run says nothing about the module; importing directly is what happened before the probe existed.
        logger.warning("native import probe for %s could not run (%s); importing without it.", name, exc)
        return None


def _cached_verdict(name: str) -> Optional[bool]:
    """The verdict already recorded for ``name`` in this process, or in the environment a parent process exported; None if unknown."""
    with _lock:
        cached = _verdicts.get(name)
        if cached is not None:
            return cast(bool, cached)
        inherited = os.environ.get(_env_key(name))
        if inherited in ("0", "1"):
            _verdicts[name] = inherited == "1"
            return cast(bool, _verdicts[name])
    return None


def _record_verdict(name: str, ok: bool, *, export: bool) -> bool:
    """Store the first verdict recorded for ``name`` and return it, so concurrent probes of one module agree on one answer."""
    with _lock:
        if name in _verdicts:
            return cast(bool, _verdicts[name])
        _verdicts[name] = ok
        if export:
            os.environ[_env_key(name)] = "1" if ok else "0"
    return ok


def native_module_importable(name: str) -> bool:
    """True when ``import name`` succeeds in a child interpreter (or ``name`` is already imported here)."""
    if name in sys.modules:
        return True
    cached = _cached_verdict(name)
    if cached is not None:
        return cached
    # The child interpreter can take seconds, so it runs outside the lock. Two threads may probe the same name at once;
    # whichever verdict is recorded first is the one every caller gets.
    proc = _probe_child(name)
    if proc is None:
        return _record_verdict(name, True, export=False)
    ok = proc.returncode == 0
    if not ok and "No module named" not in (proc.stderr or ""):
        logger.warning("Optional native module %r is unusable in this environment: %s. Falling back to the alternative backend.",
                       name, _describe_failure(proc.returncode, proc.stderr))
    return _record_verdict(name, ok, export=True)


__all__ = ["native_module_importable"]
