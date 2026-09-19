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
    return _ENV_PREFIX + name.replace(".", "_").upper()


def _describe_failure(returncode: int, stderr: str) -> str:
    # A Python-level failure always prints a traceback; an interpreter that died without one crashed in native code.
    if returncode in _ACCESS_VIOLATION or returncode in (-11, 139) or "Traceback (most recent call last)" not in (stderr or ""):
        return (
            f"the interpreter crashed while loading it (exit code {returncode}: access violation / segfault). On Windows conda installs this is "
            "usually an MSVC runtime (msvcp140.dll / vcruntime140.dll next to python.exe) older than the one the wheel "
            "was built with; update the environment's vc/vs2015_runtime to 14.40+"
        )
    last = (stderr or "").strip().splitlines()
    return f"import failed with exit code {returncode}" + (f": {last[-1][:300]}" if last else "")


def native_module_importable(name: str) -> bool:
    """True when ``import name`` succeeds in a child interpreter (or ``name`` is already imported here)."""
    if name in sys.modules:
        return True
    cached = _verdicts.get(name)
    if cached is not None:
        return cached
    with _lock:
        cached = _verdicts.get(name)
        if cached is not None:
            return cached
        inherited = os.environ.get(_env_key(name))
        if inherited in ("0", "1"):
            _verdicts[name] = inherited == "1"
            return _verdicts[name]
        try:
            proc = subprocess.run(  # nosec B603 - fixed trusted executable and argument list
                [sys.executable, "-c", f"import {name}"],
                capture_output=True,
                text=True,
                timeout=_PROBE_TIMEOUT_S,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            # A probe that cannot run says nothing about the module; importing directly is what happened before the probe existed.
            logger.warning("native import probe for %s could not run (%s); importing without it.", name, exc)
            _verdicts[name] = True
            return True
        ok = proc.returncode == 0
        if not ok and "No module named" not in (proc.stderr or ""):
            logger.warning("Optional native module %r is unusable in this environment: %s. Falling back to the alternative backend.",
                           name, _describe_failure(proc.returncode, proc.stderr))
        _verdicts[name] = ok
        os.environ[_env_key(name)] = "1" if ok else "0"
        return ok


__all__ = ["native_module_importable"]
