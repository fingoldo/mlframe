"""Shared gate for PySR-using tests.

Provides ``pysr_works()`` -- True iff Julia is installed AND ``import pysr``
succeeds in a child Python process. Caches the result at module-load time so
the ~10-30s probe runs once per pytest worker, not once per test.

Why subprocess probe (not bare ``import pysr``):
    pysr's transitive import chain (Julia runtime + PythonCall.jl + torch on
    some installs) can native-crash on broken environments, tearing down the
    xdist worker. A subprocess probe contains the blast radius -- a crash in
    the child returns a non-zero exit, the gate returns False, and the test
    file ``pytest.skip``s instead of taking down the session.

Usage pattern (paste at the top of any PySR-using test file):

    import pytest
    from tests._pysr_gate import pysr_works

    pytestmark = [
        pytest.mark.skipif(not pysr_works(), reason="PySR not usable"),
        pytest.mark.slow_only,
    ]

The slow_only marker keeps these out of --fast runs; the subprocess gate
keeps them off hosts where Julia / PythonCall.jl don't load cleanly.
"""
from __future__ import annotations

import os
import subprocess
import sys
from shutil import which

_PROBE_CACHE: "bool | None" = None


def _locate_julia() -> "str | None":
    """Return the path to a julia executable, or None if absent.

    Honours ``$JULIA_EXE`` if it points at an existing file. Otherwise tries
    well-known install roots + PATH. On Windows users typically install to
    D:/Julia/bin; juliaup puts it under ~/.juliaup/bin.
    """
    env_julia = os.environ.get("JULIA_EXE")
    if env_julia and os.path.isfile(env_julia):
        return env_julia
    candidates = [
        ("D:/Julia/bin", "julia.exe"),
        (r"C:\Program Files\Julia\bin", "julia.exe"),
        (os.path.expanduser("~/.juliaup/bin"), "julia"),
        ("/usr/local/bin", "julia"),
        ("/usr/bin", "julia"),
    ]
    for bindir, exe_name in candidates:
        julia_exe = os.path.join(bindir, exe_name)
        if os.path.isfile(julia_exe):
            return julia_exe
    return which("julia") or which("julia.exe")


def pysr_works() -> bool:
    """True iff Julia is installed AND ``import pysr`` succeeds in a child
    process. Cached after the first call.

    Side effects on first successful call: ``JULIA_EXE`` and ``PATH`` are
    populated so subsequent in-process ``import pysr`` finds the runtime.
    """
    global _PROBE_CACHE
    if _PROBE_CACHE is not None:
        return _PROBE_CACHE
    julia = _locate_julia()
    if julia is None:
        _PROBE_CACHE = False
        return False
    bindir = os.path.dirname(julia)
    # Expose JULIA_EXE / PATH process-wide so test bodies that import pysr
    # after the gate passes find the runtime. Idempotent: prepend only if
    # bindir isn't already at the head of PATH.
    os.environ["JULIA_EXE"] = julia
    if not os.environ.get("PATH", "").startswith(bindir):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import pysr"],
            env=os.environ,
            capture_output=True,
            timeout=60,
        )
        _PROBE_CACHE = (r.returncode == 0)
    except (subprocess.TimeoutExpired, OSError):
        _PROBE_CACHE = False
    return _PROBE_CACHE
