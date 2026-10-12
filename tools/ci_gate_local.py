"""Run the whole-project mypy gate the way CI does, and print the environment it ran in.

CI and a workstation can disagree on the same commit (2026-10-12: CI reported no issues on 1897 files, a local run reported
135 errors). This script makes the comparison possible: it prints a fingerprint of the installed packages and the mypy
version, and runs the gate with ``--no-incremental`` so a stale cache cannot hide an error.

Usage: ``python tools/ci_gate_local.py``. Exit code is the gate's exit code.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys

STUB_HINTS = ("stubs", "types-", "cupy", "numba")


def _freeze() -> list[str]:
    """Sorted ``pip freeze`` lines of the running interpreter."""
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False).stdout
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def main() -> int:
    """Print the fingerprint, then run the gate."""
    freeze = _freeze()
    digest = hashlib.sha256("\n".join(freeze).encode("utf-8")).hexdigest()[:16]
    print(f"python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
    print(f"pip-freeze sha256[:16] {digest} ({len(freeze)} packages)")
    for line in freeze:
        if any(h in line.lower() for h in STUB_HINTS):
            print(f"  {line}")
    cmd = [
        sys.executable, "-m", "py_ci_shared.mypy_gate", "--min-files", "1600", "src/mlframe",
        "--cache-dir=.mypy_cache_ci_local",
    ]
    print("gate:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
