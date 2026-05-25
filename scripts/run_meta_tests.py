#!/usr/bin/env python3
"""Wrapper for the mlframe-meta-tests pre-commit hook.

Meta-tests are AST / config / structure checks and never execute numba kernels.
On CUDA-equipped dev machines, the conftest auto-prewarm path can hang on
numba.cuda driver init (5+ minute lock-up observed in multi-agent sessions).
Setting NUMBA_DISABLE_JIT=1 and MLFRAME_SKIP_NUMBA_PREWARM=1 BEFORE pytest
imports anything keeps the hook fast and reliable on every contributor box.

CI workflows (.github/workflows/ci.yml + sklearn-matrix-ci.yml) leave the
env unset so the full numba prewarm + JIT path fires there.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MLFRAME_SKIP_NUMBA_PREWARM", "1")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_meta/",
        "--no-cov",
        "-p",
        "no:randomly",
        "-q",
        "-x",
        *sys.argv[1:],
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
