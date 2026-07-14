#!/usr/bin/env python3
"""Wrapper for the mlframe-meta-tests pre-commit hook.

Meta-tests are AST / config / structure checks and never execute numba kernels.
On CUDA-equipped dev machines, the conftest auto-prewarm path can hang on
numba.cuda driver init (5+ minute lock-up observed in multi-agent sessions).
Setting NUMBA_DISABLE_JIT=1 and MLFRAME_SKIP_NUMBA_PREWARM=1 BEFORE pytest
imports anything keeps the hook fast and reliable on every contributor box.

``tests/conftest.py`` also does its own CUDA (cuBLAS) warm-up matmul at import
time -- unconditional unless ``CUDA_VISIBLE_DEVICES=""`` -- to keep torch's
cublas handle healthy against a one-way corruption bug on some GPUs (see its
own comment). On this dev box that warm-up itself native-crashes the whole
interpreter (SIGSEGV / Windows STATUS_ACCESS_VIOLATION) before any test runs.
Since meta-tests never touch the GPU, force CPU-only here too.

CI workflows (.github/workflows/ci.yml + sklearn-matrix-ci.yml) leave the
env unset so the full numba prewarm + JIT path fires there.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Run the meta-tests suite CPU-only, with an explicit env dict so ``CUDA_VISIBLE_DEVICES=""`` survives."""
    # Build an explicit env dict rather than mutating os.environ + relying on implicit
    # inheritance: on Windows, subprocess silently DROPS an empty-string env var when the
    # child inherits the parent's os.environ implicitly (CreateProcess's env-block builder
    # treats "NAME=" as absent) -- the child then sees CUDA_VISIBLE_DEVICES as unset rather
    # than "", defeating the CPU-only override below. Passing env= explicitly preserves it.
    env = dict(os.environ)
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("MLFRAME_SKIP_NUMBA_PREWARM", "1")
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
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
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
