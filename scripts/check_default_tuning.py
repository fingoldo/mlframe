"""Pre-commit wrapper: keep the committed default_kernel_tuning.json in sync.

Runs ``mlframe-gen-default-tuning --check`` so a commit fails fast when a kernel
edit changed its code_version (or added/removed a kernel) but the committed
anonymized defaults file was not regenerated.

Why a wrapper (not ``entry: python -m ...`` directly):
  * Sets ``NUMBA_DISABLE_JIT=1`` + ``MLFRAME_SKIP_NUMBA_PREWARM=1`` BEFORE any
    mlframe import, mirroring scripts/run_meta_tests.py. ``discover_tuners``
    imports every mlframe module; on a CUDA-equipped dev box a numba/CUDA-driver
    init during that walk has been observed to hang in multi-agent sessions.
    The defaults *--check* path only reads code_versions + the committed file; it
    never executes a kernel, so disabling JIT costs nothing here.
  * ``PYUTILZ_KERNEL_DISABLE_SWEEP=1`` is a belt-and-braces guard: ``--check``
    uses ``skip_existing`` so it should never sweep, but if a kernel's
    code_version genuinely drifted the commit should FAIL (telling the dev to run
    the generator), NOT silently launch a multi-minute sweep inside ``git
    commit``.

Escape hatch: ``git commit --no-verify`` skips all hooks for a one-off commit.
Regenerate for real with: ``mlframe-gen-default-tuning`` (then ``git add`` the
file), or run the manual fast stage: ``pre-commit run --hook-stage manual
gen-default-tuning``.
"""
import os
import sys

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MLFRAME_SKIP_NUMBA_PREWARM", "1")
os.environ.setdefault("PYUTILZ_KERNEL_DISABLE_SWEEP", "1")

from mlframe.feature_selection._benchmarks.gen_default_tuning import main  # noqa: E402

if __name__ == "__main__":
    # Force --check regardless of extra argv pre-commit may append.
    sys.exit(main(["--check"]))
