"""Pre-push step: bring the kernel tunings of THIS machine up to date when a push touches kernel code.

The measured backend/parameter choices of the GPU and CPU kernels are cached per machine and per kernel source version. Editing a kernel (or upgrading a driver or library)
leaves them stale, and a stale cache makes the next run tune in the middle of a fit, or fall back to built-in defaults, or time a benchmark against a background sweep. A
warning nobody reads does not fix that, so by default this step does the work:

* ``mlframe-tune-kernels ensure --if-cuda``: tunes only what is missing or stale (instant when everything is current), within ``MLFRAME_PREPUSH_TUNE_MINUTES`` (default 20).
  It does nothing on a host without CUDA, so CI and GPU-less machines pass straight through.
* one tuning run per machine at a time (a lock), so pushes from several sessions do not pile up sweeps that measure each other;
* a sweep that FAILS blocks the push - a kernel that cannot be tuned is a kernel bug; running out of the time budget does not (the rest is tuned on the next push);
* ``MLFRAME_PREPUSH_TUNE=0`` downgrades the step to a report (``ensure --check --advisory``); ``MLFRAME_SKIP_KERNEL_TUNING_HOOK=1`` skips it.

A sweep on the commit path was removed once (minutes per commit, instances piling up across sessions). This runs on push only, only when kernel code changed, only when something is
stale, under a lock and a time limit.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Mapping, Optional, Sequence

_EXIT_BUDGET_SPENT = 3  # mlframe.system.kernel_tuning_cache._ensure.EXIT_BUDGET_SPENT


def _off(value: str) -> bool:
    """Whether an environment flag value spells 'no'."""
    return value.strip().lower() in ("0", "false", "off", "no")


def main(argv: Optional[Sequence[str]] = None, environ: Optional[Mapping[str, str]] = None, run: Optional[Callable[[list], int]] = None) -> int:
    """Run the pre-push kernel-tuning step; returns the exit code that decides the push."""
    env = os.environ if environ is None else environ
    if env.get("MLFRAME_SKIP_KERNEL_TUNING_HOOK", "").strip() not in ("", "0"):
        return 0
    if run is None:
        from mlframe.system.kernel_tuning_cache import main as run
    if _off(env.get("MLFRAME_PREPUSH_TUNE", "1")):
        return int(run(["ensure", "--if-cuda", "--check", "--advisory"]))
    code = int(run(["ensure", "--if-cuda", "--max-minutes", env.get("MLFRAME_PREPUSH_TUNE_MINUTES", "20")]))
    return 0 if code == _EXIT_BUDGET_SPENT else code


if __name__ == "__main__":
    sys.exit(main())
