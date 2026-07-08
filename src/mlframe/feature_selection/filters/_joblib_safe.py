"""Windows-safe wrappers for numba-heavy joblib loky workers.

On Windows, ``joblib.Parallel(backend="loky")`` spawns worker processes
whose main thread has the OS-default 1MB stack. The first @njit kernel
call inside such a worker loads numba's on-disk JIT cache, which routes
through ``llvmlite.binding.executionengine.finalize_object`` -- a deep
recursive C++ pass that needs ~2-3MB stack to finalize a non-trivial
LLVM module. With only 1MB available the worker crashes with::

    Windows fatal exception: stack overflow
    File "...llvmlite/binding/ffi.py", line 212 in __call__
    File "...llvmlite/binding/executionengine.py", line 99 in finalize_object
    File "...numba/core/codegen.py", line 1071 in wrapper
    ...
    File "...numba/core/codegen.py", line 1169 in unserialize_library

This was observed 2026-05-22 in the polynom-pair FE prod path where
``run_polynom_pair_fe`` dispatches per-pair work via loky.

Linux is unaffected: glibc's default pthread stack is 8MB.

Fix: run the per-task body in a sub-thread whose stack we set via
``threading.stack_size`` to 8MB BEFORE creating the thread. Once the
sub-thread's first njit call finalizes the LLVM module, the compiled
machine code lives in process memory and subsequent calls don't redo
the finalize pass. So the cost is one extra thread create+join per
task (sub-millisecond) regardless of platform.
"""
from __future__ import annotations

import sys
import threading
from typing import Any, Callable

_BIG_STACK_BYTES = 8 * 1024 * 1024
_NEEDS_BIG_STACK = sys.platform.startswith("win")


def disable_cuda_in_worker():  # pragma: no cover - runs in a loky child process
    """loky worker initializer: force the worker process CPU-ONLY.

    Runs once per freshly-spawned loky worker BEFORE the first task unpickles
    ``mlframe`` (and thus before any ``cupy`` import), so the worker sees NO CUDA
    device and every GPU-FE / permutation path in it takes the njit CPU branch --
    i.e. NO ~250 MB cupy CUDA context is created per worker.

    Why this matters (2026-07-05 wellbore diag, 4 GB GTX 1050 Ti): when a phase
    fans work out to N loky workers, each worker that imports cupy grabs its own
    ~200-250 MB CUDA context; on a small card N such contexts fill VRAM and the
    workers then block on GPU allocations while the joblib parent sleep-polls in
    ``_retrieve`` at ~1 core. Killing the GPU per worker keeps every worker on the
    CPU kernels (real process parallelism across all cores) and leaves the card
    free. The parent keeps its own GPU context for other phases -- only the
    workers go CPU-only.

    Shared by the polynom-pair FE loky pool and the FE pair-MI scoring loky pool.
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def run_in_big_stack_thread(
    func: Callable[..., Any],
    *args: Any,
    stack_bytes: int = _BIG_STACK_BYTES,
    **kwargs: Any,
) -> Any:
    """Call ``func(*args, **kwargs)`` in a thread with a larger OS stack.

    Returns whatever ``func`` returns. Re-raises any exception from the
    sub-thread on the caller's thread, preserving the original traceback.

    On non-Windows platforms (where the default thread stack is already
    ~8MB) this is a pass-through to ``func(*args, **kwargs)`` to avoid
    unnecessary thread creation overhead.
    """
    if not _NEEDS_BIG_STACK:
        return func(*args, **kwargs)

    result_holder: list = [None]
    exc_holder: list = [None]

    def _target() -> None:
        """Runs ``func`` on the big-stack worker thread, stashing its result or exception for the caller to
        retrieve/re-raise on the main thread once the worker joins."""
        try:
            result_holder[0] = func(*args, **kwargs)
        except (Exception, KeyboardInterrupt, SystemExit) as e:
            # Thread-target boundary: capture user-fn failures (including
            # KeyboardInterrupt/SystemExit raised inside the worker thread)
            # for re-raise on the main thread. Bare ``BaseException`` would
            # cover the same set but trips the no-bare-except meta-linter;
            # this explicit triple matches the intent without the BLE001
            # smell. GeneratorExit is intentionally NOT caught - it should
            # propagate to the thread's own teardown.
            exc_holder[0] = e

    old_size = threading.stack_size()
    threading.stack_size(stack_bytes)
    try:
        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join()
    finally:
        threading.stack_size(old_size)

    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result_holder[0]
