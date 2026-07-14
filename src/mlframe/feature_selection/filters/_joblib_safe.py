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

import logging
import sys
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_BIG_STACK_BYTES = 8 * 1024 * 1024
_NEEDS_BIG_STACK = sys.platform.startswith("win")


def disable_cuda_in_worker() -> None:  # pragma: no cover - runs in a loky child process
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


# Shared by the pre-warm below AND run_polynom_pair_fe's real LokyBackend construction (must match: loky's
# get_reusable_executor pool-reuse key includes this value, and a pre-warmed pool that's already been torn
# down before the real dispatch runs is worse than not pre-warming at all).
#
# 2026-07-11: was 900s in an earlier version of this fix, sized against the gap measured BEFORE
# maybe_prewarm_polynom_loky_pool's call site moved from fit-entry to right-after-categorization (~418s).
# After that move the real gap shrank to ~236s at 100k rows, but 900s was never revisited -- a full
# production A/B (round 13) showed idle workers lingering for up to 900s past their one use measurably
# contended with the REST of the fit (second target's categorization/training etc.): wall-clock was still
# 862s vs a 712s no-prewarm baseline, despite the polynom-pair-FE phase itself genuinely being faster
# (125.4s -> 117.4s). Reduced to 450s -- comfortably above the measured 236s gap (accounting for the gap
# likely growing at 1M/3M scale, where GPU pair-MI screening between categorization and this dispatch takes
# proportionally longer) while roughly halving how long a pool can linger unused afterward.
POLYNOM_LOKY_IDLE_WORKER_TIMEOUT = 450


def maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters: int, n_jobs: int) -> Optional[threading.Thread]:
    """Speculatively pre-warms the polynom-pair-FE loky pool on a daemon thread, IFF that phase will actually
    dispatch to it (``fe_smart_polynom_iters`` truthy and ``n_jobs`` > 1). Returns the started thread (mainly for
    tests to join on), or ``None`` when the gate isn't met.

    2026-07-11 perf: call this from ``MRMR._fit_impl`` right AFTER categorization completes, not at fit-entry.
    Isolated A/B on real (79237, 544) production shape, TRIVIAL per-task work: a COLD ``LokyBackend`` pool (16
    fresh worker processes each re-importing mlframe/numba/cupy) took 28.1s; the SAME pool warm took 0.7s.
    IMPORTANT CALIBRATION (round-13 production A/B): that 28s/0.7s gap does NOT translate directly to
    wall-clock saved on the REAL dispatch -- the real polynom-pair-FE phase's ~110-160s is dominated by actual
    Optuna/CMA-ES trial work (100 trials x 5 restarts x ~11 hard pairs), so pool cold-start is a SMALL fraction
    of it. Measured real effect of a correctly-warmed pool on the phase itself: 125.4s (cold, no pre-warm) ->
    117.4s (warm) -- an ~8s win, not ~27s. Still worth taking (free -- runs on a daemon thread while GPU work
    happens), but do not expect the isolated microbenchmark's ratio to hold end-to-end.

    THREE bugs found and fixed across two production A/B rounds before this pre-warm's net effect went from
    negative to (modestly) positive:
    (1) [round 12] An earlier version called this at fit-ENTRY (before categorization) -- categorization is
    itself CPU-active, not idle, so pre-warming there made the categorization-to-GPU-screen gap grow 85.3s ->
    153.1s (16 new "Grid size 1" GPU warnings appeared in that window, matching the 16 concurrently-spawning
    workers contending for CPU). Result: -223s wall-clock regression. Fixed by moving the call site to right
    after categorization, where the next phase (GPU pair-MI screening) is genuinely GPU-bound (blocks on
    ``.get()``/``copy_to_host``) and leaves CPU actually free.
    (2) [round 12] The gap from that (still too-early) call site to ``run_polynom_pair_fe``'s actual dispatch
    measured ~418s -- longer than loky's default ``idle_worker_timeout=300``, so the pre-warmed pool
    self-terminated before it was ever used, paying the contention cost above for zero benefit.
    (3) [round 13] Fixing (1) shrank the real gap to ~236s, but the timeout from fix (2) was left at a
    generously-oversized 900s (sized against the STALE 418s figure) -- idle workers then lingered for up to
    900s past their one use, contending with the REST of the fit (second target's own categorization/training
    etc.). Wall-clock was STILL 862s vs a 712s no-prewarm baseline despite the polynom-pair-FE phase itself
    genuinely being faster (125.4s -> 117.4s) -- the ~8s phase-level win was swamped by lingering-pool
    contention elsewhere. Fixed by reducing ``POLYNOM_LOKY_IDLE_WORKER_TIMEOUT`` to 450s (comfortably above the
    measured 236s gap, accounting for it likely growing at 1M/3M scale, while roughly halving how long an
    unused pool can linger). Round-14 validates whether this closes the remaining gap.

    Matches the EXACT ``LokyBackend(inner_max_num_threads=1, initializer=disable_cuda_in_worker,
    idle_worker_timeout=POLYNOM_LOKY_IDLE_WORKER_TIMEOUT)`` construction ``run_polynom_pair_fe`` uses so loky's
    ``get_reusable_executor`` reuse-key (initializer function identity + worker count + timeout) hits and the
    real dispatch reuses this pool. Best-effort: any failure inside the background thread is silently
    swallowed -- the real dispatch still works, just cold.
    """
    if not fe_smart_polynom_iters or not n_jobs or n_jobs <= 1:
        return None

    def _prewarm(_n_jobs: int = int(n_jobs)) -> None:
        """Runs a trivial dispatch through the polynom-pair-FE loky pool shape to force worker spawn now."""
        _t0 = time.perf_counter()
        try:
            from joblib import Parallel, delayed
            from joblib._parallel_backends import LokyBackend

            _backend = LokyBackend(
                inner_max_num_threads=1,
                initializer=disable_cuda_in_worker,
                idle_worker_timeout=POLYNOM_LOKY_IDLE_WORKER_TIMEOUT,
            )
            Parallel(n_jobs=_n_jobs, backend=_backend)(delayed(int)(1) for _ in range(_n_jobs))
            logger.info("mrmr-polynom-loky-prewarm: pool warm in %.1fs (n_jobs=%d).", time.perf_counter() - _t0, _n_jobs)
        except Exception as _e:
            logger.warning(
                "mrmr-polynom-loky-prewarm: failed after %.1fs (%s: %s) -- run_polynom_pair_fe will pay a cold start.",
                time.perf_counter() - _t0, type(_e).__name__, _e,
            )

    thread = threading.Thread(target=_prewarm, daemon=True, name="mrmr-polynom-loky-prewarm")
    thread.start()
    return thread


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


# ---------------------------------------------------------------------------
# Fit-constant memmap cache for loky-shipped big arrays
# ---------------------------------------------------------------------------

# joblib's memmapping reducer dumps every over-``max_nbytes`` ndarray argument to a fresh temp file on
# EVERY ``Parallel(...)`` invocation -- for the fit-constant ``data`` matrix shipped to the FE loky pools
# that meant re-writing the same ~200-400MB buffer to disk once per pool call (wellbore-100k GPU-strict
# profile: 45 memmap dumps, ~315s of _pickle.dumps). An ndarray that is ALREADY a np.memmap is instead
# passed to workers by FILENAME (no dump at all), so: dump each fit-constant array ONCE per process
# (content-keyed), hand back the read-only memmap view, and let every subsequent Parallel call ship it
# for free. Keyed by (shape, dtype, sampled content hash) -- cheap, and a genuinely different array
# (next fit) gets its own entry. Files live in joblib's own temp folder convention and are freed on
# process exit (best-effort unlink; Windows may defer until handles close).
_FIT_MEMMAP_CACHE: dict = {}


def _fit_constant_key(arr) -> tuple:
    """Cheap content key: shape + dtype + blake2 of a bounded sample of the buffer (first/last 64KB +
    strided middle) -- collision-safe enough for the per-fit cache while staying O(1) in array size."""
    import hashlib

    import numpy as _np

    a = _np.ascontiguousarray(arr)
    raw = a.view(_np.uint8).ravel()
    h = hashlib.blake2b(digest_size=16)
    h.update(raw[:65536].tobytes())
    if raw.size > 65536:
        h.update(raw[-65536:].tobytes())
        h.update(raw[:: max(1, raw.size // 65536)].tobytes())
    return (a.shape, str(a.dtype), h.hexdigest())


def fit_constant_memmap(arr):
    """Return a READ-ONLY ``np.memmap`` view of ``arr``, dumped to disk at most once per process per
    content (see module note above). Falls back to the original array on any failure -- callers lose
    only the dump-dedup, never correctness. The view is read-only ('r') so a worker bug can never
    corrupt the shared file."""
    import numpy as _np

    try:
        if isinstance(arr, _np.memmap):
            return arr
        key = _fit_constant_key(arr)
        cached = _FIT_MEMMAP_CACHE.get(key)
        if cached is not None:
            return cached
        import os
        import tempfile

        fd, path = tempfile.mkstemp(prefix="mlframe_fitconst_", suffix=".mmap")
        os.close(fd)
        a = _np.ascontiguousarray(arr)
        mm = _np.memmap(path, dtype=a.dtype, mode="w+", shape=a.shape)
        mm[...] = a
        mm.flush()
        ro = _np.memmap(path, dtype=a.dtype, mode="r", shape=a.shape)
        _FIT_MEMMAP_CACHE[key] = ro
        return ro
    except Exception:
        return arr
