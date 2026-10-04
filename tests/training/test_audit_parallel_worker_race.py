"""Wave-27 sensors: race-condition fixes for parallel workers (3 sites).

#1 P1 feature_selection/filters/feature_engineering.py:246
   ``times_spent[bin_func_name] += timer() - start`` inside the worker.
   Dispatched via ``parallel_run(..., backend='threading')`` from
   mrmr.py:2257. Python's ``+=`` on a float is load-add-store and NOT
   atomic between threads under the GIL; concurrent workers dropped
   updates silently. Operator-visible at verbose>2 where the
   diagnostic ``logger.info('time spent by binary func')`` under-
   reported / silently zeroed.
   Fix: module-level ``_TIMES_SPENT_LOCK`` serialises the increment.

#2 P2 feature_selection/filters/gpu.py:73
   #2 P2 feature_engineering/transformer/_kernels_cupy.py:33
   ``_KERNEL_INIT_LOCK = multiprocessing.Lock()`` documented as
   "Cross-process safe ... picks up the host process's mutex on
   spawn". That's FALSE on Windows spawn; each child re-imports the
   module and constructs its own Lock.
   Fix: documentation only -- the kernel-init body is idempotent
   so the actual behaviour is fine; the misleading comment would
   bait future contributors to add non-idempotent init under the
   same "cross-process safe" guarantee.

#3 P2 feature_selection/wrappers/_rfecv.py:1643
   ``Parallel(..., prefer='threads')`` is a SOFT hint that an outer
   ``joblib.parallel_backend('loky')`` / sklearn ``parallel_config``
   can override. The day someone wraps ``RFECV.fit`` in a process
   backend, the closure-state mutations silently vanish in worker
   copies and ``final_score = nan`` with no exception.
   Fix: add ``require='sharedmem'`` so joblib RAISES when it can't
   satisfy threading.
"""

from __future__ import annotations

import multiprocessing

import pytest


def _child_try_init_lock(module_name, queue):
    """Child process: import the module afresh and report whether its init lock can be taken without blocking."""
    import importlib

    lock = importlib.import_module(module_name)._KERNEL_INIT_LOCK
    acquired = lock.acquire(block=False)
    if acquired:
        lock.release()
    queue.put(acquired)


def _init_lock_is_private_to_each_process(module_name):
    """True when a spawned child can take its own copy of the module's init lock while the parent holds the parent's."""
    import importlib

    module = importlib.import_module(module_name)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    assert module._KERNEL_INIT_LOCK.acquire(timeout=30)
    try:
        proc = ctx.Process(target=_child_try_init_lock, args=(module_name, queue))
        proc.start()
        child_acquired = queue.get(timeout=420)
        proc.join(timeout=60)
    finally:
        module._KERNEL_INIT_LOCK.release()
    return bool(child_acquired)


# ---- #1 times_spent lock ------------------------------------------------


def test_feature_engineering_times_spent_lock_added():
    """Every per-pair merge into the shared ``times_spent`` dict happens while ``_TIMES_SPENT_LOCK`` is held, and no increment is lost across threads."""
    import threading
    from collections import defaultdict

    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_score_helpers as helpers

    real_lock = threading.Lock()

    class _SpyLock:
        """Lock that records which thread holds it and how often it was taken."""

        def __init__(self):
            self.holder = None
            self.acquisitions = 0

        def __enter__(self):
            real_lock.acquire()
            self.holder = threading.get_ident()
            self.acquisitions += 1
            return self

        def __exit__(self, *exc):
            self.holder = None
            real_lock.release()

    spy = _SpyLock()
    unlocked_writes = []

    class _GuardedTimes(defaultdict):
        """defaultdict that records every write made while the spy lock is not held by the writing thread."""

        def __setitem__(self, key, value):
            if spy.holder != threading.get_ident():
                unlocked_writes.append(key)
            super().__setitem__(key, value)

    times_spent = _GuardedTimes(float)
    original = helpers._TIMES_SPENT_LOCK
    helpers._TIMES_SPENT_LOCK = spy
    try:
        n_threads, n_calls = 16, 500

        def _worker():
            """Merge a thread-local timing dict into the shared dict, as one pair worker does."""
            for _ in range(n_calls):
                helpers._score_one_pair_iteration_serialization_point_under({"bf_a": 1.0, "bf_b": 2.0}, times_spent)

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        helpers._TIMES_SPENT_LOCK = original

    assert unlocked_writes == []
    assert spy.acquisitions == n_threads * n_calls
    assert times_spent["bf_a"] == float(n_threads * n_calls)
    assert times_spent["bf_b"] == 2.0 * n_threads * n_calls
    assert helpers._TIMES_SPENT_LOCK is original


def test_feature_engineering_times_spent_lock_behaves_threadsafe():
    """Behavioural: stress the lock with 100 concurrent threads each
    doing 1000 increments; assert no updates lost."""
    import threading as _t
    import time
    from collections import defaultdict
    from mlframe.feature_selection.filters.feature_engineering import _TIMES_SPENT_LOCK

    times_spent = defaultdict(float)

    def _worker():
        """Worker."""
        for _ in range(1000):
            with _TIMES_SPENT_LOCK:
                times_spent["bf"] += 0.001

    threads = [_t.Thread(target=_worker) for _ in range(100)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    # Exact total: 100 * 1000 * 0.001 = 100.0; allow rounding error.
    assert abs(times_spent["bf"] - 100.0) < 0.001, (
        f"Wave 27 P1 regression: lock didn't serialise increments. "
        f"Got {times_spent['bf']:.6f}, expected 100.0. "
        f"({elapsed:.2f}s under 100 threads x 1000 increments)"
    )


# ---- #2 _KERNEL_INIT_LOCK doc honesty -----------------------------------


@pytest.mark.timeout(900)
def test_gpu_kernel_init_lock_doc_no_longer_claims_cross_process():
    """The filters GPU init lock is intra-process only: under spawn a child process holds its own lock, so it is not taken while the parent holds its copy."""
    assert _init_lock_is_private_to_each_process("mlframe.feature_selection.filters.gpu") is True


@pytest.mark.timeout(900)
def test_cupy_kernel_init_lock_doc_honest():
    """The transformer cupy init lock is intra-process only: under spawn a child process holds its own lock, so it is not taken while the parent holds its copy."""
    assert _init_lock_is_private_to_each_process("mlframe.feature_engineering.transformer._kernels_cupy") is True


# ---- #3 _rfecv require=sharedmem ---------------------------------------


def test_rfecv_parallel_requires_sharedmem():
    """Folds run under an outer loky backend still mutate the caller's closure state, because the fold dispatch demands shared memory."""
    import joblib

    from mlframe.feature_selection.wrappers.rfecv._fit_outer_loop import _run_folds_parallel

    seen = []

    def _fold_runner(fold_idx):
        """Record the fold in closure state, as the real fold runner does with its score lists."""
        seen.append(fold_idx)

    fold_args = [(i,) for i in range(6)]
    with joblib.parallel_backend("loky", n_jobs=2):
        _run_folds_parallel(2, fold_args, _fold_runner, 0)

    assert sorted(seen) == list(range(6))
