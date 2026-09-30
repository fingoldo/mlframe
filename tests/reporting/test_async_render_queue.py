"""Unit tests for ``ReportRenderQueue`` / ``SharedArrayPool``: zero-copy hand-off, back-pressure, failure surfacing, ordering, cleanup."""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time

import numpy as np
import pytest

from mlframe.reporting._async_render import (
    ReportRenderQueue,
    SharedArrayPool,
    _shutdown_live_queues,
    attach_array,
    detach_array,
)


def _shm_segments() -> set:
    """Names of THIS process's shared-memory segments in /dev/shm (another pytest process may own segments of its own at the same time)."""
    try:
        return {n for n in os.listdir("/dev/shm") if n.startswith(f"mlframe_rq_{os.getpid()}_")}
    except FileNotFoundError:
        return set()


def _sum_task(arr, scale=1):
    """Module-level (picklable) task: sums its array argument."""
    return float(arr.sum()) * scale


def _array_facts(arr):
    """Report, from inside the worker, whether the array it received owns its data and is writeable."""
    return bool(arr.flags.owndata), bool(arr.flags.writeable), float(arr[0]), float(arr[-1])


def _boom(arr):
    """Always raises, to exercise failure surfacing."""
    raise ValueError("kaboom from worker")


def _same_memory(arr, other_id):
    """Thread-backend task: report whether the received array aliases the caller's buffer (by data pointer)."""
    return arr.__array_interface__["data"][0] == other_id, bool(arr.flags.writeable)


def test_shared_array_round_trip_is_zero_copy():
    """An attached view reads the owner's segment directly: a write through the owner view is visible through the attached view."""
    pool = SharedArrayPool()
    try:
        src = np.arange(1000, dtype=np.float64)
        desc = pool.share(src)
        view = attach_array(desc)
        try:
            assert view.flags.writeable is False
            assert view.flags.owndata is False
            assert np.array_equal(view, src)
            pool.owner_view(desc)[0] = 123.0
            assert view[0] == 123.0, "attached view must alias the segment, not hold a copy"
        finally:
            del view
            detach_array(desc)
        # the descriptor is what crosses the process boundary -- it must not carry the data
        assert len(pickle.dumps(desc)) < 300
    finally:
        pool.close()
    assert pool.live_segments == 0


def test_shared_array_pool_dedups_one_array_and_unlinks_at_zero_refs():
    """The same buffer submitted twice lands in ONE segment; the segment disappears when the last reference is released."""
    pool = SharedArrayPool()
    src = np.linspace(0, 1, 5000)
    d1 = pool.share(src)
    d2 = pool.share(src)
    assert d1.name == d2.name and pool.live_segments == 1
    pool.release(d1)
    assert pool.live_segments == 1
    pool.release(d2)
    assert pool.live_segments == 0
    assert d1.name not in _shm_segments()
    pool.close()


def test_shared_array_pool_rejects_object_dtype():
    """Object arrays cannot live in shared memory; the pool says so instead of copying pointers."""
    pool = SharedArrayPool()
    with pytest.raises(TypeError):
        pool.share(np.array(["a", None], dtype=object))
    pool.close()


def test_thread_backend_passes_large_arrays_by_reference_as_read_only_views():
    """Above the snapshot cap a thread worker sees the caller's buffer (no copy) through a read-only view; the caller's array stays writeable."""
    arr = np.arange(200_000, dtype=np.float64)  # 1.6 MB
    with ReportRenderQueue(backend="thread", workers=1, snapshot_max_mb=1.0) as q:
        fut = q.submit(_same_memory, arr, arr.__array_interface__["data"][0], name="alias")
        aliased, writeable = fut.result(timeout=30)
    assert aliased and not writeable
    assert arr.flags.writeable


def test_thread_backend_snapshots_small_arrays_so_later_mutation_is_not_seen():
    """A small array is copied at submit: the worker sees the values as submitted even if the caller overwrites its buffer while the task is queued."""
    release = threading.Event()

    def _read_after(arr):
        """Waits until the caller has overwritten its buffer, then reports the first element it sees."""
        release.wait(timeout=30)
        return float(arr[0]), bool(arr.flags.writeable)

    arr = np.arange(1000, dtype=np.float64)
    q = ReportRenderQueue(backend="thread", workers=1)
    try:
        fut = q.submit(_read_after, arr, name="snap")
        arr[:] = -1.0
        release.set()
        seen, writeable = fut.result(timeout=30)
    finally:
        release.set()
        q.close()
    assert seen == 0.0 and not writeable


def test_process_backend_hands_large_arrays_over_through_shared_memory():
    """A process worker receives a read-only non-owning view (zero-copy attach), the result matches, and no segment survives close."""
    before = _shm_segments()
    arr = np.arange(400_000, dtype=np.float64)
    with ReportRenderQueue(backend="process", workers=1) as q:
        f_sum = q.submit(_sum_task, arr, scale=2, name="sum")
        f_facts = q.submit(_array_facts, arr, name="facts")
        assert f_sum.result(timeout=120) == pytest.approx(float(arr.sum()) * 2)
        owndata, writeable, first, last = f_facts.result(timeout=120)
        assert owndata is False and writeable is False
        assert (first, last) == (0.0, 399_999.0)
        # both tasks referenced the same buffer -> one segment while they were in flight
    assert _shm_segments() == before


def _current_nice():
    """Worker-side: the niceness this process currently runs at."""
    return os.nice(0)


@pytest.mark.skipif(not hasattr(os, "nice"), reason="os.nice is POSIX only")
def test_process_workers_run_at_lowered_priority_so_they_yield_cores_to_training():
    """Drawing is the lowest-priority work: process workers start niced by ``process_nice`` and the parent keeps its own priority."""
    parent = os.nice(0)
    with ReportRenderQueue(backend="process", workers=1, process_nice=7) as q:
        worker_nice = q.submit(_current_nice, name="nice").result(timeout=120)
    assert worker_nice >= parent + 7
    assert os.nice(0) == parent


def test_process_backend_small_arrays_are_pickled_not_shared():
    """Arrays under ``shm_min_bytes`` ride in the pickle; no segment is allocated for them."""
    small = np.arange(10, dtype=np.float64)
    q = ReportRenderQueue(backend="process", workers=1, shm_min_bytes=1 << 20)
    try:
        assert q.submit(_sum_task, small, name="small").result(timeout=120) == 45.0
        assert q._pool is not None and q._pool.live_segments == 0
    finally:
        q.close()


def test_failed_task_is_logged_with_its_name_recorded_and_does_not_stop_the_others(caplog):
    """A worker exception becomes a WARNING naming the artifact plus an entry in the summary; sibling tasks still complete."""
    arr = np.ones(10)
    with caplog.at_level(logging.WARNING, logger="mlframe.reporting._async_render"):
        q = ReportRenderQueue(backend="thread", workers=2)
        ok = [q.submit(_sum_task, arr, name=f"ok{i}") for i in range(3)]
        bad = q.submit(_boom, arr, name="val_calibration.png")
        summary = q.close()
    assert [f.result() for f in ok] == [10.0, 10.0, 10.0]
    assert isinstance(bad.exception(), ValueError)
    assert summary.completed == 3 and summary.failed == 1
    assert summary.failures[0].name == "val_calibration.png" and "kaboom" in summary.failures[0].error
    assert "kaboom" in summary.failures[0].traceback
    assert any("val_calibration.png" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_process_worker_exception_is_surfaced_not_lost():
    """An exception raised in a spawned worker reaches the summary the same way, and its segments are released."""
    before = _shm_segments()
    arr = np.ones(300_000)
    q = ReportRenderQueue(backend="process", workers=1)
    q.submit(_boom, arr, name="remote_bad")
    summary = q.close()
    assert summary.failed == 1 and summary.failures[0].name == "remote_bad" and "kaboom" in summary.failures[0].error
    assert _shm_segments() == before


def test_unpicklable_task_is_reported_as_a_failure_not_raised():
    """A lambda cannot cross a process boundary; submit reports it as a failed artifact instead of crashing the caller."""
    q = ReportRenderQueue(backend="process", workers=1)
    fut = q.submit(lambda: 1, name="lambda_task")
    summary = q.close()
    assert fut.exception() is not None
    assert summary.failed == 1 and summary.failures[0].name == "lambda_task"


def test_back_pressure_blocks_submit_until_pending_bytes_drop():
    """With a tiny byte cap, the second submit blocks while the first is still running and proceeds once it finishes."""
    gate = threading.Event()

    def _hold(arr):
        """Blocks until the test releases it."""
        gate.wait(timeout=30)
        return arr.nbytes

    big = np.zeros(400_000)  # 3.2 MB
    q = ReportRenderQueue(backend="thread", workers=1, max_pending_mb=4.0)
    try:
        first = q.submit(_hold, big, name="first")
        done = threading.Event()

        def _second():
            """Submit from a helper thread so the test can observe that it blocks."""
            q.submit(_hold, big, name="second")
            done.set()

        t = threading.Thread(target=_second, daemon=True)
        t.start()
        assert not done.wait(timeout=0.5), "second submit must block while 3.2MB is pending under a 4MB cap"
        gate.set()
        assert done.wait(timeout=30), "second submit must proceed once the first finished"
        assert first.result(timeout=30) == big.nbytes
        t.join(timeout=30)
    finally:
        gate.set()
        q.close()


def test_one_oversize_task_is_still_admitted():
    """A single task bigger than the whole byte cap must run (back-pressure never deadlocks on it)."""
    big = np.zeros(1_000_000)
    with ReportRenderQueue(backend="thread", workers=1, max_pending_mb=1.0) as q:
        assert q.submit(_sum_task, big, name="huge").result(timeout=30) == 0.0


def test_after_pending_runs_only_once_earlier_tasks_are_done():
    """A barrier task observes the side effects of everything submitted before it, even with several workers."""
    order: list = []

    def _slow(tag):
        """Appends ``tag`` after a pause so a missing barrier would be visible."""
        time.sleep(0.3)
        order.append(tag)

    def _barrier():
        """Records what was done when the barrier ran."""
        return list(order)

    with ReportRenderQueue(backend="thread", workers=3) as q:
        for i in range(3):
            q.submit(_slow, i, name=f"slow{i}")
        seen = q.submit(_barrier, name="barrier", after_pending=True).result(timeout=30)
    assert sorted(seen) == [0, 1, 2]


def test_on_done_hook_runs_on_the_joining_thread():
    """Completion hooks run in ``join`` on the caller's thread (so they may touch caller-owned dicts), never on a worker."""
    main = threading.get_ident()
    seen: list = []
    q = ReportRenderQueue(backend="thread", workers=1)
    q.submit(_sum_task, np.ones(4), name="h", on_done=lambda f: seen.append((threading.get_ident(), f.result())))
    assert seen == []  # nothing runs before join
    q.join()
    q.close()
    assert seen == [(main, 4.0)]


def test_close_is_idempotent_and_submit_after_close_reports_failure():
    """Closing twice is harmless; a submit after close returns a failed future rather than raising."""
    q = ReportRenderQueue(backend="thread", workers=1)
    q.close()
    q.close()
    fut = q.submit(_sum_task, np.ones(2), name="late")
    assert fut.exception() is not None


def test_summary_accounts_overlap():
    """``overlapped_seconds`` is worker time minus the time the submitter had to wait, so a fully hidden task counts as overlapped."""
    q = ReportRenderQueue(backend="thread", workers=1)
    q.submit(time.sleep, 0.3, name="sleepy")
    time.sleep(0.5)  # the caller is busy elsewhere while the task runs
    summary = q.close()
    assert summary.completed == 1 and summary.busy_seconds >= 0.25
    assert summary.overlapped_seconds >= 0.2


def test_interpreter_exit_hook_unlinks_segments_of_an_unclosed_queue():
    """The atexit hook closes a queue nobody closed and unlinks its shared memory (no /dev/shm leak after Ctrl-C / crash)."""
    before = _shm_segments()
    q = ReportRenderQueue(backend="process", workers=1, shm_min_bytes=1)
    desc = q._pool.share(np.arange(50_000, dtype=np.float64))  # type: ignore[union-attr]
    assert desc.name in _shm_segments() or not os.path.isdir("/dev/shm")
    _shutdown_live_queues()
    assert _shm_segments() == before


def test_keyboard_interrupt_during_join_still_unlinks_segments():
    """An interrupted ``close`` tears the queue down in ``finally``, so its shared memory does not outlive it."""
    before = _shm_segments()
    q = ReportRenderQueue(backend="process", workers=1, shm_min_bytes=1)
    q._pool.share(np.arange(50_000, dtype=np.float64))  # type: ignore[union-attr]

    def _interrupt(*_a, **_k):
        """Stand-in for Ctrl-C arriving while the suite waits on the queue."""
        raise KeyboardInterrupt

    q.join = _interrupt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        q.close()
    assert _shm_segments() == before


def test_process_workers_do_not_rerun_an_unguarded_main_script(tmp_path):
    """A script with no ``__main__`` guard must not be re-executed by spawned workers (it used to redo the whole job in every child)."""
    import subprocess
    import sys

    script = tmp_path / "unguarded.py"
    script.write_text(
        "import os\n"
        "from mlframe.reporting._async_render import ReportRenderQueue, _worker_noop\n"
        "print('BODY-RUN', os.getpid(), flush=True)\n"
        "q = ReportRenderQueue(backend='process', workers=1)\n"
        "q.submit(_worker_noop, name='noop')\n"
        "s = q.close()\n"
        "print('DONE', s.completed, s.failed, flush=True)\n"
    )
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600)
    lines = proc.stdout.splitlines()
    assert sum(1 for ln in lines if ln.startswith("BODY-RUN")) == 1, proc.stdout + proc.stderr[-2000:]
    assert "DONE 1 0" in lines, proc.stdout + proc.stderr[-2000:]
