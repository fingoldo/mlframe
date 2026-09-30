"""Bounded background queue for report artifacts (charts, diagnostics tables) with zero-copy array hand-off.

A suite spends tens of seconds per model drawing and saving figures that nothing downstream reads back. This module runs
that work off the training thread while the next model trains, with three guarantees the callers rely on:

* every submitted task has finished (or has been reported as failed) when ``join``/``close`` returns -- a worker
  exception is logged as a WARNING naming the artifact and recorded, never raised into training and never lost;
* queued memory is bounded: ``submit`` blocks while the bytes referenced by pending tasks exceed ``max_pending_mb``;
* shared-memory segments are unlinked on ``close``, on interpreter exit and on ``KeyboardInterrupt``.

Two backends, chosen per queue:

``thread``   ndarray arguments up to ``snapshot_max_mb`` are copied read-only at submit (isolating the worker from later in-place
             changes); bigger ones travel by reference as read-only views. Right for rendering that overlaps GIL-releasing native
             training (GBDT fit/predict) and I/O.
``process``  workers are spawned processes; every ndarray argument above ``shm_min_bytes`` is copied ONCE into a
             ``multiprocessing.shared_memory`` segment and handed over as a ``(name, shape, dtype)`` descriptor, which the
             worker turns into a read-only zero-copy view. Right for CPU-heavy drawing that must not contend for the GIL.
             Also a snapshot: later in-place changes to the caller's array are not seen by the worker. Workers are started at
             low priority (``process_nice``) and never import the caller's ``__main__`` (no ``if __name__ == "__main__"`` guard is
             needed), so submitted callables must be importable from a module rather than defined in the main script.

Ordering: tasks run in submission order per worker slot but finish in any order; ``after_pending=True`` holds a task back
until everything submitted before it is done (used by the combined-HTML index, which reads the files earlier tasks write).
"""

from __future__ import annotations

import atexit
import logging
import os
import queue
import sys
import threading
import time
import traceback
import weakref
from contextlib import contextmanager
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from mlframe.utils.log_throttle import log_throttle
from mlframe.reporting._shared_arrays import SharedArrayDescriptor, SharedArrayPool, attach_array, detach_array, readonly_view

logger = logging.getLogger(__name__)

DEFAULT_SHM_MIN_BYTES = 64 * 1024


def default_worker_count() -> int:
    """Workers for the render queue: a quarter of the physical cores, at least one, so training keeps the rest."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or 0
    except Exception:
        physical = 0
    if physical <= 0:
        physical = max(1, (os.cpu_count() or 2) // 2)
    return max(1, physical // 4)


def physical_core_count() -> int:
    """Physical core count (falls back to half the logical count when psutil cannot tell)."""
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or 0
    except Exception:
        physical = 0
    return physical if physical > 0 else max(1, (os.cpu_count() or 2) // 2)


@dataclass
class RenderFailure:
    """One artifact that did not get written, with the exception text that explains why."""

    name: str
    error: str
    traceback: str = ""


@dataclass
class RenderSummary:
    """What a queue did over its lifetime: counts, failures and how much of the work was hidden behind other work."""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    failures: List[RenderFailure] = field(default_factory=list)
    busy_seconds: float = 0.0
    blocked_seconds: float = 0.0
    wall_seconds: float = 0.0

    @property
    def overlapped_seconds(self) -> float:
        """Task seconds that ran while the submitting thread was doing something else (busy minus time it had to wait)."""
        return max(0.0, self.busy_seconds - self.blocked_seconds)

    def line(self) -> str:
        """One-line summary for the suite log."""
        text = (
            f"render done: {self.completed} artifacts in {self.busy_seconds:.1f}s of worker time "
            f"(overlapped {self.overlapped_seconds:.1f}s, blocked the caller {self.blocked_seconds:.1f}s, {self.failed} failed)"
        )
        if self.failures:
            shown = ", ".join(f.name for f in self.failures[:5])
            text += f"; NOT written: {shown}" + (f" and {len(self.failures) - 5} more" if len(self.failures) > 5 else "")
        return text


class _ByteGate:
    """Back-pressure: admits a task only while pending bytes/tasks stay under the caps. One oversize task is always admitted."""

    def __init__(self, max_bytes: int, max_tasks: int) -> None:
        self._cv = threading.Condition()
        self._max_bytes = max(0, int(max_bytes))
        self._max_tasks = max(1, int(max_tasks))
        self._bytes = 0
        self._tasks = 0

    def acquire(self, nbytes: int) -> float:
        """Block until the task fits; returns the seconds spent waiting."""
        t0 = time.perf_counter()
        with self._cv:
            while self._tasks > 0 and (self._tasks >= self._max_tasks or (self._max_bytes > 0 and self._bytes + nbytes > self._max_bytes)):
                self._cv.wait(timeout=1.0)
            self._bytes += nbytes
            self._tasks += 1
        return time.perf_counter() - t0

    def __getstate__(self) -> Dict[str, Any]:
        """Condition variables cannot cross a pickle; say so plainly instead of failing inside ``threading``."""
        raise TypeError("_ByteGate holds a live condition variable and cannot be pickled")

    def release(self, nbytes: int) -> None:
        """Return a finished task's share and wake blocked submitters."""
        with self._cv:
            self._bytes -= nbytes
            self._tasks -= 1
            self._cv.notify_all()


_LIVE_QUEUES: "weakref.WeakSet[ReportRenderQueue]" = weakref.WeakSet()


def _shutdown_live_queues() -> None:
    """atexit hook: stop every queue that was never closed and unlink its segments."""
    for q in list(_LIVE_QUEUES):
        try:
            q.close(wait=False)
        except Exception:  # noqa: PERF203 -- per-queue fault isolation; nosec B110 - interpreter is exiting, nothing left to report to
            pass


atexit.register(_shutdown_live_queues)


def _thread_entry(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Tuple[Any, float]:
    """Run one task on a worker thread and time it."""
    t0 = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, time.perf_counter() - t0


def _process_entry(fn: Callable[..., Any], args: tuple, kwargs: dict, descs: List[SharedArrayDescriptor]) -> Tuple[Any, float]:
    """Worker-process entry: map shared arrays, call ``fn``, unmap. Descriptors inside ``args``/``kwargs`` become views."""
    t0 = time.perf_counter()
    attached: List[SharedArrayDescriptor] = []
    try:

        def _resolve(v: Any) -> Any:
            """Map a descriptor to a read-only view (remembering it for detach); pass anything else through."""
            if isinstance(v, SharedArrayDescriptor):
                attached.append(v)
                return attach_array(v)
            return v

        value = fn(*[_resolve(a) for a in args], **{k: _resolve(v) for k, v in kwargs.items()})
    finally:
        for d in attached:
            detach_array(d)
    return value, time.perf_counter() - t0


def _process_initializer(state: Dict[str, Any]) -> None:
    """Make a spawned worker render like the parent: headless matplotlib with the parent's rcParams and plotly template."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        rc = state.get("rcparams")
        if rc:
            matplotlib.rcParams.update(rc)
    except Exception:  # nosec B110 - matplotlib optional in the worker
        logger.debug("worker matplotlib setup failed", exc_info=True)
    try:
        tpl = state.get("plotly_template")
        if tpl:
            import plotly.io as pio

            pio.templates.default = tpl
    except Exception:  # nosec B110 - plotly optional in the worker
        logger.debug("worker plotly setup failed", exc_info=True)
    for key, val in (state.get("env") or {}).items():
        os.environ[key] = val
    # Drawing is the lowest-priority work on the host: with the GIL out of the picture (separate process) a niced worker takes idle
    # cores and cedes contended ones to the training threads instead of slowing them down.
    increment = int(state.get("nice") or 0)
    if increment > 0 and hasattr(os, "nice"):
        try:
            os.nice(increment)
        except OSError:  # nosec B110 - lowering priority is best-effort
            logger.debug("worker could not lower its priority", exc_info=True)


def _worker_noop() -> int:
    """Warm-up task: forces the worker to spawn and import mlframe's reporting stack before real work arrives."""
    import mlframe.reporting.renderers  # noqa: F401

    return os.getpid()


def capture_render_state(nice: int = 0) -> Dict[str, Any]:
    """Parent-side matplotlib rcParams / plotly default template snapshot for ``process`` workers (process-wide state does not cross spawn)."""
    state: Dict[str, Any] = {"env": {k: v for k, v in os.environ.items() if k.startswith("MLFRAME_PLOT_")}, "nice": nice}
    try:
        import matplotlib

        state["rcparams"] = {k: v for k, v in matplotlib.rcParams.items() if k not in ("backend",)}
    except Exception:  # nosec B110 - matplotlib optional
        pass
    try:
        import plotly.io as pio

        state["plotly_template"] = pio.templates.default
    except Exception:  # nosec B110 - plotly optional
        pass
    return state


@contextmanager
def _workers_do_not_reimport_main() -> Iterator[None]:
    """Spawn-start workers without letting them re-execute the caller's ``__main__`` script.

    ``spawn`` re-imports the parent's main module in every child. A script without an ``if __name__ == "__main__":`` guard would then
    re-run the whole suite inside each worker (and Python aborts that with "attempt has been made to start a new process before the
    current process has finished its bootstrapping phase", after the child has already redone the work up to that point). The render
    tasks only reference ``mlframe`` functions, so the workers have no need of ``__main__``; its ``__file__`` / ``__spec__`` are hidden
    for the moment the workers start, which is when ``multiprocessing`` snapshots them.
    """
    main = sys.modules.get("__main__")
    saved = {}
    if main is not None:
        for attr in ("__file__", "__spec__"):
            if attr in vars(main):
                saved[attr] = vars(main)[attr]
                setattr(main, attr, None)
    try:
        yield
    finally:
        for attr, val in saved.items():
            setattr(main, attr, val)


class ReportRenderQueue:
    """Run report-rendering callables on background workers, bounded in memory, with every outcome accounted for.

    Parameters
    ----------
    backend:
        ``"thread"`` or ``"process"`` (see the module docstring).
    workers:
        Worker count; ``None`` uses ``default_worker_count()`` (a quarter of the physical cores, at least one).
    max_pending_mb:
        ``submit`` blocks while the bytes referenced by pending tasks would exceed this (one oversize task is always let through).
    max_pending_tasks:
        Hard cap on queued + running tasks, a second back-pressure axis for floods of tiny tasks.
    shm_min_bytes:
        ``process`` backend: ndarray arguments at least this large go to shared memory instead of being pickled.
    snapshot_max_mb:
        ``thread`` backend: ndarray arguments up to this size are copied at submit (so the caller may keep mutating its buffer);
        bigger ones are passed by reference as read-only views and must not be changed until the task is done.
    process_nice:
        ``process`` backend: niceness increment applied to each worker (POSIX), so drawing uses idle cores without taking them from
        training threads. Threads cannot be niced safely (a descheduled thread holding the GIL would stall the submitter).
    name:
        Prefix for worker thread names and log lines.
    """

    def __init__(
        self,
        backend: str = "thread",
        workers: Optional[int] = None,
        max_pending_mb: float = 512.0,
        max_pending_tasks: int = 256,
        shm_min_bytes: int = DEFAULT_SHM_MIN_BYTES,
        snapshot_max_mb: float = 64.0,
        process_nice: int = 10,
        name: str = "mlframe-render",
    ) -> None:
        if backend not in ("thread", "process"):
            raise ValueError(f"backend must be 'thread' or 'process', got {backend!r}")
        self.backend = backend
        self.workers = int(workers) if workers else default_worker_count()
        self.name = name
        self._shm_min = int(shm_min_bytes)
        self._snapshot_max = int(snapshot_max_mb * 1024 * 1024)
        self._nice = int(process_nice)
        self._gate = _ByteGate(int(max_pending_mb * 1024 * 1024), max_pending_tasks)
        self._pool = SharedArrayPool() if backend == "process" else None
        self._executor: Any = None
        self._exec_lock = threading.Lock()  # the main thread and the dispatcher thread can both be the first to need the executor
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._pending: "set[Future]" = set()
        self._hooks: List[Tuple[Future, Callable[[Future], None]]] = []
        self._summary = RenderSummary()
        self._t_first: Optional[float] = None
        self._closed = False
        self.logged_submitted = 0  # the suite's summary line is printed again only when tasks arrived since the last one
        self._dispatch_q: "queue.SimpleQueue[Optional[Callable[[], None]]]" = queue.SimpleQueue()
        self._dispatcher: Optional[threading.Thread] = None
        _LIVE_QUEUES.add(self)

    def __getstate__(self) -> Dict[str, Any]:
        """A queue owns threads, locks and shared segments; it is process-local and never part of a pickled object."""
        raise TypeError("ReportRenderQueue holds live threads, locks and shared memory and cannot be pickled")

    # ------------------------------------------------------------------ executor / dispatcher
    def _get_executor(self) -> Any:
        """Create the executor lazily so a queue that never receives work costs nothing."""
        with self._exec_lock:
            if self._closed:
                raise RuntimeError("render queue is closed")
            if self._executor is None:
                if self.backend == "thread":
                    self._executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix=self.name)
                else:
                    self._executor = ProcessPoolExecutor(
                        max_workers=self.workers, mp_context=get_context("spawn"), initializer=_process_initializer, initargs=(capture_render_state(self._nice),),
                    )
            return self._executor

    def _ensure_dispatcher(self) -> None:
        """Start the tiny thread that launches tasks held back by ``after_pending`` (never run launches inside executor internals)."""
        if self._dispatcher is None:

            def _loop() -> None:
                """Launch held-back tasks as their predecessors finish, until the ``None`` sentinel arrives."""
                while True:
                    item = self._dispatch_q.get()
                    if item is None:
                        return
                    try:
                        item()
                    except Exception as exc:
                        self._record_internal_failure("render queue dispatcher", exc)

            self._dispatcher = threading.Thread(target=_loop, name=f"{self.name}-dispatch", daemon=True)
            self._dispatcher.start()

    def warm(self) -> None:
        """Process backend: spawn workers and import the reporting stack now, so the first real task does not pay for it."""
        if self.backend != "process" or self._closed:
            return
        ex = self._get_executor()
        for _ in range(self.workers):
            try:
                with _workers_do_not_reimport_main():
                    ex.submit(_worker_noop)
            except Exception:  # noqa: PERF203 -- a failed warm-up submit ends the loop, it is not retried
                logger.debug("render queue warm-up submit failed", exc_info=True)
                return

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        name: str = "",
        after_pending: bool = False,
        on_done: Optional[Callable[[Future], None]] = None,
        **kwargs: Any,
    ) -> Future:
        """Queue ``fn(*args, **kwargs)`` and return a future for its value.

        ``name`` labels the artifact in failure messages. ``after_pending`` starts the task only after everything submitted
        earlier has finished. ``on_done`` runs later on the thread that calls ``join``/``close`` (never on a worker), so it
        may safely touch caller-owned structures such as a metadata dict. ndarray arguments are handed over per the backend.
        The call blocks while the queue is over its byte/task caps.
        """
        label = name if name else getattr(fn, "__name__", "task")
        proxy: Future = Future()
        if self._closed:
            self._fail_now(proxy, label, RuntimeError("render queue is closed"))
            return proxy
        try:
            call_args, call_kwargs, descs, nbytes = self._prepare(args, kwargs)
        except Exception as exc:
            self._fail_now(proxy, label, exc)
            return proxy
        waited = self._gate.acquire(nbytes)
        with self._lock:
            self._summary.submitted += 1
            self._summary.blocked_seconds += waited
            if self._t_first is None:
                self._t_first = time.perf_counter()
            preds = [f for f in self._pending if not f.done()] if after_pending else []
            self._pending.add(proxy)
            if on_done is not None:
                self._hooks.append((proxy, on_done))

        def _launch() -> None:
            """Hand the task to the executor (or settle it as failed if the executor refuses it)."""
            try:
                ex = self._get_executor()
                if self.backend == "thread":
                    inner = ex.submit(_thread_entry, fn, call_args, call_kwargs)
                else:
                    with _workers_do_not_reimport_main():
                        inner = ex.submit(_process_entry, fn, call_args, call_kwargs, descs)
            except Exception as exc:
                self._settle_error(proxy, label, exc, nbytes, descs)
                return
            inner.add_done_callback(lambda f: self._settle(proxy, label, f, nbytes, descs))

        if not preds:
            _launch()
        else:
            self._ensure_dispatcher()
            remaining = [len(preds)]
            lock = threading.Lock()

            def _one_done(_f: Future) -> None:
                """Count a predecessor down; the last one queues the launch on the dispatcher thread."""
                with lock:
                    remaining[0] -= 1
                    last = remaining[0] == 0
                if last:
                    self._dispatch_q.put(_launch)

            for p in preds:
                p.add_done_callback(_one_done)
        return proxy

    def _prepare(self, args: tuple, kwargs: dict) -> Tuple[tuple, dict, List[SharedArrayDescriptor], int]:
        """Turn ndarray arguments into read-only views (thread) or shared-memory descriptors (process); size the task for back-pressure."""
        descs: List[SharedArrayDescriptor] = []
        nbytes = 0

        def conv(v: Any) -> Any:
            """Convert one argument for the backend and add its bytes to the task size."""
            nonlocal nbytes
            if isinstance(v, np.ndarray):
                if self.backend == "thread":
                    nbytes += int(v.nbytes)
                    if v.nbytes <= self._snapshot_max:
                        # Small enough that isolating the worker from later in-place changes costs less than it protects.
                        snap = v.copy()
                        snap.flags.writeable = False
                        return snap
                    return readonly_view(v)
                if v.nbytes >= self._shm_min and not v.dtype.hasobject and self._pool is not None:
                    d = self._pool.share(v)
                    descs.append(d)
                    nbytes += int(v.nbytes)
                    return d
                nbytes += int(v.nbytes)
            return v

        try:
            new_args = tuple(conv(a) for a in args)
            new_kwargs = {k: conv(v) for k, v in kwargs.items()}
        except BaseException:
            if self._pool is not None:
                for d in descs:
                    self._pool.release(d)
            raise
        return new_args, new_kwargs, descs, nbytes

    # ------------------------------------------------------------------ completion
    def _record_internal_failure(self, label: str, exc: BaseException) -> None:
        """Record a failure of the queue's own machinery (dispatcher, completion hook) next to the artifact failures, and log it throttled."""
        with self._lock:
            self._summary.failed += 1
            self._summary.failures.append(RenderFailure(label, f"{type(exc).__name__}: {exc}", "".join(traceback.format_exception(exc))))
        log_throttle(logger, f"async_render_internal::{label}", logging.WARNING, "async render: %s failed: %s", label, exc, exc_info=True)

    def _fail_now(self, proxy: Future, label: str, exc: BaseException) -> None:
        """Settle a task that never reached a worker (closed queue, unpicklable argument) as a recorded failure."""
        with self._lock:
            self._summary.submitted += 1
            self._summary.failed += 1
            self._summary.failures.append(RenderFailure(label, repr(exc), "".join(traceback.format_exception(exc))))
        logger.warning("async render: %s was not queued (%s: %s)", label, type(exc).__name__, exc)
        proxy.set_exception(exc)

    def _settle_error(self, proxy: Future, label: str, exc: BaseException, nbytes: int, descs: List[SharedArrayDescriptor]) -> None:
        """Settle a launch that failed before a worker ran it."""
        self._finish(proxy, label, None, exc, 0.0, nbytes, descs)

    def _settle(self, proxy: Future, label: str, inner: Future, nbytes: int, descs: List[SharedArrayDescriptor]) -> None:
        """Copy an executor future's outcome onto the caller's proxy and account for it."""
        exc = inner.exception() if not inner.cancelled() else RuntimeError("render task was cancelled")
        if exc is None:
            value, secs = inner.result()
            self._finish(proxy, label, value, None, secs, nbytes, descs)
        else:
            self._finish(proxy, label, None, exc, 0.0, nbytes, descs)

    def _finish(
        self, proxy: Future, label: str, value: Any, exc: Optional[BaseException], secs: float, nbytes: int, descs: List[SharedArrayDescriptor]
    ) -> None:
        """Release back-pressure + shared segments, record the outcome, then resolve the proxy and wake ``join``."""
        if self._pool is not None:
            for d in descs:
                self._pool.release(d)
        self._gate.release(nbytes)
        with self._lock:
            self._summary.busy_seconds += secs
            if exc is None:
                self._summary.completed += 1
            else:
                self._summary.failed += 1
                self._summary.failures.append(RenderFailure(label, f"{type(exc).__name__}: {exc}", "".join(traceback.format_exception(exc))))
        if exc is not None:
            logger.warning("async render: %s failed and was not written (%s: %s)", label, type(exc).__name__, exc)
            proxy.set_exception(exc)
        else:
            proxy.set_result(value)
        with self._idle:
            self._pending.discard(proxy)
            self._idle.notify_all()

    # ------------------------------------------------------------------ join / close
    @property
    def submitted(self) -> int:
        """Tasks accepted so far (queued or finished)."""
        with self._lock:
            return self._summary.submitted

    def pending(self) -> int:
        """Tasks submitted and not yet finished."""
        with self._lock:
            return len(self._pending)

    def join(self, timeout: Optional[float] = None) -> RenderSummary:
        """Wait for every pending task, run the ``on_done`` hooks on this thread, and return the running summary."""
        deadline = None if timeout is None else time.monotonic() + timeout
        t0 = time.perf_counter()
        with self._idle:
            while self._pending:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._idle.wait(timeout=remaining if remaining is not None else 1.0)
            hooks, self._hooks = self._hooks, []
            self._summary.blocked_seconds += time.perf_counter() - t0
            if self._t_first is not None:
                self._summary.wall_seconds = time.perf_counter() - self._t_first
        for fut, hook in hooks:
            try:
                hook(fut)
            except Exception as exc:  # noqa: PERF203 -- per-hook fault isolation is intentional
                self._record_internal_failure("completion hook", exc)
        return self.summary()

    def summary(self) -> RenderSummary:
        """Snapshot of the counters so far."""
        with self._lock:
            s = self._summary
            return RenderSummary(
                s.submitted, s.completed, s.failed, list(s.failures), s.busy_seconds, s.blocked_seconds, s.wall_seconds,
            )

    def close(self, wait: bool = True) -> RenderSummary:
        """Finish (or, with ``wait=False``, abandon) outstanding work, stop the workers and unlink shared memory. Idempotent."""
        if wait and not self._closed:
            finished = False
            try:
                self.join()
                finished = True
            finally:
                # Everything has finished on the normal path, so the workers can be reaped synchronously; on an interrupt they
                # are abandoned (cancel_futures) rather than waited on, and the segments are unlinked either way.
                self._teardown(wait=finished)
            return self.summary()
        self._teardown(wait=wait)
        return self.summary()

    def _teardown(self, wait: bool) -> None:
        """Shut the executor and dispatcher down and release every segment."""
        with self._exec_lock:
            self._closed = True
            ex, self._executor = self._executor, None
        if ex is not None:
            try:
                ex.shutdown(wait=wait, cancel_futures=True)
            except Exception:
                logger.debug("render executor shutdown failed", exc_info=True)
        if self._dispatcher is not None:
            self._dispatch_q.put(None)
            self._dispatcher = None
        if self._pool is not None:
            self._pool.close()
        _LIVE_QUEUES.discard(self)

    def __enter__(self) -> "ReportRenderQueue":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close(wait=exc_info[0] is None)
