"""Running cells in worker processes, with a timeout that actually stops a cell.

Three requirements pull in different directions and this module is where they meet.

**Processes, not threads.** The kernels underneath these arms are already `njit(parallel=True)`, so
threading them gives roughly nothing measured (about 1.0x at four threads, 1.5x at sixteen) while adding
contention inside each kernel. Four worker processes with four threads each is the shape this machine's
own measurements support, and the thread count is set in the worker's initializer BEFORE numpy is
imported, because every BLAS reads its environment once at import and ignores later changes.

**A timeout has to be enforceable.** `Future.cancel()` does nothing to a task that has already started,
and `Pool.terminate()` kills every worker rather than the one that hung -- discarding the work every other
worker had in flight. So each slot is its own single-worker executor: a slot that exceeds its budget is
shut down, its process killed, and a fresh slot put in its place, while the other slots keep running.

**A killed cell is data.** It writes a record with `status="timeout"` exactly as a crashed cell writes
`status="crashed"`. A grid where the hardest scenarios kill the weakest arms, and where those cells then
simply vanish, is the textbook shape of survivorship bias -- the arms that fail most would come out of the
aggregate looking best.

The pool is deliberately small and explicit rather than general. It does not need a work-stealing
scheduler or a shared memory manager; it needs to run a few thousand independent cells, none of which
communicates with another, and to survive the ones that do not come back.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["WORKERS", "WORKERS_ENV_VAR", "THREADS_PER_WORKER", "DEFAULT_CELL_TIMEOUT_S", "resolve_workers", "worker_initializer", "CellPool", "PoolOutcome"]

#: Four of this machine's sixteen physical cores. Half of them was measured to exhaust the Windows paging
#: file under joblib fan-out, which is a failure mode that looks like a flaky benchmark rather than like
#: over-subscription.
WORKERS = 4

#: Overrides :data:`WORKERS`. Needed because the default is calibrated for THIS machine's sixteen cores,
#: and a two-core CI runner running four workers of four threads each measures its own queue rather than
#: the arms.
WORKERS_ENV_VAR = "MLFRAME_BENCH_WORKERS"

#: Threads inside each worker. Four workers times four threads saturates the machine without letting any
#: single arm's inner parallelism fight another's.
THREADS_PER_WORKER = 4

#: Per-cell wall-clock budget. Generous: the point is to stop a cell that has hung, not to race the slow
#: wrapper arms, whose honest cost on the widest beds runs into minutes.
DEFAULT_CELL_TIMEOUT_S = 1800.0

#: Set in the worker before numpy is imported. Every one of these is read once at library import time, so
#: setting them afterwards is silently ineffective -- which is the failure this list exists to prevent.
_THREAD_VARS: Tuple[str, ...] = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")


def resolve_workers(default: int = WORKERS) -> int:
    """Return the worker count, honouring the environment override.

    A value that is not a positive integer is ignored WITH a warning rather than silently accepted: a
    typo'd override that fell through to one worker would make a nightly run take four times as long and
    nothing would say why.
    """
    raw = os.environ.get(WORKERS_ENV_VAR)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using the default of %d workers", WORKERS_ENV_VAR, raw, default)
        return int(default)
    if value < 1:
        logger.warning("%s=%d is not a usable worker count; using the default of %d", WORKERS_ENV_VAR, value, default)
        return int(default)
    return value


def worker_initializer(threads: int = THREADS_PER_WORKER, cpu_only: bool = True, prewarm: bool = True) -> None:
    """Configure a fresh worker process: thread counts, GPU visibility, and a warmed numba cache.

    Args:
        threads: Threads each numerical library may use inside this worker.
        cpu_only: Hide the GPU from this worker. CPU workers must not touch it: several of them contending
            for one device is slower than any of them alone, and the GPU cells run in their own serial
            queue for exactly that reason.
        prewarm: Compile the hot numba kernels here. The parent compiles them first, so this is a cache
            HIT rather than a compile -- concurrent writes to the numba cache files are a known race, and
            the parent going first is what makes the workers' warm-up safe.
    """
    for name in _THREAD_VARS:
        os.environ[name] = str(int(threads))
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TQDM_DISABLE", "1")
    if prewarm:
        prewarm_kernels()


def prewarm_kernels() -> bool:
    """Compile the numba kernels the arms use, returning whether the warm-up completed.

    Run in the PARENT before any worker starts, and again inside each worker where it is a cache hit.
    Concurrent first-compiles race on the ``.nbi``/``.nbc`` cache files -- a race this repository's own
    test suite already skips tests over under xdist -- so the ordering is the point, not the warm-up.

    Returns:
        True when the warm-up ran to completion. A failure is logged and reported, never swallowed: a
        silent failure here means every worker pays a first-compile inside its first timed cell, and that
        cell's cost is then mostly the compiler's.
    """
    try:
        import numpy as np

        from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

        rng = np.random.default_rng(0)
        # Integer class codes with their frequency tables, which is the shape every mutual-information arm
        # in the roster eventually reaches. Compiling it here is what the workers get to skip.
        classes_x = rng.integers(0, 8, 512).astype(np.int32)
        classes_y = rng.integers(0, 2, 512).astype(np.int32)
        compute_mi_from_classes(classes_x, np.bincount(classes_x, minlength=8).astype(np.int32), classes_y, np.bincount(classes_y, minlength=2).astype(np.int32))
        return True
    except Exception as exc:
        logger.warning("numba prewarm did not complete (%s: %s); the first timed cell in each worker will include a compile", type(exc).__name__, exc)
        return False


#: How long a fresh worker gets to come up. Its cold start imports numpy, sklearn and the numba subgraph,
#: which is tens of seconds on this machine -- far longer than a cheap cell's own budget, which is exactly
#: why the warm-up is separated from the first cell rather than charged to it.
WORKER_STARTUP_TIMEOUT_S = 600.0


def _ready() -> bool:
    """Trivial job whose only purpose is to prove a worker has finished starting."""
    return True


@dataclass(frozen=True)
class PoolOutcome:
    """What came back from one dispatched cell."""

    key: Any
    record: Optional[Dict[str, Any]]
    status: str
    elapsed_s: float
    error: str = ""


class _Slot:
    """One worker slot: a single-process executor that can be replaced when its process has to be killed."""

    def __init__(self, threads: int, cpu_only: bool) -> None:
        self.threads = threads
        self.cpu_only = cpu_only
        self.executor: ProcessPoolExecutor = self._spawn()
        self.future: Optional["Future[Dict[str, Any]]"] = None
        self.key: Any = None
        self.started_at: float = 0.0

    def _spawn(self) -> ProcessPoolExecutor:
        """Start a fresh single-worker executor and WAIT for it to finish coming up.

        Waiting here is the whole point. A worker's cold start imports numpy, sklearn and the numba
        subgraph, which takes tens of seconds; without this, that time lands inside the first cell the slot
        runs and the per-cell budget is spent before the cell begins. Measured: with a five-second budget,
        every cell in a freshly spawned pool was killed as a timeout, including ones that return instantly.

        The warm-up also makes the initializer's numba prewarm happen off the clock, which is what it was
        for: the parent compiles first, the worker hits the cache, and neither cost lands in a timing.
        """
        executor = ProcessPoolExecutor(max_workers=1, initializer=worker_initializer, initargs=(self.threads, self.cpu_only, True))
        try:
            executor.submit(_ready).result(timeout=WORKER_STARTUP_TIMEOUT_S)
        except Exception as exc:
            # A worker that cannot start is fatal for this slot; the pool would otherwise dispatch into it
            # forever and charge each cell a timeout for a process that was never alive.
            executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(f"a benchmark worker did not finish starting within {WORKER_STARTUP_TIMEOUT_S:.0f}s: {type(exc).__name__}: {exc}") from exc
        return executor

    def busy(self) -> bool:
        """True while this slot is holding an unfinished cell."""
        return self.future is not None

    def submit(self, key: Any, function: Callable[..., Dict[str, Any]], *args: Any) -> None:
        """Dispatch one cell into this slot."""
        self.key = key
        self.started_at = time.perf_counter()
        self.future = self.executor.submit(function, *args)

    def recycle(self) -> None:
        """Kill this slot's process and put a fresh one in its place.

        ``cancel_futures`` plus a non-waiting shutdown is what actually stops a running task: the executor
        tears its process down rather than waiting for work that, by the time this is called, has already
        proven it will not finish.
        """
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            logger.warning("could not shut a hung worker down cleanly (%s: %s); replacing it anyway", type(exc).__name__, exc)
        self.future = None
        self.key = None
        self.executor = self._spawn()

    def close(self) -> None:
        """Shut this slot down at the end of a run."""
        self.executor.shutdown(wait=True)


class CellPool:
    """A small fixed pool that runs independent cells and enforces a per-cell wall-clock budget.

    Two queues, not one. The CPU slots run in parallel with the GPU hidden from them; the GPU slots are a
    SERIAL queue of one, because several processes contending for a single device is measurably slower
    than any one of them alone -- and worse, it makes each one's timing a function of what the others
    happened to be doing, which is the one thing a cost axis cannot survive.
    """

    def __init__(
        self,
        workers: Optional[int] = None,
        threads: int = THREADS_PER_WORKER,
        timeout_s: float = DEFAULT_CELL_TIMEOUT_S,
        cpu_only: bool = True,
        heartbeat_s: float = 60.0,
        gpu_workers: int = 0,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.heartbeat_s = float(heartbeat_s)
        # Compiled here, once, before any worker exists. The workers then hit a warm cache instead of
        # racing each other to write it.
        self.prewarmed = prewarm_kernels()
        self.slots: List[_Slot] = [_Slot(threads, cpu_only) for _ in range(max(1, resolve_workers() if workers is None else int(workers)))]
        # One at most, and only when asked for. A second GPU slot would not be a second queue; it would be
        # the same device, contended.
        self.gpu_slots: List[_Slot] = [_Slot(threads, cpu_only=False) for _ in range(min(1, max(0, int(gpu_workers))))]

    def _harvest(self, slot: _Slot, block: bool) -> Optional[PoolOutcome]:
        """Collect a finished or timed-out cell from one slot, or ``None`` when it is still running."""
        if slot.future is None:
            return None
        elapsed = time.perf_counter() - slot.started_at
        remaining = max(0.0, self.timeout_s - elapsed)
        try:
            record = slot.future.result(timeout=remaining if block else 0.0)
        except FutureTimeoutError:
            if elapsed < self.timeout_s:
                return None
            key = slot.key
            logger.warning("cell %s exceeded its %.0fs budget; killing the worker and recording a timeout", key, self.timeout_s)
            slot.recycle()
            return PoolOutcome(key=key, record=None, status="timeout", elapsed_s=elapsed)
        except Exception as exc:
            # A worker that died takes its executor with it, so the slot is replaced rather than reused.
            key = slot.key
            logger.warning("cell %s died in its worker (%s: %s)", key, type(exc).__name__, exc)
            slot.recycle()
            return PoolOutcome(key=key, record=None, status="crashed", elapsed_s=elapsed, error=f"{type(exc).__name__}: {exc}")
        key = slot.key
        slot.future = None
        slot.key = None
        return PoolOutcome(key=key, record=record, status=str(record.get("status", "ok")), elapsed_s=elapsed)

    def map(self, jobs: Iterable[Any]) -> Iterator[PoolOutcome]:
        """Run every job, yielding outcomes as they complete.

        Args:
            jobs: ``(key, function, args)`` triples, optionally with a fourth element -- a bool saying the
                cell needs the GPU. The function must be importable in a fresh process and its arguments
                picklable, which is why the runner passes a scenario NAME and a seed rather than a built
                frame: shipping a wide frame to every worker costs more than regenerating it, and the
                generator is a pure function of the two.

        Yields:
            One :class:`PoolOutcome` per job, in completion order rather than submission order.

        Raises:
            ValueError: When a job asks for the GPU and the pool was built without a GPU slot. Running it
                on a CPU worker would silently produce a timing for a different piece of work.
        """
        pending: List[Tuple[Any, Callable[..., Dict[str, Any]], Sequence[Any], bool]] = []
        for job in jobs:
            pending.append((job[0], job[1], job[2], bool(job[3]) if len(job) > 3 else False))
        if any(entry[3] for entry in pending) and not self.gpu_slots:
            raise ValueError("a job asks for the GPU but this pool has no GPU slot; running it on a CPU worker would time a different piece of work")

        cpu_queue = [entry for entry in pending if not entry[3]]
        gpu_queue = [entry for entry in pending if entry[3]]
        cpu_index = gpu_index = 0
        last_heartbeat = time.perf_counter()
        total = len(pending)
        done = 0

        def _dispatch(slots: List[_Slot], queue: List[Any], cursor: int) -> int:
            """Fill every free slot from its own queue, returning the new cursor."""
            for slot in slots:
                if not slot.busy() and cursor < len(queue):
                    key, function, args, _gpu = queue[cursor]
                    cursor += 1
                    slot.submit(key, function, *args)
            return cursor

        all_slots = self.slots + self.gpu_slots
        while cpu_index < len(cpu_queue) or gpu_index < len(gpu_queue) or any(slot.busy() for slot in all_slots):
            cpu_index = _dispatch(self.slots, cpu_queue, cpu_index)
            gpu_index = _dispatch(self.gpu_slots, gpu_queue, gpu_index)

            progressed = False
            for slot in all_slots:
                outcome = self._harvest(slot, block=False)
                if outcome is not None:
                    done += 1
                    progressed = True
                    yield outcome

            now = time.perf_counter()
            if now - last_heartbeat >= self.heartbeat_s:
                last_heartbeat = now
                running = [(slot.key, round(now - slot.started_at, 1)) for slot in all_slots if slot.busy()]
                logger.info("HEARTBEAT %d/%d cells done, %d in flight: %s", done, total, len(running), running)

            if not progressed:
                # Nothing finished this pass. Sleep briefly rather than spinning: the loop is a supervisor,
                # and a busy-wait here would take a core away from the work it is supervising.
                time.sleep(0.2)

    def close(self) -> None:
        """Shut every slot down, GPU queue included."""
        for slot in self.slots + self.gpu_slots:
            slot.close()

    def __enter__(self) -> "CellPool":
        """Return the pool, so it can be used as a context manager."""
        return self

    def __exit__(self, *exc: Any) -> "Literal[False]":
        """Always shut the workers down, so a failed run does not leave processes behind."""
        self.close()
        return False
