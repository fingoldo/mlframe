"""Per-kernel GPU time and launch counts without an external profiler.

Nsight Systems needs an elevated process, Nsight Compute dropped Pascal and ``nvprof`` needs a CUPTI library that is not installed, so on a dev box that
cannot run them the launches are timed with CUDA events: every ``cupy.RawKernel``, ``ElementwiseKernel``, ``ReductionKernel`` and ``numba.cuda`` launch is
wrapped with a start / stop event pair on the launch stream, and the pairs are resolved after the run. The events add a few microseconds per launch, so
the totals are a ranking of where device time goes, not an exact timeline.

Usage::

    from mlframe.feature_selection._benchmarks.gpu_kernel_timer import kernel_timer
    with kernel_timer() as t:
        fit(...)
    t.report(top=25)
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections import defaultdict


class _Timer:
    """Collected launches: ``(name, start_event, stop_event)`` plus a per-name launch count."""

    def __init__(self) -> None:
        self.records: list = []
        self.launch_config: dict = {}
        self.first_stack: dict = {}
        self.host_ms: dict = {}
        self.first_args: dict = {}

    def totals(self) -> "dict[str, tuple[int, float]]":
        """``{kernel name: (launches, total milliseconds)}`` once every event has completed."""
        import cupy as cp

        cp.cuda.Device().synchronize()
        out: dict = defaultdict(lambda: [0, 0.0])
        for name, a, b in self.records:
            out[name][0] += 1
            out[name][1] += cp.cuda.get_elapsed_time(a, b)
        return {k: (v[0], v[1]) for k, v in out.items()}

    def shapes(self) -> "dict[str, str]":
        """``{kernel name: 'grid x block of the longest launch'}`` for the numba.cuda kernels (the launch configuration is known there)."""
        return dict(self.launch_config)

    def report(self, top: int = 25) -> str:
        """A text table of the ``top`` kernels by device time, with the launch total."""
        rows = sorted(self.totals().items(), key=lambda kv: -kv[1][1])
        total_ms = sum(v[1] for _, v in rows)
        total_n = sum(v[0] for _, v in rows)
        lines = [f"{'kernel':70s} {'launches':>9s} {'total ms':>10s} {'avg us':>9s} {'share':>6s}"]
        for name, (n, ms) in rows[:top]:
            lines.append(f"{name[:70]:70s} {n:9d} {ms:10.1f} {1000.0 * ms / max(n, 1):9.1f} {100.0 * ms / max(total_ms, 1e-9):5.1f}%  {self.launch_config.get(name, '')} host_launch_ms={self.host_ms.get(name, 0.0):.0f}")
        lines.append(f"{'TOTAL (' + str(len(rows)) + ' kernels)':70s} {total_n:9d} {total_ms:10.1f}")
        text = "\n".join(lines)
        print(text, flush=True)
        return text


@contextlib.contextmanager
def kernel_timer():
    """Time every cupy and numba.cuda kernel launched inside the block (see the module docstring).

    cupy's kernel classes are compiled extension types whose ``__call__`` cannot be replaced, so ``cupy.RawKernel`` / ``ElementwiseKernel`` /
    ``ReductionKernel`` are swapped for timing subclasses: enter the block BEFORE importing the code under test, since a module that already
    bound the original class (or built a kernel at import) is not seen.
    """
    import cupy as cp

    timer = _Timer()
    undo = []

    def timed_subclass(base):
        """A subclass of ``base`` whose calls are bracketed by events."""

        class Timed(base):  # type: ignore[valid-type, misc]
            """Timing wrapper."""

            def __call__(self, *a, **k):
                """Launch with start / stop events on the current stream."""
                s, e = cp.cuda.Event(), cp.cuda.Event()
                s.record()
                r = super().__call__(*a, **k)
                e.record()
                timer.records.append((getattr(self, "name", base.__name__), s, e))
                return r

        Timed.__name__ = base.__name__
        return Timed

    for attr in ("RawKernel", "ElementwiseKernel", "ReductionKernel"):
        orig = getattr(cp, attr)
        setattr(cp, attr, timed_subclass(orig))
        undo.append((cp, attr, orig))
    try:
        from numba.cuda.dispatcher import _LaunchConfiguration

        orig_call = _LaunchConfiguration.__call__

        def numba_timed(self, *a, **k):
            """Launch a numba.cuda kernel with start / stop events."""
            s, e = cp.cuda.Event(), cp.cuda.Event()
            s.record()
            t0 = time.perf_counter()
            r = orig_call(self, *a, **k)
            host_ms = (time.perf_counter() - t0) * 1000.0
            e.record()
            fn = self.dispatcher.py_func
            name = f"{getattr(fn, '__module__', '').rsplit('.', 1)[-1]}.{getattr(fn, '__name__', 'numba_kernel')}"
            timer.records.append((name, s, e))
            timer.host_ms[name] = timer.host_ms.get(name, 0.0) + host_ms
            timer.launch_config[name] = f"grid={self.griddim} block={self.blockdim} shared={getattr(self, 'sharedmem', 0)}"
            if name not in timer.first_stack:
                import traceback

                timer.first_args[name] = ", ".join(f"{getattr(x, 'shape', x)}:{getattr(x, 'dtype', '')}" for x in a)
                timer.first_stack[name] = "".join(traceback.format_stack(limit=9)[:-1])
            return r

        _LaunchConfiguration.__call__ = numba_timed
        undo.append((_LaunchConfiguration, "__call__", orig_call))
    except Exception as exc:  # numba.cuda internals differ between versions: its launches are then simply not listed
        logging.getLogger(__name__).debug("numba.cuda launches are not timed: %r", exc)
    try:
        yield timer
    finally:
        for cls, attr, orig in undo:
            setattr(cls, attr, orig)
