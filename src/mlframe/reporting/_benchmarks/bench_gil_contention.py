"""How much does a background render thread slow a pure-Python main thread, versus a background render PROCESS?

The main thread runs a fixed pure-Python workload ``W`` (a hot loop: GIL-bound, like the suite's own per-model Python work) while a
worker renders captured FigureSpecs in (a) nothing, (b) a thread, (c) a spawned process. ``W``'s wall time in (b) minus (c) is the
GIL contention cost of the thread backend; (c) minus (a) is plain CPU competition. A GIL-releasing workload (catboost) is covered
by ``bench_async_render_overlap``. Paired/interleaved, median of N rounds; also reports the workload's thread CPU time.

Run:  ``python -m mlframe.reporting._benchmarks.bench_gil_contention --specs specs.pkl --rounds 5``
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np


def _workload(iters: int) -> Tuple[float, float]:
    """Pure-Python hot loop; returns (wall, thread_cpu) seconds."""
    w0, c0 = time.perf_counter(), time.thread_time()
    acc = 0
    for i in range(iters):
        acc += i * i % 7
    return time.perf_counter() - w0, time.thread_time() - c0


def _render_loop(specs: List[Tuple[Any, Any, str, dict]], out: str, stop: threading.Event) -> None:
    """Render the captured specs repeatedly until ``stop`` is set."""
    from mlframe.reporting.async_render_hooks import _render_spec_task

    k = 0
    while not stop.is_set():
        for spec, output, base, _kw in specs:
            if stop.is_set():
                return
            _render_spec_task(spec, output, os.path.join(out, f"{k}_{os.path.basename(base)}"), True)
        k += 1


def _process_render_loop(specs_path: str, out: str, stop_path: str) -> None:
    """Process-backend body: the same loop, stopped by the presence of ``stop_path``."""
    with open(specs_path, "rb") as fh:
        specs = pickle.load(fh)
    from mlframe.reporting.async_render_hooks import _render_spec_task

    k = 0
    while not os.path.exists(stop_path):
        for spec, output, base, _kw in specs:
            if os.path.exists(stop_path):
                return
            _render_spec_task(spec, output, os.path.join(out, f"{k}_{os.path.basename(base)}"), True)
        k += 1


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20_000_000)
    a = ap.parse_args(argv)
    with open(a.specs, "rb") as fh:
        specs = pickle.load(fh)
    out = tempfile.mkdtemp(prefix="bench_gil_")
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    ex = ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn"))
    ex.submit(os.getpid).result()
    _workload(2_000_000)
    res: Dict[str, List[Tuple[float, float]]] = {"alone": [], "thread": [], "process": []}
    for r in range(a.rounds):
        res["alone"].append(_workload(a.iters))
        stop = threading.Event()
        t = threading.Thread(target=_render_loop, args=(specs, os.path.join(out, "t"), stop), daemon=True)
        t.start()
        time.sleep(1.0)  # let rendering reach steady state
        res["thread"].append(_workload(a.iters))
        stop.set()
        t.join()
        stop_path = os.path.join(out, f"stop{r}")
        fut = ex.submit(_process_render_loop, a.specs, os.path.join(out, "p"), stop_path)
        time.sleep(3.0)
        res["process"].append(_workload(a.iters))
        open(stop_path, "w").close()
        fut.result()
        print(f"round {r}: " + ", ".join(f"{k}={v[-1][0]:.2f}s(cpu {v[-1][1]:.2f})" for k, v in res.items()), flush=True)
    base = float(np.median([w for w, _ in res["alone"]]))
    print("\nmedian main-thread workload wall (x alone):")
    for k, v in res.items():
        m = float(np.median([w for w, _ in v]))
        c = float(np.median([c for _, c in v]))
        print(f"  {k:8s} {m:6.2f}s  x{m / base:4.2f}   thread-cpu {c:5.2f}s")
    ex.shutdown()
    shutil.rmtree(out, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
