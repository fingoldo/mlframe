"""Thread vs process backend for report rendering overlapped with a GIL-releasing "training" workload.

Replays FigureSpecs captured from a real suite run (``--specs specs.pkl``: a pickled list of ``(spec, output, base_path, kwargs)``)
while the main thread fits a CatBoost model; measures wall time for: train alone, render alone, sequential, thread-overlapped,
process-overlapped. Paired/interleaved, best of N.

Run:  ``python -m mlframe.reporting._benchmarks.bench_async_render_overlap --specs specs.pkl --rounds 3``
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np


def _train(n: int = 120_000, iters: int = 60, threads: int = 2, sleep_seconds: float = 0.0) -> float:
    """A native, GIL-releasing CatBoost fit standing in for the next model's training (or, with ``sleep_seconds``, an idle-CPU wait).

    The sleep variant models training that does not use this host's CPU (GPU fit) or hogs none of it: it isolates the overlap
    mechanism from how many free cores the benchmark host happens to have.
    """
    if sleep_seconds > 0:
        t0 = time.perf_counter()
        time.sleep(sleep_seconds)
        return time.perf_counter() - t0
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 30)).astype("float32")
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    t0 = time.perf_counter()
    CatBoostClassifier(iterations=iters, verbose=0, thread_count=threads, task_type="CPU").fit(X, y)
    return time.perf_counter() - t0


def _rebase(specs: List[Tuple[Any, Any, str, dict]], out: str) -> List[Tuple[Any, Any, str]]:
    """Point every captured spec at ``out`` (keeping the file stem)."""
    return [(s, o, os.path.join(out, os.path.basename(b))) for s, o, b, _kw in specs]


def _render_all_sync(items: List[Tuple[Any, Any, str]]) -> float:
    """Render every spec inline."""
    from mlframe.reporting.renderers.save import render_and_save_now

    t0 = time.perf_counter()
    for spec, output, base in items:
        render_and_save_now(spec, output, base, interactive=False, format_subfolders=True)
    return time.perf_counter() - t0


def _overlapped(items: List[Tuple[Any, Any, str]], backend: str, workers: int, train_kwargs: Dict[str, Any]) -> Tuple[float, float]:
    """Submit every render to a queue, train on the main thread, then join. Returns ``(wall, train_seconds)``."""
    from mlframe.reporting._async_render import ReportRenderQueue
    from mlframe.reporting.async_render_hooks import _render_spec_task

    q = ReportRenderQueue(backend=backend, workers=workers)
    if backend == "process":
        q.warm()
    t0 = time.perf_counter()
    for spec, output, base in items:
        q.submit(_render_spec_task, spec, output, base, True, name=os.path.basename(base))
    train_s = _train(**train_kwargs)
    summary = q.close()
    wall = time.perf_counter() - t0
    if summary.failed:
        raise RuntimeError(f"{summary.failed} renders failed: {summary.failures[:2]}")
    return wall, train_s


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--train-threads", type=int, default=2)
    ap.add_argument("--train-sleep", type=float, default=0.0, help="replace the CatBoost fit by a sleep of this many seconds")
    a = ap.parse_args(argv)
    with open(a.specs, "rb") as fh:
        specs = pickle.load(fh)  # nosec B301 - round-trip of an object this code just pickled
    tk: Dict[str, Any] = {"threads": a.train_threads, "sleep_seconds": a.train_sleep}
    out = tempfile.mkdtemp(prefix="bench_async_render_")
    _train(**tk)  # warm
    _render_all_sync(_rebase(specs, os.path.join(out, "warm")))
    res: Dict[str, List[float]] = {k: [] for k in ("train_alone", "render_alone", "sequential", "thread_overlap", "process_overlap")}
    for r in range(a.rounds):
        items = _rebase(specs, os.path.join(out, f"r{r}"))
        res["train_alone"].append(_train(**tk))
        res["render_alone"].append(_render_all_sync(items))
        res["sequential"].append(res["train_alone"][-1] + res["render_alone"][-1])
        res["thread_overlap"].append(_overlapped(items, "thread", a.workers, tk)[0])
        res["process_overlap"].append(_overlapped(items, "process", a.workers, tk)[0])
        print(f"round {r}: " + ", ".join(f"{k}={v[-1]:.2f}s" for k, v in res.items()), flush=True)
    print("\nbest-of-%d:" % a.rounds)
    for k, v in res.items():
        print(f"  {k:16s} best {min(v):6.2f}s  median {float(np.median(v)):6.2f}s")
    shutil.rmtree(out, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
