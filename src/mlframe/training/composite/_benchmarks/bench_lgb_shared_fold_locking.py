"""What serialising LightGBM costs the threaded tiny-rerank.

Run: ``python -m mlframe.training.composite._benchmarks.bench_lgb_shared_fold_locking [n_threads] [rounds]``

Three modes over the same workload, shaped like the rerank that crashed in production (fold matrices of 13.3k x 100
float32, 60 rounds, 15 leaves, several specs per fold so the dataset is reused):

* ``none``      -- what the code did before: every thread constructs and trains without serialisation;
* ``construct`` -- what it does now: one process-wide lock around ``Dataset.construct()`` only;
* ``full``      -- construct and ``lgb.train`` both serialised, i.e. the rerank's LightGBM work single-file.

``none`` is the unsafe baseline and is here only to price the fix; it is the configuration that took an access
violation plus a heap corruption (0xc0000374) in the production kernel.

Verdict (16 threads, 3 rounds, 3 specs per fold, lightgbm 4.6.0): none 58.9 s, construct 55.4 s (0.94x, i.e. free at
this thread count), full 117.5 s (1.99x). Serialising construction only is therefore the shipped behaviour; full
serialisation is rejected -- it doubles the rerank to protect the part LightGBM already runs in parallel safely.
"""
from __future__ import annotations

import contextlib
import sys
import threading
import time

import numpy as np

from mlframe.training.composite.discovery import _lgb_shared_fold as sf

_TRAIN_LOCK = threading.Lock()


def _worker(seed: int, rounds: int, n_rows: int, n_cols: int, n_estimators: int, specs: int, full: bool) -> None:
    """One rerank worker: a fresh fold matrix per round, several labels scored on it."""
    rng = np.random.default_rng(seed)
    params = sf.lgb_params(num_leaves=15, learning_rate=0.1, random_state=seed, deterministic=False, num_threads=1)
    for _ in range(rounds):
        x = rng.normal(size=(n_rows, n_cols)).astype(np.float32)
        rows = np.arange(n_rows)
        for _spec in range(specs):
            target = rng.normal(size=n_rows)
            with _TRAIN_LOCK if full else contextlib.nullcontext():
                booster = sf.fit_on_shared_fold(x, rows, target, params=params, n_estimators=n_estimators)
            booster.predict(x[:256])


def _run(mode: str, n_threads: int, rounds: int, n_rows: int, n_cols: int, n_estimators: int, specs: int) -> float:
    """Wall time of one mode."""
    original = sf._CONSTRUCT_LOCK
    if mode == "none":
        sf._CONSTRUCT_LOCK = contextlib.nullcontext()  # type: ignore[assignment]
    try:
        sf._CACHE.clear()
        threads = [
            threading.Thread(target=_worker, args=(i, rounds, n_rows, n_cols, n_estimators, specs, mode == "full"))
            for i in range(n_threads)
        ]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return time.perf_counter() - t0
    finally:
        sf._CONSTRUCT_LOCK = original


if __name__ == "__main__":
    n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_rows, n_cols, n_estimators, specs = 13_300, 100, 60, 3
    _run("construct", 2, 1, 2_000, 20, 10, 1)  # warm the import / first-fit paths so the timings compare like with like
    results = {}
    for mode in ("none", "construct", "full"):
        results[mode] = _run(mode, n_threads, rounds, n_rows, n_cols, n_estimators, specs)
        print(f"{mode:>9}: {results[mode]:7.1f}s  (threads={n_threads}, rounds={rounds}, specs/fold={specs})")
    base = results["none"]
    for mode in ("construct", "full"):
        print(f"{mode:>9}: {results[mode] / base:5.2f}x the unsafe baseline")
