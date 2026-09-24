"""What protecting the MASKED-fold LightGBM path costs, and which way of protecting it to pick.

Run: ``python -m mlframe.training.composite._benchmarks.bench_lgb_masked_fold_locking [n_threads] [rounds]``

A rerank fold whose spec trains on a subset of its rows cannot reuse the cached bins, so it used to go through the
sklearn wrapper, which constructs its dataset inside ``fit`` with no serialisation. That is the construction site a
production kernel died in AFTER the cached path had been serialised. Three ways to run it:

* ``sklearn``     -- the old behaviour: ``LGBMRegressor.fit``, construction unprotected;
* ``sklearn_lock`` -- the whole ``fit`` under the shared lock, so training serialises with it;
* ``native``      -- what the code does now: dataset built under the lock and cached by its row set, ``lgb.train`` outside it.

``native`` predicts bit-identically to ``sklearn`` (verified for both the deterministic and the default mode).

Verdict (16 threads, 3 rounds, 3 specs per fold, lightgbm 4.6.0): sklearn 27.2 s, sklearn_lock 164.7 s (6.06x),
native 23.4 s (0.86x). The safe path is the cheapest of the three because the dataset is cached under the rows it was
binned on, so the seed repeats of one spec reuse it where the wrapper rebuilt it every time. Locking the wrapper's
whole fit is rejected: it serialises the training too, which is most of the work.
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np

from mlframe.training.composite.discovery._lgb_shared_fold import _CONSTRUCT_LOCK, fit_on_rows, lgb_params
from mlframe.training.composite.discovery._screening_tiny import _build_tiny_model


def _sklearn_fit(x, fit_rows, target, *, num_leaves, learning_rate, random_state, n_estimators, locked: bool):
    """The wrapper path, optionally with the whole fit serialised."""
    model = _build_tiny_model(
        "lgb", n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate,
        random_state=random_state, deterministic=False, inner_n_jobs=1,
    )
    model.set_params(n_jobs=1)
    if locked:
        with _CONSTRUCT_LOCK:
            model.fit(x[fit_rows], target)
    else:
        model.fit(x[fit_rows], target)
    return model


def _worker(mode: str, seed: int, rounds: int, n_rows: int, n_cols: int, n_estimators: int, specs: int) -> None:
    """One rerank worker on masked folds: a fresh matrix per round, several specs each training on ~60% of its rows."""
    rng = np.random.default_rng(seed)
    params = lgb_params(num_leaves=15, learning_rate=0.1, random_state=seed, deterministic=False, num_threads=1)
    for _ in range(rounds):
        x = rng.normal(size=(n_rows, n_cols)).astype(np.float32)
        fit_rows = np.sort(rng.choice(n_rows, int(n_rows * 0.6), replace=False))
        for _spec in range(specs):
            target = rng.normal(size=fit_rows.shape[0])
            if mode == "native":
                fit_on_rows(x, fit_rows, target, params=params, n_estimators=n_estimators)
            else:
                _sklearn_fit(
                    x, fit_rows, target, num_leaves=15, learning_rate=0.1, random_state=seed,
                    n_estimators=n_estimators, locked=(mode == "sklearn_lock"),
                )


def _run(mode: str, n_threads: int, rounds: int, n_rows: int, n_cols: int, n_estimators: int, specs: int) -> float:
    """Wall time of one mode."""
    threads = [
        threading.Thread(target=_worker, args=(mode, i, rounds, n_rows, n_cols, n_estimators, specs))
        for i in range(n_threads)
    ]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0


if __name__ == "__main__":
    n_threads = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_rows, n_cols, n_estimators, specs = 13_300, 100, 60, 3
    _run("native", 2, 1, 2_000, 20, 10, 1)  # warm the import / first-fit paths
    results = {}
    for mode in ("sklearn", "sklearn_lock", "native"):
        results[mode] = _run(mode, n_threads, rounds, n_rows, n_cols, n_estimators, specs)
        print(f"{mode:>13}: {results[mode]:7.1f}s  (threads={n_threads}, rounds={rounds}, specs/fold={specs})")
    base = results["sklearn"]
    for mode in ("sklearn_lock", "native"):
        print(f"{mode:>13}: {results[mode] / base:5.2f}x the unprotected wrapper path")
