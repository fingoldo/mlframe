"""Verify the Windows loky stack-overflow fix for numba JIT cache load.

Background: ``joblib.Parallel(backend="loky")`` workers on Windows have
the OS-default 1MB main-thread stack. Numba's first JIT-cache load
calls ``llvmlite.binding.executionengine.finalize_object`` which uses
~2-3MB of stack and crashes the worker.

Fix lives in ``_joblib_safe.run_in_big_stack_thread`` -- runs the body
in a thread with 8MB stack (Windows only; pass-through on Linux).

This script exercises ``optimise_hermite_pair`` in loky workers via
the fix wrapper. PASS criterion: all tasks return a finite MI value
(no worker crash, no NaN).

Run::

    set PYTHONUNBUFFERED=1
    python profiling/verify_loky_stackoverflow_fix.py
"""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

# Force unbuffered stdout so progress is visible even when redirected.
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

import numpy as np
from joblib import Parallel, delayed

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

# Direct-module import bypasses mlframe.feature_selection.__init__ which
# pulls scipy.stats (broken in this env -- "error return without exception
# set" on import). Path is stable; only the package init is poisoned.
_pair_mod = importlib.import_module(
    "mlframe.feature_selection.filters._hermite_fe_optimise_pair"
)
_safe_mod = importlib.import_module(
    "mlframe.feature_selection.filters._joblib_safe"
)
optimise_hermite_pair = _pair_mod.optimise_hermite_pair
run_in_big_stack_thread = _safe_mod.run_in_big_stack_thread


def _worker_impl(seed: int):
    rng = np.random.default_rng(seed)
    x_a = rng.uniform(-1.5, 1.5, 2000).astype(np.float64)
    x_b = rng.uniform(-1.5, 1.5, 2000).astype(np.float64)
    y_cont = 2.0 * x_a * x_a - x_b + rng.normal(0, 0.15, 2000)
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    res = optimise_hermite_pair(
        x_a=x_a, x_b=x_b, y=y,
        discrete_target=True,
        max_degree=4, min_degree=3,
        n_trials=20, coef_range=(-2.0, 2.0),
        seed=42 + seed,
        sweep_degrees=True, basis="hermite", mi_estimator="plugin",
        optimizer="cma_batch", multi_fidelity=False,
    )
    return float(res.mi) if res is not None else float("nan")


def _worker_with_fix(seed: int):
    return run_in_big_stack_thread(_worker_impl, seed)


def main():
    n_workers = 4
    n_tasks = 8

    print(f"# verify_loky_stackoverflow_fix")
    print(f"#   platform={sys.platform}  n_workers={n_workers}  n_tasks={n_tasks}")
    print()
    sys.stdout.flush()

    t0 = time.perf_counter()
    mis = Parallel(n_jobs=n_workers, backend="loky", verbose=5)(
        delayed(_worker_with_fix)(s) for s in range(n_tasks)
    )
    elapsed = time.perf_counter() - t0

    finite = [mi for mi in mis if np.isfinite(mi)]
    ok = len(finite) == n_tasks
    print()
    print(f"# elapsed={elapsed:.1f}s  finite={len(finite)}/{n_tasks}  ok={ok}")
    print(f"# mis={[f'{m:.3f}' for m in mis]}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
