"""Multi-pair benchmark: ``cma_batch`` per-pair via joblib loop vs ``numba_kernel``
all-pairs in one ``@njit(parallel=True)`` call.

Single-pair bench (``bench_polynom_optimizers.py``) showed CMA-ES is a
strictly smarter sampler than random+elitism, so ``cma_batch`` wins
per-pair. But the polynom-pair FE production path runs 12-54 pairs
each call -- if we can avoid joblib's per-worker spinup + memmap
setup by handing all pairs to one numba prange kernel, the
sampler-quality gap may be outweighed by parallelism gains.

This bench answers: at what pair count does ``numba_kernel`` overtake
``cma_batch + joblib`` on total wall-clock for the full FE pass?

Both paths compute the same per-pair output (best mi). Reports total
wall-clock per path + per-pair MI agreement.

Run::

    python -m mlframe.profiling.bench_polynom_optimizers_multipair
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from mlframe.feature_selection.filters._hermite_fe_optimise_pair import optimise_hermite_pair
from mlframe.feature_selection.filters._numba_polynom_optimizer import (
    optimize_all_pairs_numba_kernel,
)


N_ROWS = 4000
N_FEATURES = 12          # pairs per call grow ~quadratically with this.
N_TRIALS = 200
MIN_DEGREE = 3
MAX_DEGREE = 6
COEF_RANGE = (-2.0, 2.0)
BASIS = "hermite"
BF_NAMES = ("mul", "add", "sub", "div")
N_SEEDS = 1              # multi-pair bench is heavier per cell; one seed.


def _make_synthetic(n_rows: int, n_features: int, seed: int = 0):
    """Build (X, y, pair_list). y is sign(2*x_0^2 - x_1 + noise)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, (n_rows, n_features)).astype(np.float64)
    noise = rng.normal(0, 0.15, n_rows)
    y_cont = 2.0 * X[:, 0] * X[:, 0] - X[:, 1] + noise
    y = (y_cont > np.median(y_cont)).astype(np.int64)
    # All C(F, 2) pairs.
    pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pairs.append((i, j))
    return X, y, pairs


def _eval_one_pair_cma_batch(X, y, pair_tuple):
    """Per-pair worker for path A. Module-level so cloudpickle doesn't
    have to serialise enclosing test-scope state."""
    ca, cb = pair_tuple
    x_a = X[:, ca]
    x_b = X[:, cb]
    best_mi = -np.inf
    # Mirror polynom_pair_fe.py's restart loop -- fe_smart_polynom_iters=3.
    for seed_offset in range(3):
        res = optimise_hermite_pair(
            x_a=x_a, x_b=x_b, y=y,
            discrete_target=True,
            max_degree=MAX_DEGREE, min_degree=MIN_DEGREE,
            n_trials=N_TRIALS, coef_range=COEF_RANGE,
            seed=42 + seed_offset,
            sweep_degrees=True, basis=BASIS, mi_estimator="plugin",
            optimizer="cma_batch", multi_fidelity=False,
        )
        if res is not None and res.mi > best_mi:
            best_mi = res.mi
    return best_mi


def _path_a_joblib_cma_batch(X, y, pairs, n_jobs: int):
    """Current production path: joblib loop over pairs, each calls
    ``optimise_hermite_pair`` with cma_batch optimizer.

    Backend: threading. On Windows, ``loky`` workers hit numba/llvmlite
    cache-load stack overflow during JIT finalize (1MB Windows thread
    stack + deep LLVM finalize call chain). Threading avoids the fresh
    process JIT-cache reload. cma_batch is Python-bound on the CMA
    tell/ask so threading parallelism is bounded by GIL but the bench
    measures wall-clock the same way prod observes it on Windows.
    """
    return Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_eval_one_pair_cma_batch)(X, y, p) for p in pairs
    )


def _path_b_numba_kernel(X, y, pairs):
    """All-pairs numba kernel: ONE call processes all pairs via prange.

    Loops degree combinations on the Python side (each degree combo has
    different coef sizes, can't vary inside one njit call), but each
    degree call processes ALL pairs in parallel via numba threads.
    """
    pair_indices = np.array(pairs, dtype=np.int64)
    P = len(pairs)
    best_per_pair = np.full(P, -np.inf, dtype=np.float64)

    # Degree sweep: try each (deg_a, deg_b) combination. ca_size = deg+1.
    # 3 restarts to match path A; per restart use different seed.
    for restart in range(3):
        for deg_a in range(MIN_DEGREE, MAX_DEGREE + 1):
            for deg_b in range(MIN_DEGREE, MAX_DEGREE + 1):
                ca_size = deg_a + 1
                cb_size = deg_b + 1
                result = optimize_all_pairs_numba_kernel(
                    X, y, pair_indices,
                    ca_size=ca_size, cb_size=cb_size,
                    coef_range=COEF_RANGE,
                    basis=BASIS,
                    bf_names=BF_NAMES,
                    n_trials=N_TRIALS,
                    batch_size=20, elitism_k=4, perturb_sigma_frac=0.1,
                    n_bins=20, l2_penalty=0.05,
                    direction_only=False, discrete_target=True,
                    seed=42 + restart * 100 + deg_a * 10 + deg_b,
                )
                for p in range(P):
                    if result["best_raws"][p] > best_per_pair[p]:
                        best_per_pair[p] = result["best_raws"][p]
    return best_per_pair.tolist()


def main():
    # Warmup: trigger JIT compile in the parent process so threading
    # workers reuse the warm cache. Tiny problem size keeps warmup cheap.
    print("# warmup (JIT compile)...")
    Xw, yw, pw = _make_synthetic(n_rows=400, n_features=6, seed=99)
    # Warmup path A by calling per-pair fn once in main proc.
    _eval_one_pair_cma_batch(Xw, yw, pw[0])
    _path_b_numba_kernel(Xw, yw, pw)
    print()

    print("# bench_polynom_optimizers_multipair")
    print(f"#   N_ROWS={N_ROWS}, N_FEATURES={N_FEATURES}, N_TRIALS={N_TRIALS}, "
          f"degrees={MIN_DEGREE}-{MAX_DEGREE}, restarts=3")
    print(f"#   basis={BASIS}, bf_names={BF_NAMES}")
    print()

    X, y, pairs = _make_synthetic(N_ROWS, N_FEATURES, seed=0)
    print(f"#   n_pairs={len(pairs)} (C({N_FEATURES}, 2))")
    print()

    # Path A: joblib + cma_batch
    n_jobs = 16
    t0 = time.perf_counter()
    mis_a = _path_a_joblib_cma_batch(X, y, pairs, n_jobs=n_jobs)
    t_a = time.perf_counter() - t0
    print(f"  cma_batch + joblib(n_jobs={n_jobs}):  {t_a:.2f}s  "
          f"mi_mean={np.mean(mis_a):.4f}  mi_max={np.max(mis_a):.4f}")

    # Path B: numba multi-pair kernel
    t0 = time.perf_counter()
    mis_b = _path_b_numba_kernel(X, y, pairs)
    t_b = time.perf_counter() - t0
    print(f"  numba_kernel multi-pair (one prange call/deg-combo): "
          f"{t_b:.2f}s  mi_mean={np.mean(mis_b):.4f}  mi_max={np.max(mis_b):.4f}")

    print()
    speedup = t_a / t_b if t_b > 0 else float("nan")
    print(f"#   speedup numba_kernel vs cma_batch+joblib: {speedup:.2f}x")
    print()

    # Per-pair MI delta
    mis_a_arr = np.asarray(mis_a)
    mis_b_arr = np.asarray(mis_b)
    delta = mis_b_arr - mis_a_arr
    print(f"# per-pair MI delta (numba_kernel - cma_batch):")
    print(f"   mean = {np.mean(delta):+.4f}  (positive = numba_kernel better)")
    print(f"   max  = {np.max(delta):+.4f}")
    print(f"   min  = {np.min(delta):+.4f}")
    print(f"   |max| / mi_mean = {np.max(np.abs(delta)) / np.mean(mis_a_arr) * 100:.2f}%")


if __name__ == "__main__":
    main()
