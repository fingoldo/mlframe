"""Bench: per-member MAE / STD vs cross-member median.

Compares the LOOP-MAE replacement variants on representative (K, N) shapes.

Variants:
- ``loop_py``       -- the prior Python-loop implementation.
- ``vectorised_np`` -- single broadcast over axis=1 (current default for small inputs).
- ``numba_njit``    -- ``@njit(parallel=True, fastmath=True, cache=True)`` (current default
  above the (K, N) crossover; see ``_per_member_mae_std`` dispatcher).

Usage::

    python -m mlframe.training._benchmarks.bench_ensemble_mae

Outputs a markdown table: shape | loop_py ms | numpy ms | numba ms | numpy speedup | numba speedup.
"""

from __future__ import annotations

import time
import numpy as np


def _loop_py(arr: np.ndarray, median_preds: np.ndarray):
    K = arr.shape[0]
    per_member_mae = np.empty(K, dtype=np.float64)
    per_member_std = np.empty(K, dtype=np.float64)
    for i in range(K):
        diffs = np.abs(arr[i] - median_preds)
        mae_per_col = diffs.mean(axis=0) if diffs.ndim > 0 else float(diffs)
        if diffs.ndim == 1:
            per_member_mae[i] = float(diffs.mean())
            per_member_std[i] = float(diffs.std())
        else:
            std_per_col = np.sqrt(((diffs - mae_per_col) ** 2).mean(axis=0))
            per_member_mae[i] = float(mae_per_col.mean())
            per_member_std[i] = float(std_per_col.mean())
    return per_member_mae, per_member_std


def _vectorised_np(arr: np.ndarray, median_preds: np.ndarray):
    diffs = np.abs(arr - median_preds)
    if arr.ndim == 2:
        per_member_mae = diffs.mean(axis=1)
        per_member_std = np.sqrt(((diffs - per_member_mae[:, None]) ** 2).mean(axis=1))
    else:
        mae_per_col = diffs.mean(axis=1)
        std_per_col = np.sqrt(((diffs - mae_per_col[:, None, :]) ** 2).mean(axis=1))
        per_member_mae = mae_per_col.mean(axis=1)
        per_member_std = std_per_col.mean(axis=1)
    return per_member_mae, per_member_std


def _bench(shape: tuple, n_repeats: int = 5) -> dict:
    rng = np.random.default_rng(42)
    arr = rng.normal(size=shape).astype(np.float64)
    median_preds = np.quantile(arr, 0.5, axis=0)

    # warm-up
    _loop_py(arr, median_preds)
    _vectorised_np(arr, median_preds)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        a, b = _loop_py(arr, median_preds)
    t_loop = (time.perf_counter() - t0) * 1000 / n_repeats

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        c, d = _vectorised_np(arr, median_preds)
    t_np = (time.perf_counter() - t0) * 1000 / n_repeats

    assert np.allclose(a, c) and np.allclose(b, d)  # nosec B101 - internal invariant check in src/mlframe/training/_benchmarks, not reachable with untrusted input

    # numba (best-effort)
    try:
        from mlframe.models.ensembling import _per_member_mae_std_njit, _HAS_NUMBA_PER_MEMBER

        if _HAS_NUMBA_PER_MEMBER and arr.ndim == 2:
            _per_member_mae_std_njit(arr, median_preds)  # warm-up JIT
            t0 = time.perf_counter()
            for _ in range(n_repeats):
                e, f = _per_member_mae_std_njit(arr, median_preds)
            t_numba = (time.perf_counter() - t0) * 1000 / n_repeats
            assert np.allclose(a, e, atol=1e-9) and np.allclose(b, f, atol=1e-9)  # nosec B101 - internal invariant check in src/mlframe/training/_benchmarks, not reachable with untrusted input
        else:
            t_numba = float("nan")
    except Exception:
        t_numba = float("nan")

    return {"shape": shape, "loop_ms": t_loop, "np_ms": t_np, "numba_ms": t_numba}


def main() -> None:
    print("LOOP-MAE bench (averaged over 5 calls; cold JIT excluded)\n")
    print("| shape | loop ms | numpy ms | numba ms | np speedup | numba speedup |")
    print("|---|---|---|---|---|---|")
    for shape in [(3, 1000), (5, 1000), (10, 1000), (20, 1000), (50, 10_000), (100, 50_000), (5, 500_000), (10, 1_000_000)]:
        r = _bench(shape, n_repeats=5)
        np_x = r["loop_ms"] / r["np_ms"] if r["np_ms"] else float("nan")
        nb_x = r["loop_ms"] / r["numba_ms"] if r["numba_ms"] == r["numba_ms"] else float("nan")
        print(f"| {shape} | {r['loop_ms']:.3f} | {r['np_ms']:.3f} | {r['numba_ms']:.3f} | {np_x:.1f}x | {nb_x:.1f}x |")


if __name__ == "__main__":
    main()
