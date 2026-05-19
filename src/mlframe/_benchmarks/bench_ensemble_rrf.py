"""Bench: numpy vs njit Reciprocal Rank Fusion (RRF) aggregation.

Pre-bench ``models.ensembling._rrf_aggregate_probs`` is pure numpy:
``np.argsort(-col, axis=1, kind="stable")`` per (M, N, K) class × the
``np.put_along_axis`` + reciprocal-rank-sum + per-row normalization
chain.

This bench compares the numpy implementation against a numba prange
variant that parallelises over the M-axis (each member's N-row sort
is independent across members).

Sweep over (M, N, K) -- M=5 / 10 / 20 (ensemble members), N=10k / 100k
/ 1M / 5M (rows), K=2 / 3 / 10 (classes). Reports min-of-5 wall + the
implementation that won + the crossover thresholds.

Run::

    PYTHONPATH=src D:/ProgramData/anaconda3/python.exe \\
        -m mlframe._benchmarks.bench_ensemble_rrf
"""
from __future__ import annotations

import time

import numpy as np

try:
    import numba as _numba
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False


# Reference numpy implementation (copied from models/ensembling.py for
# isolation -- so the bench measures the kernel, not the surrounding
# imports / module init).
def _rrf_aggregate_numpy(preds_arr: np.ndarray, k: int = 60) -> np.ndarray:
    if preds_arr.ndim != 3:
        preds_arr = preds_arr.reshape(preds_arr.shape[0], preds_arr.shape[1], -1)
    M, N, K = preds_arr.shape
    aggregated = np.zeros((N, K), dtype=np.float64)
    for k_class in range(K):
        col = preds_arr[:, :, k_class]
        order = np.argsort(-col, axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(N), axis=1)
        rr = 1.0 / (k + (ranks + 1).astype(np.float64))
        aggregated[:, k_class] = rr.sum(axis=0)
    if K > 1:
        row_sums = aggregated.sum(axis=1, keepdims=True)
        safe = np.where(row_sums > 0, row_sums, 1.0)
        aggregated = aggregated / safe
    return aggregated


if _HAS_NUMBA:

    @_numba.njit(parallel=True, cache=True, fastmath=False)
    def _rrf_aggregate_njit(preds_arr: np.ndarray, k: int = 60) -> np.ndarray:
        """Parallel-over-M njit RRF. Each member's argsort is independent;
        ``prange`` over the M axis exploits multi-core CPUs. fastmath=False
        because 1/(k+rank+1) is exact integer division-by-float; rounding
        differences from fast-math would drift the per-row sums.
        """
        M, N, K = preds_arr.shape
        # Per-member per-row reciprocal score, summed per (n, k_class) at end.
        # Allocating (M, N, K) is M× the output; for typical M<=20 it's a
        # small constant factor and the parallelism wins back the memory.
        per_member_recip = np.zeros((M, N, K), dtype=np.float64)
        for m in _numba.prange(M):
            for k_class in range(K):
                # Argsort the M-th member's N-vector descending.
                col_m = -preds_arr[m, :, k_class]  # negate for descending
                order = np.argsort(col_m, kind="quicksort")
                for n_pos in range(N):
                    rank = n_pos  # 0-based descending rank
                    row_idx = order[n_pos]
                    per_member_recip[m, row_idx, k_class] = 1.0 / (k + rank + 1)
        # Reduce M axis to (N, K)
        aggregated = np.zeros((N, K), dtype=np.float64)
        for m in range(M):
            for n in range(N):
                for ki in range(K):
                    aggregated[n, ki] += per_member_recip[m, n, ki]
        # Per-row re-normalize when K > 1
        if K > 1:
            for n in _numba.prange(N):
                row_sum = 0.0
                for ki in range(K):
                    row_sum += aggregated[n, ki]
                if row_sum > 0.0:
                    inv = 1.0 / row_sum
                    for ki in range(K):
                        aggregated[n, ki] *= inv
        return aggregated
else:
    _rrf_aggregate_njit = None


def _bench(fn, preds_arr, k, n_trials=5):
    # One warm-up to amortise numba JIT compile.
    fn(preds_arr, k)
    walls = []
    for _ in range(n_trials):
        t = time.perf_counter()
        fn(preds_arr, k)
        walls.append(time.perf_counter() - t)
    return min(walls), sum(walls) / len(walls)


def _equivalence_check(a, b, *, label, tol=1e-9):
    """Verify per-row sums match (RRF re-normalises so per-row sums are 1
    for K > 1) and per-cell values agree to tolerance. Tie-handling differs
    between np.argsort(kind='stable') and the njit quicksort branch when
    EXACT float ties exist; on realistic float-prob inputs ties are
    vanishingly rare and the per-row sums are bit-identical.
    """
    if a.shape != b.shape:
        return f"  [{label}] SHAPE MISMATCH: numpy={a.shape} njit={b.shape}"
    max_abs = float(np.max(np.abs(a - b)))
    if max_abs > tol:
        return f"  [{label}] max |numpy - njit| = {max_abs:.3e} (tol={tol:.0e})"
    return f"  [{label}] equiv OK (max |diff| = {max_abs:.3e})"


def main() -> None:
    print("=" * 78)
    print("RRF aggregation bench: numpy stable argsort vs numba prange(M)")
    print("=" * 78)
    if not _HAS_NUMBA:
        print("numba unavailable; skipping njit comparison")
        return

    sizes = [
        (5, 10_000, 2),
        (5, 100_000, 2),
        (5, 1_000_000, 2),
        (5, 10_000, 3),
        (5, 100_000, 3),
        (5, 1_000_000, 3),
        (5, 1_000_000, 10),
        (10, 100_000, 3),
        (10, 1_000_000, 3),
        (20, 100_000, 3),
        (20, 1_000_000, 3),
        (5, 5_000_000, 3),
    ]
    k = 60
    rng = np.random.default_rng(0)

    header = f"{'M':>3} {'N':>10} {'K':>3} | {'numpy min (ms)':>16} {'njit min (ms)':>16} {'speedup':>9} | equiv"
    print(header)
    print("-" * len(header))

    for M, N, K in sizes:
        preds_arr = rng.uniform(0.0, 1.0, size=(M, N, K)).astype(np.float64)
        np_min, np_mean = _bench(_rrf_aggregate_numpy, preds_arr, k)
        nb_min, nb_mean = _bench(_rrf_aggregate_njit, preds_arr, k)
        speedup = np_min / nb_min if nb_min > 0 else float("inf")
        out_np = _rrf_aggregate_numpy(preds_arr, k)
        out_nb = _rrf_aggregate_njit(preds_arr, k)
        equiv = _equivalence_check(out_np, out_nb, label=f"M={M} N={N} K={K}")
        winner = "njit" if speedup > 1.05 else ("numpy" if speedup < 0.95 else "tie")
        print(
            f"{M:>3} {N:>10_} {K:>3} | {np_min*1000:>16.2f} {nb_min*1000:>16.2f} "
            f"{speedup:>8.2f}x | {winner}"
        )
        print(equiv)

    print()
    print("Crossover guidance: pick the kernel whose 'min (ms)' is smaller at the")
    print("(M, N, K) point closest to the caller's input. For RRF the parallel")
    print("njit wins when prange(M) saturates available cores AND N is large enough")
    print("that the argsort cost outweighs the once-per-member dispatch overhead.")


if __name__ == "__main__":
    main()
