"""Bench whether parallelising the outer per-group loop in
``compute_grouped_group_aucs`` is worthwhile.

The current impl is sequential numba: builds a typed Dict, scans the
sorted-by-group array, computes AUC per group via
``fast_numba_aucs_simple``. Each group is independent, so prange across
groups should win when (a) many valid groups, (b) groups are large
enough that per-iteration work amortises the per-thread spawn.

Decision rule: ship a _par variant if speedup >= 2x at production-typical
shapes (~22k valid groups after the Fix-5 NaN filter, ~5-50 samples per
group).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import numba
from numba import prange

sys.path.insert(0, ".")

from mlframe.metrics.core import (  # noqa: E402
    compute_grouped_group_aucs,
    fast_numba_aucs_simple,
    NUMBA_NJIT_PARAMS,
)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _grouped_group_aucs_par_arrays(
    sorted_group_ids: np.ndarray,
    sorted_y_true: np.ndarray,
    sorted_y_score: np.ndarray,
    boundaries: np.ndarray,  # length n_groups+1
    out_group_ids: np.ndarray,
    out_rocs: np.ndarray,
    out_prs: np.ndarray,
):
    """Parallel inner loop. Caller pre-computes group boundaries and
    output arrays. Each thread writes to its own slot — no contention."""
    n_groups = len(boundaries) - 1
    for g in prange(n_groups):
        s = boundaries[g]
        e = boundaries[g + 1]
        out_group_ids[g] = sorted_group_ids[s]
        if e - s <= 1:
            out_rocs[g] = np.nan
            out_prs[g] = np.nan
            continue
        gy = sorted_y_true[s:e]
        gp = sorted_y_score[s:e]
        order = np.argsort(gp)[::-1]
        roc, pr = fast_numba_aucs_simple(gy, gp, order)
        out_rocs[g] = roc
        out_prs[g] = pr


def _build_boundaries(sorted_group_ids: np.ndarray) -> np.ndarray:
    """Return boundaries[i:i+1] = (start, end) indices for each group.
    O(N) sequential pass."""
    n = len(sorted_group_ids)
    if n == 0:
        return np.array([0], dtype=np.int64)
    bnd = [0]
    for i in range(1, n):
        if sorted_group_ids[i] != sorted_group_ids[i - 1]:
            bnd.append(i)
    bnd.append(n)
    return np.asarray(bnd, dtype=np.int64)


def parallel_grouped_group_aucs(
    sorted_group_ids: np.ndarray,
    sorted_y_true: np.ndarray,
    sorted_y_score: np.ndarray,
):
    """Public driver: build boundaries + dispatch parallel + assemble dict."""
    boundaries = _build_boundaries(sorted_group_ids)
    n_groups = len(boundaries) - 1
    out_group_ids = np.empty(n_groups, dtype=np.int64)
    out_rocs = np.empty(n_groups, dtype=np.float64)
    out_prs = np.empty(n_groups, dtype=np.float64)
    _grouped_group_aucs_par_arrays(
        sorted_group_ids, sorted_y_true, sorted_y_score,
        boundaries, out_group_ids, out_rocs, out_prs,
    )
    return {int(out_group_ids[i]): (out_rocs[i], out_prs[i]) for i in range(n_groups)}


def make_data(n_groups, samples_per_group, rng):
    """Build sorted (group_ids, y_true, y_score) arrays.

    Each group has ~samples_per_group rows (Poisson-distributed for
    realism), random binary labels with ~50% positive rate.
    """
    sizes = rng.poisson(samples_per_group, n_groups)
    sizes = np.maximum(sizes, 1)  # >= 1 each
    n_total = int(sizes.sum())
    group_ids = np.repeat(np.arange(n_groups, dtype=np.int64), sizes)
    y_true = (rng.standard_normal(n_total) > 0).astype(np.int8)
    y_score = rng.standard_normal(n_total)
    return group_ids, y_true, y_score


def time_op(fn, *args, repeats=3, warmup=1):
    for _ in range(warmup):
        fn(*args)
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        t.append(time.perf_counter() - t0)
    return out, min(t)


def fmt(t):
    if t < 1e-3:
        return f"{t*1e6:7.1f}us"
    if t < 1.0:
        return f"{t*1e3:7.2f}ms"
    return f"{t:7.3f}s"


def main():
    rng = np.random.default_rng(0)
    print(f"numba: {numba.__version__}, num_threads={numba.get_num_threads()}\n")
    print(f"{'n_groups':>10} {'avg_size':>9} | {'seq':>10} | {'par':>10} | {'par/seq':>8}")
    print("-" * 60)
    cases = [
        (100, 10),       # tiny
        (1_000, 10),     # small
        (10_000, 10),    # medium
        (22_000, 5),     # production-ish (post Fix-5 filter)
        (100_000, 10),   # large
        (10_000, 50),    # fewer-bigger
    ]
    for n_groups, avg_size in cases:
        gids, yt, ys = make_data(n_groups, avg_size, rng)
        compute_grouped_group_aucs(gids, yt, ys)  # warm seq
        parallel_grouped_group_aucs(gids, yt, ys)  # warm par

        seq_out, t_seq = time_op(compute_grouped_group_aucs, gids, yt, ys)
        par_out, t_par = time_op(parallel_grouped_group_aucs, gids, yt, ys)

        # Equivalence: same keys, same values (within fp tol).
        seq_keys = set(seq_out.keys())
        par_keys = set(par_out.keys())
        assert seq_keys == par_keys, f"keys diverge: {len(seq_keys)} vs {len(par_keys)}"
        max_err = 0.0
        for k in seq_keys:
            sr, sp = seq_out[k]
            pr, pp = par_out[k]
            for a, b in [(sr, pr), (sp, pp)]:
                if np.isnan(a) and np.isnan(b):
                    continue
                max_err = max(max_err, abs(a - b))

        print(f"{n_groups:>10} {avg_size:>9} | {fmt(t_seq):>10} | {fmt(t_par):>10} | "
              f"{t_par/t_seq:7.2f}x  err={max_err:.2e}")


if __name__ == "__main__":
    main()
