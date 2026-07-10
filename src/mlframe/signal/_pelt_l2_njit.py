"""njit PELT (Killick et al.) specialized for the l2 (mean-shift) cost, using cumulative sums.

``ruptures``' own l2 cost function recomputes ``np.var()`` over each candidate window from scratch, so its
PELT driver costs O(1) numpy calls but each call is O(window) -- no cumsum reuse. Segment sum-of-squared-
deviations is additive under prefix sums (``cost(a, b) = SS[b]-SS[a] - (S[b]-S[a])**2/(b-a)``), so a candidate
cost is O(1) here, and the whole PELT pruning loop runs inside one njit function with no per-step Python/numpy
dispatch overhead. Measured ~13x faster than ``ruptures.Pelt(model="l2")`` at n=50,000 with bit-identical
breakpoints (`_benchmarks/bench_changepoint_detection.py`).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _pelt_l2_kernel(y: np.ndarray, min_size: int, penalty: float) -> np.ndarray:
    n = y.shape[0]
    S = np.zeros(n + 1)
    SS = np.zeros(n + 1)
    for i in range(n):
        S[i + 1] = S[i] + y[i]
        SS[i + 1] = SS[i] + y[i] * y[i]

    F = np.zeros(n + 1)
    F[0] = -penalty
    cp = np.zeros(n + 1, dtype=np.int64)

    R = np.zeros(n + 1, dtype=np.int64)
    R_size = 1  # R[0] = 0 already

    for t in range(min_size, n + 1):
        best_cost = np.inf
        best_s = 0
        for idx in range(R_size):
            s = R[idx]
            if t - s < min_size:
                continue
            seg_sum = S[t] - S[s]
            seg_ss = SS[t] - SS[s]
            length = t - s
            c = seg_ss - seg_sum * seg_sum / length
            total = F[s] + c + penalty
            if total < best_cost:
                best_cost = total
                best_s = s
        F[t] = best_cost
        cp[t] = best_s

        new_R_size = 0
        for idx in range(R_size):
            s = R[idx]
            seg_sum = S[t] - S[s]
            seg_ss = SS[t] - SS[s]
            length = t - s
            c = seg_ss - seg_sum * seg_sum / length
            if F[s] + c <= F[t]:
                R[new_R_size] = s
                new_R_size += 1
        R[new_R_size] = t
        R_size = new_R_size + 1

    bps = np.zeros(n, dtype=np.int64)
    n_bps = 0
    t = n
    while t > 0:
        s = int(cp[t])
        if s > 0:
            bps[n_bps] = s
            n_bps += 1
        t = s
    return bps[:n_bps][::-1]


def pelt_l2(y: np.ndarray, min_size: int, penalty: float) -> list:
    """Return sorted changepoint indices for the l2 (mean-shift) cost model, PELT-exact."""
    bps: np.ndarray = _pelt_l2_kernel(np.ascontiguousarray(y, dtype=np.float64), min_size, float(penalty))
    return list(bps)


__all__ = ["pelt_l2"]
