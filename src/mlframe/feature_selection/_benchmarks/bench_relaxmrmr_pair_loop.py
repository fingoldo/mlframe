"""Bench the RelaxMRMR 3-way interaction term: the per-pair Python loop against the fused parallel kernel.

The term sums a co-information over every pair of selected features, so its cost grows as ``|S|^2`` and it is paid once per candidate per
greedy round. Run from the repository root::

    python -m mlframe.feature_selection._benchmarks.bench_relaxmrmr_pair_loop

Sweeps n and |S| because the crossover is the whole question: the parallel kernel has thread-launch overhead that a tiny table cannot repay,
while at production width the per-pair interpreter round trips and the per-pair n-length composite allocation dominate.
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters._relaxmrmr_kernels import _cmi_mm_njit, _composite_codes_njit, _mi_mm_njit
from mlframe.feature_selection.filters._relaxmrmr_pair_loop import pair_interaction_sum

_N_GRID = (2_000, 50_000, 500_000, 2_000_000)
_S_GRID = (5, 20, 50)


def _serial_pair_sum(x, y, sel, K_sel, K_x, K_y, cmi_y_mm, marg_mm, min_rows_per_cell) -> float:
    """The per-pair Python loop the kernel replaced."""
    n_S = len(sel)
    n_rows = float(x.shape[0])
    inter = 0.0
    for i in range(n_S):
        for j in range(i + 1, n_S):
            K_i, K_j = K_sel[i], K_sel[j]
            if n_rows < float(min_rows_per_cell) * K_x * K_i * K_j * K_y:
                continue
            z_pair = _composite_codes_njit(sel[i], sel[j], K_j)
            cmi_ij = _cmi_mm_njit(x, z_pair, y, K_x, K_i * K_j, K_y)
            mi_x_zz = _mi_mm_njit(x, z_pair, K_x, K_i * K_j)
            inter += (cmi_y_mm[i] + cmi_y_mm[j] - cmi_ij) - (marg_mm[i] + marg_mm[j] - mi_x_zz)
    return inter


def _case(n: int, n_S: int, seed: int = 0) -> tuple:
    """A candidate, a target, and ``n_S`` selected columns with the per-column marginals the term needs."""
    rng = np.random.default_rng(seed)
    K_x, K_y = 4, 3
    x = rng.integers(0, K_x, size=n).astype(np.int64)
    y = rng.integers(0, K_y, size=n).astype(np.int64)
    K_sel = [2 + (i % 3) for i in range(n_S)]
    sel = [rng.integers(0, K_sel[i], size=n).astype(np.int64) for i in range(n_S)]
    marg = np.array([_mi_mm_njit(x, sel[i], K_x, K_sel[i]) for i in range(n_S)])
    cmi_y = np.array([_cmi_mm_njit(x, sel[i], y, K_x, K_sel[i], K_y) for i in range(n_S)])
    return x, y, sel, K_sel, K_x, K_y, cmi_y, marg


def _best(fn, repeat: int) -> float:
    """Best-of-``repeat`` wall seconds, warmed once so no run pays compilation."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    """Run the sweep and print a speedup table with the agreement of the two forms."""
    print(f"{'n':>10} {'|S|':>5} {'pairs':>7} {'serial s':>10} {'fused s':>10} {'speedup':>8} {'rel diff':>10}")
    for n in _N_GRID:
        for n_S in _S_GRID:
            x, y, sel, K_sel, K_x, K_y, cmi_y, marg = _case(n, n_S)
            repeat = 5 if n <= 50_000 else 3
            t_serial = _best(lambda: _serial_pair_sum(x, y, sel, K_sel, K_x, K_y, cmi_y, marg, 1.0), repeat)
            t_fused = _best(lambda: pair_interaction_sum(x, y, sel, K_sel, K_x, K_y, cmi_y, marg, 1.0), repeat)
            got = pair_interaction_sum(x, y, sel, K_sel, K_x, K_y, cmi_y, marg, 1.0)
            want = _serial_pair_sum(x, y, sel, K_sel, K_x, K_y, cmi_y, marg, 1.0)
            rel = abs(got - want) / max(abs(want), 1e-12)
            pairs = n_S * (n_S - 1) // 2
            print(f"{n:>10} {n_S:>5} {pairs:>7} {t_serial:>10.4f} {t_fused:>10.4f} {t_serial / t_fused:>7.2f}x {rel:>10.2e}")


if __name__ == "__main__":
    main()
