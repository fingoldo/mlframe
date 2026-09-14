"""Benchmark: inverting a permutation with ``np.argsort`` versus an O(n) scatter.

The FE permutation-null loops (``_integer_lattice_fe``, ``_conditional_gate_fe``, ``_pairwise_modular_resident``) used to
build each shuffled column as ``feat[np.argsort(perm)]`` -- an O(n log n) sort, run 12 times per candidate. For a
duplicate-free permutation ``out[perm] = feat`` produces the identical array in O(n). This measures the op on its own and
inside the 12-permutation matrix build it sits in, which also pays the unchanged ``rng.permutation`` cost and is therefore
the honest end-to-end number. Every build is asserted bit-identical between the two forms.

Measured when this landed (warm, paired, interleaved, quiet machine):
    op 16.5-18.6x faster; 12-permutation build 1.89x at n=10k, 2.43x at 200k, 2.59x at 1M, 2.91x at 5M.

Run: ``python -m mlframe.feature_selection.filters._benchmarks.bench_perm_inverse_scatter``
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

_SIZES: tuple[int, ...] = (10_000, 200_000, 1_000_000, 5_000_000)
_N_PERM = 12


def _best_of(fn: Callable[[], object], reps: int) -> float:
    """Minimum wall time of ``reps`` calls to ``fn``, in seconds."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _build(feat: np.ndarray, n_perm: int, *, scatter: bool) -> np.ndarray:
    """Build the ``(n, n_perm)`` shuffled-feature matrix exactly as the FE null loops do, in either form."""
    n = feat.shape[0]
    rng = np.random.default_rng(1)
    mat = np.empty((n, n_perm), dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(n)
        if scatter:
            mat[perm, i] = feat
        else:
            mat[:, i] = feat[np.argsort(perm)]
    return mat


def run(sizes: tuple[int, ...] = _SIZES) -> list[dict[str, float]]:
    """Time both forms at each size and return one result row per size; raises if any build is not bit-identical."""
    rows: list[dict[str, float]] = []
    for n in sizes:
        rng = np.random.default_rng(0)
        feat = rng.normal(size=n)
        perm = rng.permutation(n)
        out = np.empty(n, dtype=np.float64)
        reps = 20 if n <= 200_000 else 5

        def op_argsort() -> np.ndarray:
            """The original gather-by-argsort inverse."""
            return feat[np.argsort(perm)]

        def op_scatter() -> None:
            """The O(n) scatter inverse."""
            out[perm] = feat

        for _ in range(2):  # warm both paths before timing
            op_argsort()
            op_scatter()
        t_sort = t_scat = float("inf")
        for _ in range(3):  # interleave so drift and contention hit both arms equally
            t_sort = min(t_sort, _best_of(op_argsort, reps))
            t_scat = min(t_scat, _best_of(op_scatter, reps))

        if not np.array_equal(_build(feat, _N_PERM, scatter=False), _build(feat, _N_PERM, scatter=True)):
            raise AssertionError(f"scatter build is not bit-identical to the argsort build at n={n}")
        b_sort = b_scat = float("inf")
        for _ in range(3 if n <= 1_000_000 else 2):
            b_sort = min(b_sort, _best_of(lambda: _build(feat, _N_PERM, scatter=False), 1))
            b_scat = min(b_scat, _best_of(lambda: _build(feat, _N_PERM, scatter=True), 1))

        rows.append({"n": float(n), "op_argsort_s": t_sort, "op_scatter_s": t_scat, "build_argsort_s": b_sort, "build_scatter_s": b_scat})
    return rows


def main() -> None:
    """Print the benchmark table."""
    print(f"{'n':>10} | {'op argsort':>11} {'op scatter':>11} {'op x':>6} | {'build argsort':>14} {'build scatter':>14} {'build x':>8}")
    for r in run():
        print(
            f"{int(r['n']):>10,} | {r['op_argsort_s'] * 1e3:8.2f} ms {r['op_scatter_s'] * 1e3:8.2f} ms {r['op_argsort_s'] / r['op_scatter_s']:5.1f}x | "
            f"{r['build_argsort_s'] * 1e3:11.1f} ms {r['build_scatter_s'] * 1e3:11.1f} ms {r['build_argsort_s'] / r['build_scatter_s']:7.2f}x"
        )


if __name__ == "__main__":
    main()
