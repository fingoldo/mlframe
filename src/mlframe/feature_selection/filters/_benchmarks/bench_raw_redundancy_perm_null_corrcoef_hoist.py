"""Bench: hoist the invariant correlation denominator out of the permutation
null loop in ``raw_retains_linear_signal_given_children``.

The permutation-null leg shuffles the raw residual ``rx`` ``nperm`` times and
scores ``abs(corr(perm, ry))`` against the FIXED residualized target ``ry``.
The legacy inner call ``np.corrcoef(perm, ry)`` rebuilds the full 2x2 matrix
every iteration, recomputing ``mean(ry)``, ``std(ry)``, ``std(perm)`` and
``mean(perm)`` -- all loop-invariant: ``perm`` is a reordering of ``rx`` so its
mean/std never change, and ``ry`` is fixed. The only varying quantity is the
cross term ``perm @ (ry - mean(ry))``.

Hoisting the centred target ``ryc = ry - mean(ry)`` and the constant denominator
``n * std(rx) * std(ry)`` out of the loop reduces the inner work to a single
dot product. Bit-identical to ``np.corrcoef`` up to FP reduction order
(max abs diff ~1e-17, single-ULP, selection-safe).

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters._benchmarks.bench_raw_redundancy_perm_null_corrcoef_hoist
"""
from __future__ import annotations

import time

import numpy as np


def _old(rx: np.ndarray, ry: np.ndarray, nperm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    null = np.empty(int(nperm), dtype=np.float64)
    for k in range(int(nperm)):
        perm = rng.permutation(rx)
        null[k] = abs(float(np.corrcoef(perm, ry)[0, 1]))
    return null


def _new(rx: np.ndarray, ry: np.ndarray, nperm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rx.shape[0]
    ryc = ry - ry.mean()
    denom = n * float(np.std(rx)) * float(np.std(ry))
    null = np.empty(int(nperm), dtype=np.float64)
    for k in range(int(nperm)):
        perm = rng.permutation(rx)
        null[k] = abs(float(perm @ ryc) / denom) if denom > 0.0 else 0.0
    return null


def _best(fn, *args, reps: int = 9) -> float:
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t)
    return min(ts)


def main() -> None:
    rng = np.random.default_rng(0)
    nperm = 32
    for n in (2000, 10000, 100000):
        rx = rng.standard_normal(n)
        ry = rng.standard_normal(n)
        o = _old(rx, ry, nperm, 123)
        ne = _new(rx, ry, nperm, 123)
        md = float(np.max(np.abs(o - ne)))
        to = _best(_old, rx, ry, nperm, 123)
        tn = _best(_new, rx, ry, nperm, 123)
        print(f"n={n:>7}: old={to*1e3:7.3f}ms new={tn*1e3:7.3f}ms " f"speedup={to/tn:4.2f}x max_abs_diff={md:.2e}")


if __name__ == "__main__":
    main()
