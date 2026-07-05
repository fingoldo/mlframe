"""A/B bench for the ``_CMI_PARALLEL_MIN_CANDS`` crossover (serial vs prange CPU CMI loop).

Context (2026-07, wellbore full-fit investigation): ``_cpu_cmi_loop`` routes to the serial hoisted loop
when the candidate count ``p < _CMI_PARALLEL_MIN_CANDS`` (was 32), else to the prange loop. Once the Q1
screen SUBSAMPLE engages, every candidate's exact ``conditional_mi`` runs on ~30k rows -- work that dwarfs
the ~50us thread spawn -- so prange wins even for small pools. This bench sweeps ``p`` at several ``n`` to
locate the crossover.

Measured (best-of-7, this host, nbins=20):
  n=30000: prange wins ALL p>=4 (2.4x-7.8x).
  n=5000 : prange wins ALL p>=4 (2.5x-3.9x).
  n=1000 : prange wins p>=6 (1.4x-8.7x); p=4 marginal serial win (0.85x, ~0.06ms).
=> threshold lowered 32 -> 8 (universal win at p>=8; captures the bulk at the 30k screen size). Env override
   MLFRAME_CMI_PARALLEL_MIN_CANDS. Both branches are exact CMI -> selection-equivalent (numeric-order only).

Run: python -m mlframe.feature_selection.filters.info_theory._benchmarks.bench_cmi_parallel_threshold [n]
"""
import os, sys, time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from mlframe.feature_selection.filters.info_theory._cmi_cuda import (  # noqa: E402
    _cpu_cmi_loop_hoisted_parallel,
    _cpu_cmi_loop_hoisted_serial,
)


def _make(n, P, nbins=20, seed=0):
    rng = np.random.default_rng(seed)
    cols = P + 8
    fd = rng.integers(0, nbins, size=(n, cols)).astype(np.int32)
    nb = np.full(cols, nbins, dtype=np.int64)
    cand = np.arange(P, dtype=np.int64)
    y = np.array([cols - 1], dtype=np.int64)
    z = np.array([cols - 2], dtype=np.int64)
    return fd, cand, y, z, nb


def _best_of(fn, args, reps=7):
    fn(*args); fn(*args)  # warm the njit compile + cache
    return min(_timed(fn, args) for _ in range(reps))


def _timed(fn, args):
    t = time.perf_counter(); fn(*args); return time.perf_counter() - t


def main(n):
    print(f"n={n}  serial vs prange (best-of-7, ms)")
    print(f"{'p':>4} {'serial_ms':>10} {'par_ms':>10} {'speedup':>8} {'winner':>8}")
    for P in [4, 6, 8, 10, 12, 16, 20, 24, 28, 31, 32, 40, 48, 64]:
        fd, cand, y, z, nb = _make(n, P)
        s = _best_of(_cpu_cmi_loop_hoisted_serial, (fd, cand, y, z, nb)) * 1e3
        p = _best_of(_cpu_cmi_loop_hoisted_parallel, (fd, cand, y, z, nb)) * 1e3
        print(f"{P:>4} {s:>10.3f} {p:>10.3f} {s / p:>8.2f} {'PAR' if p < s else 'ser':>8}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30000)
