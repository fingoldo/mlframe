"""Microbench: fused `joint + c*mult` + refactorize njit kernel vs the
two-step numpy-multiply-add + `_factorize_dense_njit` baseline in
`_renumber_joint`.

`_renumber_joint` is ~23937 calls / 0.434s tottime in the scene-2500 MRMR
profile; the dominant arities are 2-col (`(x,y)`, `(x,z)`, `(y,z)`) and 3-col.
Each extra column does a numpy `joint + c64*mult` (allocates 2 temp arrays:
`c64*mult` and the sum) then a full `_factorize_dense_njit` pass. The fused
kernel folds the multiply-add INTO the factorize loop -- no temporaries, one
walk instead of (multiply + add + walk).

Run:
  PYTHONPATH=<worktree>/src python bench_renumber_joint_combine.py
"""
from __future__ import annotations
import time

import numpy as np

from mlframe.feature_selection.filters._mi_greedy_cmi_fe import (
    _factorize_dense_njit,
    _combine_factorize_njit,
    _renumber_joint,
)


def _baseline_combine(joint, c64, mult):
    joint = joint + c64 * mult
    return _factorize_dense_njit(joint)


def main():
    rng = np.random.default_rng(0)
    for n, nb in ((2500, 10), (1667, 10), (5000, 12)):
        c0 = rng.integers(0, nb, size=n).astype(np.int64)
        c1 = rng.integers(0, nb, size=n).astype(np.int64)
        j0, mult0 = _factorize_dense_njit(np.ascontiguousarray(c0))

        # correctness: fused vs baseline
        ja, ma = _baseline_combine(j0, c1, mult0)
        jb, mb = _combine_factorize_njit(j0, c1, mult0)
        assert ma == mb, (n, ma, mb)
        # induced partition must match (first-seen labels can differ; here both
        # are first-seen so they coincide)
        assert np.array_equal(ja, jb), (n, "partition mismatch")

        # full _renumber_joint bit-identity vs itself (sanity)
        r0, _ = _renumber_joint(c0, c1)
        assert np.array_equal(r0, ja)

        reps = 20000
        t0 = time.perf_counter()
        for _ in range(reps):
            _baseline_combine(j0, c1, mult0)
        t_base = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(reps):
            _combine_factorize_njit(j0, c1, mult0)
        t_fused = time.perf_counter() - t0

        per_b = t_base / reps * 1e6
        per_f = t_fused / reps * 1e6
        print(f"n={n:5d} nb={nb:2d}  baseline={per_b:7.2f}us  fused={per_f:7.2f}us  speedup={per_b/per_f:.2f}x")


if __name__ == "__main__":
    main()
