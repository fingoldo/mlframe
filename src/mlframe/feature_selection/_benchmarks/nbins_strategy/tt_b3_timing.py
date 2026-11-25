"""Determinism, vectorisation and wall-time check of ``recommend_target_transform`` (agent B3, 2026-11-24).

Usage: ``CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python tt_b3_timing.py``. Prints: identical output for the same seed (twice), the time per call for (n, p, head, n_null) and the time split of one call
(analysis = LL for all candidates on the data, null = sign-flip draws, gate = held-out original-scale errors).
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tt_b3_fixtures as F
import tt_b3_reference as R


def main() -> None:
    X, y = F.make("lognormal", 4000, 5, 3)
    a = R.recommend_target_transform(X, y, seed=7)
    b = R.recommend_target_transform(X, y, seed=7)
    same = (a.name, a.params, a.ll_gain, a.null_threshold, a.err_chosen) == (b.name, b.params, b.ll_gain, b.null_threshold, b.err_chosen)
    print(f"deterministic for a seed: {same}  ({a.name}, gain {a.ll_gain:.4f}, tau {a.null_threshold:.4f})")
    print("n_rows p head n_null  seconds (median of 3)")
    for n, p, head, n_null in ((2000, 5, "linear", 40), (2000, 40, "linear", 40), (15000, 5, "linear", 40), (15000, 40, "linear", 40), (15000, 5, "spline", 40), (15000, 40, "spline", 40), (15000, 5, "linear", 0)):
        X, y = F.make("lognormal", n, p, 1)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            R.recommend_target_transform(X, y, seed=1, head=head, n_null=n_null) if n_null else R.analyse_target_transforms(X, y, seed=1, head=head, n_null=0)
            ts.append(time.perf_counter() - t0)
        print(f"{n:6d} {p:3d} {head:6s} {n_null:3d}  {np.median(ts):.2f}  (min {min(ts):.2f}, max {max(ts):.2f})")


if __name__ == "__main__":
    main()
