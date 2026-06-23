"""Bench: non-finite cell count in transformer/_utils.validate_numeric_input.

OLD: int(np.count_nonzero(~np.isfinite(X))) -- allocates TWO full N*d temporaries
(the isfinite bool array AND its bitwise-inverse) just to count bad cells, then
walks the array twice. NEW: a numba njit fused single-pass counter (no temporary,
size-dispatched serial vs prange).

validate_numeric_input runs on EVERY transformer FE call over the FULL X (which can
be 100+ GB), so removing the two N*d bool temporaries matters for both speed and peak RAM.

Run: CUDA_VISIBLE_DEVICES="" python bench_validate_nonfinite_count.py
"""
from __future__ import annotations

import time

import numpy as np

from mlframe.feature_engineering.transformer._utils import _count_nonfinite_cells


def _old(X: np.ndarray) -> int:
    return int(np.count_nonzero(~np.isfinite(X)))


def main() -> None:
    rng = np.random.default_rng(0)
    # warm numba
    _count_nonfinite_cells(rng.standard_normal((100, 5)).astype(np.float32))

    for shape in [(50_000, 50), (100_000, 100), (200_000, 200), (1_000_000, 50)]:
        X = rng.standard_normal(shape).astype(np.float32)
        # inject a few non-finite to exercise the count path
        X[3, 4] = np.nan
        X[7, 1] = np.inf
        assert _old(X) == _count_nonfinite_cells(X) == 2
        res = {}
        for f, nm in [(_old, "old"), (_count_nonfinite_cells, "njit")]:
            ts = []
            for _ in range(9):
                t = time.perf_counter()
                f(X)
                ts.append(time.perf_counter() - t)
            res[nm] = min(ts) * 1000.0
        print(f"{shape}: old={res['old']:.2f}ms njit={res['njit']:.2f}ms  speedup={res['old']/res['njit']:.2f}x")


if __name__ == "__main__":
    main()
