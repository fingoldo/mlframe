"""cProfile harness for pseudo-BMA composite-ensemble weighting (``composite/_pseudo_bma.py``).

Run: ``python -m mlframe.training.composite._benchmarks.bench_pseudo_bma``

Representative shape: n=2000 OOF rows, K=5 components, bb_draws=1000 (the BB path is the expensive one). Top hotspots are
the ``(bb_draws, n)`` gamma draw (numpy RNG C loop) and the ``(bb_draws, n) @ (n, K)`` matmul (BLAS); the per-draw softmax
is a fused ``numba.njit`` reduction. Neither the RNG nor BLAS has an actionable Python-level speedup -- documented as
``no actionable speedup`` in the module docstring. The point path (bb_draws=0) is a single vectorised Gaussian-lpd pass.
"""
from __future__ import annotations

import cProfile
import pstats

import numpy as np

from mlframe.training.composite._pseudo_bma import blend, pseudo_bma_weights


def _data(n=2000, K=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 2.0 * x + 1.0
    cols = [y + rng.normal(scale=0.2 + 0.5 * k, size=n) for k in range(K)]
    return np.column_stack(cols), y


def main():
    P, y = _data()
    # Warm numba njit softmax so JIT compile is not attributed to the profiled region.
    pseudo_bma_weights(P, y, bb_draws=4, random_state=0)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        w = pseudo_bma_weights(P, y, bb_draws=1000, random_state=1)
        _ = blend(P, w)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(15)


if __name__ == "__main__":
    main()
