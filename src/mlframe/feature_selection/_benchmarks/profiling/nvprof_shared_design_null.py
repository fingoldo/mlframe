"""nvprof driver of the gradient seeder's permutation null on one shared random-feature design (``_rff_gram_energy_gpu.rff_gram_shared_design_maxima_device``) against the per-surrogate null.

Run from a shell that has the CUPTI directory on PATH::

    nvprof --profile-from-start off --print-gpu-summary python nvprof_shared_design_null.py shared
    nvprof --profile-from-start off --print-gpu-summary python nvprof_shared_design_null.py stacked

Only the profiled window (cupy ``profiler.start`` / ``stop``) is counted: the null of 12 permutation surrogates at 2000 rows, a 30-column pool (435 pairs) and 400 random features.
"""

from __future__ import annotations

import sys
import time

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _gradient_interaction_seeder as seeder
from mlframe.feature_selection.filters._linear_probes import RffSampler
from mlframe.feature_selection.filters._rff_gram_energy_gpu import rff_gram_null_maxima_device


def main(mode: str) -> None:
    """Run the null ``mode`` ('shared' or 'stacked') once warm, then once inside the profiler window, and print the wall of the profiled call."""
    rng = np.random.default_rng(0)
    n, p, d, k = 2000, 30, 400, 12
    X = rng.standard_normal((n, p))
    y = X[:, 0] * X[:, 1] + 0.3 * rng.standard_normal(n)
    pairs = [(a, b) for a in range(p) for b in range(a + 1, p)]
    rbf = RffSampler(gamma=1.0 / p, n_components=d, random_state=0).fit(X)

    def shared() -> None:
        """The null on the observed design."""
        seeder._shared_design_null_maxima(rbf, X, y, pairs, k, 1.0, 0)

    def stacked() -> None:
        """The null with one design per surrogate (the previous route)."""
        fits = [seeder._fit_rff_ridge(X, np.random.default_rng(1000 + s).permutation(y), d, 1.0 / p, 1.0, s) for s in range(k)]
        rff_gram_null_maxima_device([f[0] for f in fits], [f[1] for f in fits], X, pairs)

    run = shared if mode == "shared" else stacked
    run()
    cp.cuda.Device().synchronize()
    cp.cuda.profiler.start()
    t0 = time.perf_counter()
    run()
    cp.cuda.Device().synchronize()
    wall = time.perf_counter() - t0
    cp.cuda.profiler.stop()
    print(f"MODE {mode} wall {wall * 1e3:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "shared")
