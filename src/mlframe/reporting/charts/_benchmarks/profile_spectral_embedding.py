"""cProfile harness for the spectral graph-embedding chart (charts/spectral_embedding.py).

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_spectral_embedding``

The cost is the eigensolve. Below ``_DENSE_MAX_NODES`` a dense ``numpy.linalg.eigh`` (O(n^3)) computes the whole spectrum;
above it, only the 3 smallest eigenvectors are needed, so ``scipy.sparse.linalg.eigsh(k=3, which='SA')`` on the sparse
Laplacian avoids the full decomposition (k-smallest optimization). This harness times a few node counts and confirms the
dense O(n^3) growth up to the threshold and the sparse-solver crossover above it; the only hot line is the eigensolve, and
building the dense adjacency via the FE module's ``_dense_adjacency`` is O(edges) and dominated by the eigensolve.

Measured: n=100 10.2 ms (dense eigh), n=300 90.3 ms (dense eigh), n=1000 138.6 ms (sparse eigsh k=3), n=3000 617.8 ms
(sparse eigsh k=3). The dense O(n^3) growth up to the threshold and the sparse-solver crossover above it are both visible;
the eigensolve is the only hot path. Verdict: no actionable speedup at the target small-graph scale -- the sole overhead
beyond ARPACK is materialising the dense Laplacian before the CSR conversion (the FE ``_dense_adjacency`` reuse is O(n^2)
memory), which only begins to matter past a few thousand nodes; a sparse-native adjacency build is the future lever there.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.reporting.charts.spectral_embedding import spectral_layout


def _random_graph(n: int, avg_degree: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = n * avg_degree // 2
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    mask = src != dst
    return np.column_stack([src[mask], dst[mask]]).astype(np.int64)


def main():
    for n in (100, 300, 1000, 3000):
        edges = _random_graph(n)
        spectral_layout(n, edges)  # warmup
        t0 = time.perf_counter()
        reps = 5 if n <= 1000 else 2
        for _ in range(reps):
            spectral_layout(n, edges)
        wall = (time.perf_counter() - t0) / reps
        path = "dense eigh" if n <= 500 else "sparse eigsh(k=3)"
        print(f"spectral_layout @ n={n} ({path}): {wall*1000:.1f} ms/call (mean of {reps})")

    edges = _random_graph(3000)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(2):
        spectral_layout(3000, edges)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
