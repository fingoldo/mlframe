"""Times the whole batch_pair_mi_cuda call (best-of-N wall, warm) at n = 200k, 1M, 2M; run once per checkout (cwd = its root) to compare old and new code."""

import sys
import time

import numpy as np

from mlframe.feature_selection.filters import _batch_pair_mi_cuda_kernels as K


def main() -> None:
    """Print the best call wall per size, alternating two blocks of five calls."""
    rng = np.random.default_rng(0)
    nb, nf, npairs = 8, 32, 64
    for n in (200_000, 1_000_000, 2_000_000):
        f = rng.integers(0, nb, (n, nf)).astype(np.int32)
        y = rng.integers(0, 4, n).astype(np.int32)
        fy = np.bincount(y, minlength=4) / n
        nbins = np.full(nf, nb, dtype=np.int32)
        pa = rng.integers(0, nf, npairs).astype(np.int64)
        pb = ((pa + rng.integers(1, nf, npairs)) % nf).astype(np.int64)
        K.batch_pair_mi_cuda(f, pa, pb, nbins, y, fy)
        best = float("inf")
        for _ in range(12):
            t = time.perf_counter()
            K.batch_pair_mi_cuda(f, pa, pb, nbins, y, fy)
            best = min(best, time.perf_counter() - t)
        print(sys.argv[1], n, f"{best * 1e3:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
