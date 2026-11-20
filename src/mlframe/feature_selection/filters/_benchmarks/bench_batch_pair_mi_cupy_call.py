"""Times the whole batch_pair_mi_cupy call (best-of-N wall, warm) at n = 200k and 1M; run once per checkout (cwd = its root) to compare old and new code."""

import sys
import time

import numpy as np

from mlframe.feature_selection.filters import batch_pair_mi_gpu as g


def main() -> None:
    """Print the best call wall per size."""
    rng = np.random.default_rng(0)
    for n in (200_000, 1_000_000, 2_000_000):
        f = rng.integers(0, 8, (n, 32)).astype(np.int32)
        y = rng.integers(0, 4, n).astype(np.int32)
        fy = np.bincount(y) / n
        nb = np.full(32, 8, dtype=np.int32)
        pa = rng.integers(0, 32, 64)
        pb = (pa + 1) % 32
        g.batch_pair_mi_cupy(f, pa, pb, nb, y, fy)
        best = float("inf")
        for _ in range(8):
            t = time.perf_counter()
            g.batch_pair_mi_cupy(f, pa, pb, nb, y, fy)
            best = min(best, time.perf_counter() - t)
        print(sys.argv[1], n, f"{best * 1e3:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
