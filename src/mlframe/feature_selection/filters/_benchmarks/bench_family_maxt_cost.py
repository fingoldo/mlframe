"""Wall-clock cost of one exact family max-T pass (audit 2026-10-14, F02).

Times ``family_maxt_pass_codes`` on synthetic noise at a few (n_rows, K) sizes, best of three after a warm-up call.
Run: ``python -m mlframe.feature_selection.filters._benchmarks.bench_family_maxt_cost``.
"""

import time

import numpy as np

from mlframe.feature_selection.filters._family_maxt_gate import family_maxt_pass_codes

SIZES = [(2000, 200), (2000, 1734), (20000, 200)]


def main() -> None:
    """Print the best wall-clock time of one family pass per size."""
    rng = np.random.default_rng(0)
    for n, k in SIZES:
        cols = rng.integers(0, 8, (k, n)).astype(np.int64)
        nx = np.full(k, 8, dtype=np.int64)
        y = rng.integers(0, 4, n).astype(np.int64)
        family_maxt_pass_codes(cols, nx, y, 0.01, 20, 0)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            family_maxt_pass_codes(cols, nx, y, 0.01, 20, 0)
            best = min(best, time.perf_counter() - t0)
        print(f"n={n} K={k} perms=20 best={best:.3f}s", flush=True)


if __name__ == "__main__":
    main()
