"""iter104 profile: windowed_shape family @10M. entropy_binned per-window Python loop was the dominant hotspot.

Baseline (pre-fix): rolling_shannon_entropy_binned @1M = 142.9s single-shot (per-window Python loop: one
np.quantile + np.unique + np.histogram + np.log dispatch per window). Vectorized siblings @10M: total_variation
1.35s, quantile_spread 4.67s, zero_crossings 2.71s, n_peaks 1.57s. After the njit prange kernel:
entropy_binned @1M = 0.285s (~670x e2e separate-process), @10M = 2.285s best-of-3. Bin counts bit-identical;
entropy values differ only by ULP-level summation order (global max|delta| = 8.882e-16). RESOLVED.
"""

import sys
import time

import numpy as np

K = 20


def main() -> None:
    """Time the windowed-shape family with cupy blocked for the duration of the run, and put the import table back afterwards."""
    previous = sys.modules.get("cupy", sys.modules)  # the sentinel object means "was not imported"
    sys.modules["cupy"] = None  # type: ignore[assignment]
    try:
        from mlframe.feature_engineering.windowed_shape import (
            rolling_n_peaks,
            rolling_quantile_spread,
            rolling_shannon_entropy_binned,
            rolling_total_variation,
            rolling_zero_crossings,
        )

        rng = np.random.default_rng(0)

        def mk(n: int) -> tuple:
            """Returns ``(rng.standard_normal(n).astype(np.float64), (np.arange(n) // 1000000).astype(np.int64))``."""
            return rng.standard_normal(n).astype(np.float64), (np.arange(n) // 1_000_000).astype(np.int64)

        # entropy single-shot at 1M (per-window loop is O(N) python -> too slow to rep at 10M)
        v, g = mk(1_000_000)
        s = time.perf_counter()
        rolling_shannon_entropy_binned(v, g, window_K=K)
        print(f"entropy_binned @1M  {time.perf_counter() - s:.3f}s (single)")

        # cheap vectorized fns @10M best-of-3
        v, g = mk(10_000_000)
        for name, f in [
            ("total_variation", rolling_total_variation),
            ("quantile_spread", rolling_quantile_spread),
            ("zero_crossings", rolling_zero_crossings),
            ("n_peaks", rolling_n_peaks),
        ]:
            f(v, g, window_K=K)
            best = min(((lambda: (time.perf_counter(), f(v, g, window_K=K), time.perf_counter()))() for _ in range(3)), key=lambda x: x[2] - x[0])
            print(f"{name:18s} @10M {best[2] - best[0]:.3f}s")
    finally:
        if previous is sys.modules:
            sys.modules.pop("cupy", None)
        else:
            sys.modules["cupy"] = previous


if __name__ == "__main__":
    main()
