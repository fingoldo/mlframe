"""CPX33 bench: Kendall-tau prescreen, old O(n^2) hand-rolled loop vs scipy.stats.kendalltau (Knight's O(n log n)).

The old O(n^2) reference is preserved in-tree as ``_kendall_tau_z`` (now only the scipy-ImportError fallback). We bench
it directly against scipy across n in {500, 2000, 10000, 50000}, warm + best-of-N. The old path silently subsampled to
2000 rows at n>2000 (changing the statistic); scipy uses the FULL n -- so this is a speed AND correctness win.

Run:  CUDA_VISIBLE_DEVICES="" python src/mlframe/feature_selection/wrappers/_benchmarks/bench_cpx33_kendall.py
"""

import time

import numpy as np
from scipy.stats import kendalltau

from mlframe.feature_selection.wrappers._univariate_ht import _kendall_tau_z

SIZES = (500, 2000, 10000, 50000)
REPEAT = 5


def _best_of(fn, repeat=REPEAT):
    fn()  # warm (numba JIT for the njit O(n^2) kernel)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    print(f"{'n':>8} {'old O(n^2) ms':>16} {'scipy ms':>12} {'speedup':>9}  old-subsampled?")
    for n in SIZES:
        x = rng.normal(size=n)
        y = 0.5 * x + rng.normal(size=n)

        # The old path ACTUALLY ran the O(n^2) loop on a <=2000-row subsample; bench it on that reduced shape to reflect
        # the real prior cost, and separately note that scipy runs full-n.
        n_old = min(n, 2000)
        xo, yo = x[:n_old], y[:n_old]
        t_old = _best_of(lambda: _kendall_tau_z(xo, yo))
        t_scipy = _best_of(lambda: kendalltau(x, y, variant="b"))
        sub = "yes (->2000)" if n > 2000 else "no"
        print(f"{n:>8} {t_old * 1e3:>16.3f} {t_scipy * 1e3:>12.3f} {t_old / t_scipy:>8.2f}x  {sub}")


if __name__ == "__main__":
    main()
