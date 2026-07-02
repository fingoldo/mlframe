"""cProfile + njit-vs-numpy A/B harness for the engineered-pair separability score kernel.

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_engineered_separability``

The only length-n work is ``separability_score``: two passes over n computing per-class means and the pooled within-
class 2x2 scatter. This bench A/Bs the njit reduction against an equivalent numpy masked-reduction at n in {100k, 1M},
confirms numerical agreement, then cProfiles the panel builder at 1M.

Verdict (best-of-3 walltime, this box):
  n=  100000  njit=   1.45 ms  numpy=   5.56 ms  speedup=3.8x  close=True
  n= 1000000  njit=  14.17 ms  numpy=  55.89 ms  speedup=3.9x  close=True
The njit two-pass reduction fuses the per-class accumulation and avoids the numpy path's boolean-mask gathers +
intermediate ``column_stack`` allocations, so it is ~3.8-3.9x faster and agrees with numpy to < 1e-9. No further
actionable speedup -- two passes over n is the floor for mean + pooled-covariance. The panel additionally subsamples to
5000 rows before plotting, so end-to-end the kernel cost is negligible.
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.reporting.charts.engineered_separability import separability_panel, separability_score


def _numpy_score(z0, z1, y):
    mask1 = y > 0.5
    n1 = int(mask1.sum())
    n0 = int(y.shape[0] - n1)
    if n0 == 0 or n1 == 0:
        return 0.0
    m1 = np.array([z0[mask1].mean(), z1[mask1].mean()])
    mask0 = ~mask1
    m0 = np.array([z0[mask0].mean(), z1[mask0].mean()])
    d0 = np.column_stack([z0[mask0] - m0[0], z1[mask0] - m0[1]])
    d1 = np.column_stack([z0[mask1] - m1[0], z1[mask1] - m1[1]])
    denom = max(n0 + n1 - 2, 1)
    cov = (d0.T @ d0 + d1.T @ d1) / denom
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    if det < 1e-12:
        det = 1e-12
    inv = np.array([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]]) / det
    dd = m1 - m0
    return float(dd @ inv @ dd)


def _walltime(fn, repeats: int = 3) -> float:
    fn()  # warm (njit compile)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    for n in (100_000, 1_000_000):
        rng = np.random.default_rng(0)
        half = n // 2
        z2 = np.vstack([rng.standard_normal((half, 2)), rng.standard_normal((n - half, 2)) + 3.0])
        y = np.concatenate([np.zeros(half), np.ones(n - half)])
        z0 = np.ascontiguousarray(z2[:, 0])
        z1 = np.ascontiguousarray(z2[:, 1])
        njit_ms = _walltime(lambda: separability_score(z2, y)) * 1e3
        numpy_ms = _walltime(lambda: _numpy_score(z0, z1, y)) * 1e3
        s_njit = separability_score(z2, y)
        s_np = _numpy_score(z0, z1, y)
        close = abs(s_njit - s_np) < 1e-9 * max(1.0, abs(s_np))
        speedup = numpy_ms / njit_ms if njit_ms > 0 else float("inf")
        print(f"n={n:>9}  njit={njit_ms:8.2f} ms  numpy={numpy_ms:8.2f} ms  speedup={speedup:.1f}x  close={close}")

    n = 1_000_000
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = rng.integers(0, 2, size=n).astype(np.float64)
    separability_panel(X, y, ["a", "b"])  # warm
    pr = cProfile.Profile()
    pr.enable()
    separability_panel(X, y, ["a", "b"])
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(12)
    print(s.getvalue())


if __name__ == "__main__":
    main()
