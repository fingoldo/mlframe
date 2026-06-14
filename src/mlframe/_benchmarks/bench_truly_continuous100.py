"""iter100 bench: is_variable_truly_continuous fract-digits probe at 10M rows.

The continuity probe loops up to ``max_fract_digits-1`` precisions, each time computing the distinct-count of ``np.round(fract_part, d)`` -- a fresh O(n log n)
sort plus a rounded-copy allocation per precision. Rounding is monotone, so ``sort(round(x, d)) == round(sort(x), d)`` elementwise: sorting the fractional part
ONCE and counting distinct of the inline-rounded values over that single sorted array is bit-identical and removes every re-sort.

Run:  python -m mlframe._benchmarks.bench_truly_continuous100

Measured (10M float64, randn*100 rounded to 6 digits; probe breaks at d=7, py3.14 store, CUDA off):
  isolated loop section (7 precisions): OLD 1.796s -> NEW 0.565s (3.2x)
  separate-process end-to-end:          OLD min 3.629s / median 5.759s -> NEW min 2.362s / median 2.894s (1.54x min, 1.99x median)
  verdict bit-identical across cont6 / cont2 / intish / withnan / lowcard / single-digit fixtures.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def main() -> None:
    sys.modules.setdefault("cupy", None)
    import scipy.stats  # noqa: F401
    import numba  # noqa: F401

    from mlframe.preprocessing.cleaning import is_variable_truly_continuous

    rng = np.random.default_rng(0)
    n = 10_000_000
    vals = np.round(rng.normal(size=n) * 100, 6)
    is_variable_truly_continuous(values=vals[:2000].copy())

    ts = []
    for _ in range(9):
        t = time.perf_counter()
        is_variable_truly_continuous(values=vals)
        ts.append(time.perf_counter() - t)
    ts.sort()
    print(f"n={n} min={ts[0]:.3f}s median={ts[len(ts) // 2]:.3f}s")


if __name__ == "__main__":
    main()
