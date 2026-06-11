"""cProfile harness for the multi-dimensional weak-slice finder (charts/slice_finder.py).

Run: ``python -m mlframe.reporting.charts._benchmarks.profile_slice_finder``

The slice finder bins every feature once into an int16 code matrix, then evaluates
each candidate slice with a single ``np.bincount`` (sum_error) + ``np.bincount``
(count) over the combined linear bin code -- one O(n) pass per combination, no
per-row python. At n=1e6 / p=8 features the cost is dominated by the per-feature
quantile binning (``np.quantile`` sort per column) and the bincount passes over
the (n,) flat code; both are O(n). The python decode loop runs only over the
SURVIVING cells (>= support floor, positive degradation), which is bounded by the
small per-combo cell count, not n. This harness records that wall stays bounded
and that the bincount aggregation -- not any python row loop -- is the hot path.

Optimization trail (n=1e6, p=8, 2-way, 36 combos): 1240 ms -> 827 ms (-33%).
Two changes, both removing per-combo length-n work: (1) the bin-code matrix is
stored int64 so the mixed-radix flatten reuses it without re-casting int16->int64
per slice (was ~16% of wall in 420 .astype calls); (2) the flatten accumulates in
place and skips the stride-1 multiply on the last (LSB) feature; and _bin_matrix
drops np.nan_to_num for a cheap masked low-edge fill. The residual is the two
np.bincount passes per combo + the one-time per-column np.quantile sort -- both
O(n), the genuine aggregate-first floor; no further actionable speedup without
fusing the per-combo bincounts into a single numba kernel (deferred: 36 combos
already amortise the dispatch, and bincount is C-level).
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.reporting.charts.slice_finder import find_weak_slices


def _make_data(n: int, p: int = 8):
    rng = np.random.default_rng(0)
    cols = {f"f{j}": rng.random(n) for j in range(p)}
    X = pd.DataFrame(cols)
    bad = (X["f0"] > 0.7) & (X["f1"] > 0.7)
    err = np.where(bad, 6.0, 1.0 + 0.2 * rng.random(n))
    return X, np.zeros(n), err


def main(n: int = 1_000_000):
    X, y_true, y_pred = _make_data(n)
    find_weak_slices(X, y_true, y_pred, max_arity=2)  # warmup

    t0 = time.perf_counter()
    for _ in range(3):
        find_weak_slices(X, y_true, y_pred, max_arity=2)
    wall = (time.perf_counter() - t0) / 3.0
    print(f"slice_finder (p=8, 2-way) @ n={n}: {wall*1000:.1f} ms/call (mean of 3)")

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        find_weak_slices(X, y_true, y_pred, max_arity=2)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    main()
