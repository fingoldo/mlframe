"""iter111 bench: conditional-residual generate at n=10M.

Replaces the inner-loop ``np.add.at`` unbuffered scatter (2 calls per (x_i, x_j)
pair) with ``np.bincount`` (a single C pass, bit-identical accumulation order)
and hoists the per-x_i finiteness mask + global mean out of the inner loop.

Run (block cupy before import; py3.14 contention segfault otherwise):

    PYTHONPATH=src MLFRAME_SKIP_NUMBA_PREWARM=1 CUDA_VISIBLE_DEVICES="" \
        NUMBA_DISABLE_CUDA=1 python -m mlframe.feature_selection._benchmarks.bench_conditional_residual_generate_iter111

Measured (separate-process, 4 cols incl. one discrete + one NaN-mixed, n=10M):
    OLD min=9.177s  NEW min=6.130s  -> 1.497x, bit-identical (maxabsdiff 0.0).
In-process paired A/B: NEW faster 5/5, median 1.232x.
"""

import sys
import time

import numpy as np
import pandas as pd

if "cupy" not in sys.modules:
    sys.modules["cupy"] = None
import scipy.stats  # noqa: F401  (py3.14 import-order segfault guard)
import numba  # noqa: F401

from mlframe.feature_selection.filters._extra_fe_families import (
    generate_conditional_residual_features,
)


def main(n: int = 10_000_000) -> None:
    rng = np.random.default_rng(1)
    cols = [f"c{i}" for i in range(4)]
    data = {
        cols[0]: rng.standard_normal(n),
        cols[1]: rng.integers(0, 7, n).astype(float),
        cols[2]: rng.standard_normal(n),
        cols[3]: rng.standard_normal(n),
    }
    data[cols[2]][rng.integers(0, n, n // 100)] = np.nan
    X = pd.DataFrame(data)
    generate_conditional_residual_features(X.iloc[:5000], cols)  # warm
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        generate_conditional_residual_features(X, cols)
        ts.append(time.perf_counter() - t)
    print(f"generate min={min(ts):.3f}s median={sorted(ts)[len(ts) // 2]:.3f}s")


if __name__ == "__main__":
    main()
