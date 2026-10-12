"""Median wall time of a warm whole fit of the F2 example at several row counts (the fixed-cost benchmark).

Usage: ``python -m mlframe.feature_selection._benchmarks.profiling.fit_wall [rounds] [rows comma separated] [extra MRMR kwargs as key=value, ...]``. Every fit gets novel data (MRMR memoises by content hash); one warm fit
per size comes first. GPU strict-resident mode, ``fe_max_steps=2`` as in ``offset_product/ab_fit.py``. Run it on a quiet box.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

for _k, _v in {"MLFRAME_FE_GPU_STRICT": "1", "MLFRAME_CMI_GPU": "1", "MLFRAME_FE_VRAM_F32": "1", "MLFRAME_FE_GPU_DISCRETIZE": "1", "MLFRAME_FE_GPU_BINNING": "1"}.items():
    os.environ.setdefault(_k, _v)
warnings.simplefilter("ignore")

import numpy as np

from mlframe.feature_selection._benchmarks.offset_product.ab_fit import _make
from mlframe.feature_selection.filters.mrmr import MRMR


def main() -> None:
    """Warm up, time the rounds, print the medians."""
    import ast

    import cupy as cp

    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sizes = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [10_000, 100_000]
    extra = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[3:])}
    kw = dict(full_npermutations=10, baseline_npermutations=20, fe_max_steps=2, fe_min_pair_mi_prevalence=1.05, verbose=0, n_jobs=1) | extra
    for n in sizes:
        df, y = _make(n, 1)
        MRMR(random_seed=1, **kw).fit(df, y)
        walls = []
        for r in range(rounds):
            df, y = _make(n, 10 + r)
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            MRMR(random_seed=10 + r, **kw).fit(df, y)
            cp.cuda.Device().synchronize()
            walls.append(time.perf_counter() - t0)
        print(f"WALL n={n} median {np.median(walls):.2f}s  (rounds {', '.join(f'{w:.2f}' for w in walls)})", flush=True)


if __name__ == "__main__":
    main()
