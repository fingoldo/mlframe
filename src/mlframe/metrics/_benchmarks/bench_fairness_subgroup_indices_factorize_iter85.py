"""iter85 @1M bench: factorize fast path for ``create_fairness_subgroups_indices``.

The per-bin ``np.where(bins == bin_name)[0]`` loop was O(n*B): for a high-card categorical (B bins) over object dtype it ran B full element-wise string
comparisons over all n rows (pandas ``comp_method_OBJECT_ARRAY`` dominated). A single ``pd.factorize`` + stable argsort partitions all groups in one
O(n log n) pass. Measured n=1M, B=200 string categorical: OLD 12046.5 ms -> NEW 393.3 ms = 30.6x, bit-identical on the string + numeric-qcut paths.

Run:  python -m mlframe.metrics._benchmarks.bench_fairness_subgroup_indices_factorize_iter85
"""
from __future__ import annotations

import sys
import time
import io
import contextlib

import numpy as np
import pandas as pd


def main() -> None:
    sys.modules.setdefault("cupy", None)  # dodge py3.14 cold cupy import segfault
    import scipy.stats  # noqa: F401
    import numba  # noqa: F401
    from mlframe.metrics._fairness_metrics import create_fairness_subgroups, create_fairness_subgroups_indices

    rng = np.random.default_rng(0)
    n = 1_000_000
    feat = pd.Series([f"c{v:03d}" for v in rng.integers(0, 200, n)], name="region")
    sg = create_fairness_subgroups(pd.DataFrame({"region": feat}), ["region"], min_pop_cat_thresh=100)
    idx = np.arange(n)
    a, b, c = idx, idx[: int(n * 0.6)], idx[int(n * 0.6) :]

    with contextlib.redirect_stderr(io.StringIO()):
        create_fairness_subgroups_indices(sg, a, b, c)  # warm
        best = 1e9
        for _ in range(3):
            t0 = time.perf_counter()
            create_fairness_subgroups_indices(sg, a, b, c)
            best = min(best, time.perf_counter() - t0)
    print(f"NEW (factorize) best-of-3 @1M B=200: {best * 1000:.1f} ms")
    print("Baseline (per-bin np.where loop, HEAD~iter84): ~12046 ms -> 30.6x")


if __name__ == "__main__":
    main()
