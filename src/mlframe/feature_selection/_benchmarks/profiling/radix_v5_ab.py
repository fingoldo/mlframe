"""Same-process A/B of the warp-aggregated first pass of the f32 radix select (v5) against the compaction kernel (v4).

MLFRAME_RADIX_F32_V5 is read per launch; the order statistics must be equal bit for bit. 100000 rows x K columns of float32 values of three distributions.
"""
import os
import time
import warnings

import cupy as cp
import numpy as np

warnings.simplefilter("ignore")
from mlframe.feature_selection.filters import _gpu_resident_select as S

rng = np.random.default_rng(0)
n = 100_000
for K, kind in ((100, "normal"), (100, "uniform"), (30, "lognormal")):
    raw = {"normal": rng.standard_normal((n, K)), "uniform": rng.random((n, K)), "lognormal": np.exp(rng.standard_normal((n, K)))}[kind]
    X = cp.asarray(raw.astype(np.float32))

    def run(v5: bool):
        """One timed edge computation with the v5 switch set; the matrix is copied because the NaN canonicalisation writes in place."""
        os.environ["MLFRAME_RADIX_F32_V5"] = "1" if v5 else "0"
        Xc = X.copy()
        cp.cuda.Stream.null.synchronize()
        t = time.perf_counter()
        out = S._radix_select_interior_edges(Xc, 10)
        cp.cuda.Stream.null.synchronize()
        return time.perf_counter() - t, cp.asnumpy(out)

    run(True)
    run(False)
    t4, t5 = [], []
    for _ in range(7):
        a, ref = run(False)
        b, new = run(True)
        t4.append(a)
        t5.append(b)
        if not (np.array_equal(ref, new)):
            raise AssertionError("v5 changed an order statistic")
    print(f"{kind:9s} K={K:3d}: v4 median {np.median(t4) * 1e3:7.2f} ms   v5 median {np.median(t5) * 1e3:7.2f} ms   speed-up {np.median(t4) / np.median(t5):.2f}x   bit-identical")
