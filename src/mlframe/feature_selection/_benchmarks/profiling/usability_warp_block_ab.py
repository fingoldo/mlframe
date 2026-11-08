"""Same-process A/B of the threads per block of the usability-correlation warp kernel (batch_pair_usability_corr_cuda_warp).

The kernel gives one warp to each (pair, form), so every width is a whole number of warps; the call returns the host copy of the result, so each timing includes the kernel's end.
"""
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
from mlframe.feature_selection.filters import batch_pair_usability_corr_gpu as G

rng = np.random.default_rng(0)
n, m, n_pairs = 100_000, 60, 5000
y = rng.standard_normal(n)
operand = rng.standard_normal((n, m)).astype(np.float32)
pa = rng.integers(0, m, n_pairs).astype(np.int64)
pb = rng.integers(0, m, n_pairs).astype(np.int64)
forms = np.arange(5, dtype=np.int64)

G.batch_pair_usability_corr_cuda_warp(y, operand, pa, pb, forms)
widths = (32, 64, 128, 256, 512)
times = {w: [] for w in widths}
reference = None
for rep in range(5):
    for w in widths:
        G._WARP_BLOCK_THREADS = w
        t = time.perf_counter()
        out = G.batch_pair_usability_corr_cuda_warp(y, operand, pa, pb, forms)
        times[w].append(time.perf_counter() - t)
        if reference is None:
            reference = out
        if not (np.allclose(out, reference, equal_nan=True)):
            raise AssertionError(f"result differs at {w} threads")
for w in widths:
    print(f"{w:4d} threads per block: median {np.median(times[w]):.3f} s  min {min(times[w]):.3f} s")
