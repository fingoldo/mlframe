"""End-to-end host-to-host timing of pair_mi_shared_fused: the production wrapper against the legacy row-major kernel.

Production gathers the referenced columns column-major on the device; this times it against the legacy row-major kernel. Kernel-only timings live in bench_pair_mi_shared_fused_layout.py.
"""

import time

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _batch_pair_mi_cuda_shared_fused as shared
from mlframe.feature_selection.filters._benchmarks._pair_mi_row_major_src import ROW_MAJOR_SRC

k_row = cp.RawKernel(ROW_MAJOR_SRC, "pair_mi_shared_fused")


def e2e_row(factors, pa, pb, nbins, classes_y, freqs_y):
    """The legacy path: the raw row-major (n, F) matrix is uploaded and read with a stride of F rows by the kernel."""
    n, F = factors.shape
    P = len(pa)
    d_rows = cp.asarray(np.ascontiguousarray(factors, dtype=np.int32))
    d_pa, d_pb = cp.asarray(pa.astype(np.int64)), cp.asarray(pb.astype(np.int64))
    d_nb, d_cy = cp.asarray(nbins.astype(np.int32)), cp.asarray(classes_y.astype(np.int32))
    d_fy = cp.asarray(freqs_y.astype(np.float64))
    d_out = cp.zeros(P, dtype=cp.float64)
    max_joint = int(nbins.max()) ** 2
    ncy = len(freqs_y)
    shared_bytes = (max_joint * ncy + max_joint) * 4
    k_row.max_dynamic_shared_size_bytes = shared_bytes
    k_row((P,), (128,), (d_rows, d_pa, d_pb, d_nb, d_cy, d_fy, np.int64(n), np.int32(F), np.int32(P), np.int32(max_joint), np.int32(ncy), np.float64(1.0 / n), d_out), shared_mem=shared_bytes)  # launch-ok: legacy row-major baseline
    return cp.asnumpy(d_out)


def timed(fn, reps=3):
    fn()
    cp.cuda.Device().synchronize()
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        best = min(best, time.perf_counter() - s)
    return best


rng = np.random.default_rng(0)
F, NCY, NB = 40, 2, 8
for n, P in ((100_000, 2_000), (1_000_000, 500)):
    factors = rng.integers(0, NB, size=(n, F)).astype(np.int32)
    nbins = np.full(F, NB, dtype=np.int32)
    classes_y = rng.integers(0, NCY, size=n).astype(np.int32)
    pa = rng.integers(0, F, size=P).astype(np.int64)
    pb = ((pa + 1 + rng.integers(0, F - 1, size=P)) % F).astype(np.int64)
    freqs_y = np.bincount(classes_y, minlength=NCY).astype(np.float64) / n
    ref = shared.batch_pair_mi_cuda_shared_fused(factors, pa, pb, nbins, classes_y, freqs_y)
    got = e2e_row(factors, pa, pb, nbins, classes_y, freqs_y)
    t_new = timed(lambda: shared.batch_pair_mi_cuda_shared_fused(factors, pa, pb, nbins, classes_y, freqs_y))
    t_cur = timed(lambda: e2e_row(factors, pa, pb, nbins, classes_y, freqs_y))
    print(f"n={n:>9} P={P:>5}  row-major e2e={t_cur*1e3:9.2f}ms  production (column) e2e={t_new*1e3:9.2f}ms  speedup={t_cur/t_new:5.2f}x  identical={np.array_equal(got, ref)}")
