"""A/B of the pair_mi_shared_fused read layout: row-major factors (n, F) against column-major (F, n).

Row-major is the production kernel (through ``batch_pair_mi_cuda_shared_fused``). The column-major variant only swaps the two
factor reads, and its ``(F, n)`` copy is built once and reused, so the timing is reported with and without that one-off transpose.
"""

import time

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _batch_pair_mi_cuda_shared_fused as shared

from mlframe.feature_selection.filters._benchmarks._pair_mi_row_major_src import ROW_MAJOR_SRC

ROW_SRC = ROW_MAJOR_SRC
COL_SRC = shared._PAIR_MI_SHARED_FUSED_SRC
k_row = cp.RawKernel(ROW_SRC, "pair_mi_shared_fused")
k_col = cp.RawKernel(COL_SRC, "pair_mi_shared_fused")


def timed(fn, reps=5):
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
    pb = (pa + 1 + rng.integers(0, F - 1, size=P)) % F
    pb = pb.astype(np.int64)
    freqs_y = np.bincount(classes_y, minlength=NCY).astype(np.float64) / n
    max_joint = NB * NB
    shared_bytes = (max_joint * NCY + max_joint) * 4

    ref = shared.batch_pair_mi_cuda_shared_fused(factors, pa, pb, nbins, classes_y, freqs_y)
    d_row = cp.asarray(factors)
    d_col = cp.ascontiguousarray(d_row.T)
    d_pa, d_pb = cp.asarray(pa), cp.asarray(pb)
    d_nb, d_cy, d_fy = cp.asarray(nbins), cp.asarray(classes_y), cp.asarray(freqs_y)
    d_out = cp.zeros(P, dtype=cp.float64)
    args_common = (np.int64(n), np.int32(F), np.int32(P), np.int32(max_joint), np.int32(NCY), np.float64(1.0 / n), d_out)

    def run_row():
        k_row((P,), (128,), (d_row, d_pa, d_pb, d_nb, d_cy, d_fy, *args_common), shared_mem=shared_bytes)  # launch-ok: layout benchmark, the block size is part of the measured variant

    def run_col():
        k_col.max_dynamic_shared_size_bytes = max(shared_bytes, 0)
        k_col((P,), (128,), (d_col, d_pa, d_pb, d_nb, d_cy, d_fy, *args_common), shared_mem=shared_bytes)  # launch-ok: layout benchmark, the block size is part of the measured variant

    run_col()
    got = cp.asnumpy(d_out)
    identical = np.array_equal(got, ref)
    t_row = timed(run_row)
    t_col = timed(run_col)
    t_tr = timed(lambda: cp.ascontiguousarray(d_row.T))
    print(f"n={n:>9} P={P:>5}  row={t_row*1e3:9.2f}ms  col(kernel)={t_col*1e3:9.2f}ms  speedup={t_row/t_col:5.2f}x  transpose={t_tr*1e3:8.2f}ms  identical={identical}")
