"""Row-major versus column-major X read for the production f32 mi_mm_from_values_nek kernel, kernel alone and with the device transpose included."""

import time

import cupy as cp
import numpy as np

from mlframe.feature_selection.filters import _fe_batched_mi as m
from mlframe.feature_selection.filters._gpu_resident_select_kernels import _transpose_to_cm

SRC_ROW = m._MI_MM_FROM_VALUES_NEK_F32_SRC
OLD = "X[i * (long long)K + c]"
assert SRC_ROW.count(OLD) == 1, SRC_ROW.count(OLD)
KERNEL_ROW = cp.RawKernel(SRC_ROW, "mi_mm_from_values_nek_f32")
KERNEL_COL = cp.RawKernel(SRC_ROW.replace(OLD, "X[(long long)c * n + i]"), "mi_mm_from_values_nek_f32")
NBINS, KY = 10, 2


def timed(fn, reps=7):
    """Best of ``reps`` synchronized wall times of ``fn`` in ms."""
    fn()
    cp.cuda.Stream.null.synchronize()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        best = min(best, time.perf_counter() - t)
    return best * 1000


def main():
    """Print the row-major, column-major and column-major-plus-transpose times for several widths."""
    rng = np.random.default_rng(0)
    for n in (30000, 100000):
        for k in (1, 8, 64, 256):
            x = cp.asarray(rng.standard_normal((n, k)).astype(np.float32))
            edges = cp.asarray(np.sort(rng.standard_normal((NBINS, k)), axis=0))
            ne_k = cp.full(k, NBINS - 1, dtype=cp.int32)
            y = cp.asarray(rng.integers(0, KY, n).astype(np.int64))
            out_r, out_c = cp.empty(k), cp.empty(k)
            shared = NBINS * KY * 4

            def launch(kernel, xa, out):
                kernel((k,), (256,), (xa, edges, y, np.int64(n), np.int32(k), np.int32(NBINS), np.int32(KY), np.float64(1.0 / n), np.float64(0.69), np.int32(2), ne_k, out), shared_mem=shared)  # launch-ok: layout benchmark

            xt = _transpose_to_cm(x)
            t_row = timed(lambda: launch(KERNEL_ROW, x, out_r))
            t_col = timed(lambda: launch(KERNEL_COL, xt, out_c))
            t_tr = timed(lambda: launch(KERNEL_COL, _transpose_to_cm(x), out_c))
            print(f"n={n:>6} K={k:>3} row={t_row:7.3f} ms  col(kernel)={t_col:7.3f}  col+transpose={t_tr:7.3f}  equal={bool(cp.array_equal(out_r, out_c))}")


if __name__ == "__main__":
    main()
