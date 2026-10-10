"""Time of the offset-product scan (all column pairs x unary pairs of 6 columns) on the CPU kernel and on the fused CUDA kernel, at several scan sizes.

Run: ``python -m mlframe.feature_selection._benchmarks.offset_product.scan_bench``. The device time includes the host-to-device upload of the unary outputs and the copy-back of the score tables.
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_selection.filters import _offset_product_fe as fe
from mlframe.feature_selection.filters._offset_product_gpu import scan_offset_products_gpu
from mlframe.feature_selection.filters._offset_product_kernels import N_BASELINES, scan_offset_products
from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

N_COLS = 6
N_BINS = 10
SIZES = (20000, 100000, 300000)
REPEATS = 3


def _inputs(n: int):
    """Unary outputs, tasks, rank target, class codes and clips of the sign-crossing case."""
    r = np.random.default_rng(0)
    cols = [r.random(n) for _ in range(N_COLS)]
    y = 0.2 * cols[0] ** 2 / cols[1] + np.log(cols[2] * 2) * np.sin(cols[3] / 3)
    unaries = fe.OFFSET_UNARIES
    U, clips = fe._scan_inputs(cols, fe._unary_funcs("minimal"), unaries, np.arange(n))
    nu = len(unaries)
    tasks = np.array([(i, j, p, q) for i in range(N_COLS) for j in range(i + 1, N_COLS) for p in range(nu) for q in range(nu)], dtype=np.int64)
    codes = np.asarray(encode_y_for_classif_mi(y), dtype=np.int64)
    return U, tasks, fe._rank_scaled(y), codes, int(codes.max()) + 1, clips


def _best(fn) -> float:
    """Best wall time of ``REPEATS`` calls (the first call, which compiles, is made before timing)."""
    fn()
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    """Print CPU and GPU scan seconds per size."""
    for n in SIZES:
        U, tasks, yr, codes, ky, clips = _inputs(n)
        out_shift = np.empty((len(tasks), 2))
        out_mi = np.empty((len(tasks), 2, 1 + N_BASELINES))
        cpu = _best(lambda: scan_offset_products(U, tasks, yr, codes, ky, N_BINS, clips, out_shift, out_mi))
        gpu = _best(lambda: scan_offset_products_gpu(U, tasks, yr, codes, ky, N_BINS, clips))
        print(f"n={n:7d} tasks={len(tasks)}  cpu {cpu:.3f}s  gpu {gpu:.3f}s  ({cpu / gpu:.1f}x)", flush=True)


if __name__ == "__main__":
    main()
