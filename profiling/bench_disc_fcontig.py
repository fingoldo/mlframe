"""Q4 PROTOTYPE: F-contiguous discretize buffers (unit-stride inner loop).

Both _quantile_edges_2d_njit and _searchsorted_2d_right_njit iterate
  for j (cols): for r (rows): arr2d[r, j]
For a C-contiguous array, the inner r-loop strides by n_cols (cache-hostile).
For an F-contiguous (column-major) array, arr2d[r, j] over inner r is UNIT stride.

This bench measures the searchsorted + edges kernels on C-contig vs F-contig input,
INCLUDING the asfortranarray copy cost (which the production caller would pay), to
decide if the layout win beats the copy.
"""
import math, time
import numpy as np
from numba import njit, prange


@njit(parallel=True, nogil=True, cache=False)
def _edges(arr2d, quantiles, edges_out):
    n_rows = arr2d.shape[0]; n_cols = arr2d.shape[1]; n_q = quantiles.shape[0]
    for j in prange(n_cols):
        col = np.empty(n_rows, dtype=arr2d.dtype)
        for r in range(n_rows):
            col[r] = arr2d[r, j]
        col.sort()
        for qi in range(n_q):
            v = (quantiles[qi] / 100.0) * (n_rows - 1)
            lo = int(math.floor(v))
            if lo >= n_rows - 1:
                edges_out[qi, j] = col[n_rows - 1]
            else:
                a = float(col[lo]); b = float(col[lo + 1]); t = v - lo; d = b - a
                edges_out[qi, j] = (b - d * (1.0 - t)) if t >= 0.5 else (a + d * t)


@njit(parallel=True, nogil=True, cache=False)
def _ss(edges_inner, arr2d, out):
    n_rows = arr2d.shape[0]; n_cols = arr2d.shape[1]; n_edges = edges_inner.shape[0]
    for j in prange(n_cols):
        for r in range(n_rows):
            v = arr2d[r, j]; lo = 0; hi = n_edges
            while lo < hi:
                mid = (lo + hi) >> 1
                if v < edges_inner[mid, j]:
                    hi = mid
                else:
                    lo = mid + 1
            out[r, j] = lo


def _time(fn, *a, repeats=6):
    fn(*a)
    best = 1e30
    for _ in range(repeats):
        t = time.perf_counter(); fn(*a); best = min(best, time.perf_counter() - t)
    return best


def run(n_rows, n_cols, n_bins=10, dtype=np.float32):
    rng = np.random.default_rng(0)
    arr_c = np.ascontiguousarray(rng.standard_normal((n_rows, n_cols)).astype(dtype))
    quantiles = np.linspace(0, 100, n_bins + 1)

    # C-contig path (current production)
    edges_c = np.empty((n_bins + 1, n_cols), dtype=np.float64)
    out_c = np.empty((n_rows, n_cols), dtype=np.int16)
    def full_c():
        _edges(arr_c, quantiles, edges_c)
        ei = np.ascontiguousarray(edges_c[1:-1], dtype=np.float64)
        _ss(ei, arr_c, out_c)

    # F-contig path: pay asfortranarray copy ONCE, both kernels read unit-stride inner.
    edges_f = np.empty((n_bins + 1, n_cols), dtype=np.float64)
    out_f = np.empty((n_rows, n_cols), dtype=np.int16, order="C")
    def full_f():
        arr_f = np.asfortranarray(arr_c)              # copy cost included
        _edges(arr_f, quantiles, edges_f)
        ei = np.asfortranarray(edges_f[1:-1].astype(np.float64))  # edges col-strided too
        _ss(ei, arr_f, out_f)

    # F-contig SEARCHSORTED ONLY (edges still C, since edges kernel copies col anyway)
    edges_c2 = np.empty((n_bins + 1, n_cols), dtype=np.float64)
    out_f2 = np.empty((n_rows, n_cols), dtype=np.int16)
    def full_ss_only_f():
        _edges(arr_c, quantiles, edges_c2)            # edges on C (copies col internally)
        arr_f = np.asfortranarray(arr_c)              # copy for ss
        ei = np.asfortranarray(edges_c2[1:-1].astype(np.float64))
        _ss(ei, arr_f, out_f2)

    t_c = _time(full_c)
    t_f = _time(full_f)
    t_ssf = _time(full_ss_only_f)

    # correctness
    full_c(); full_f(); full_ss_only_f()
    ok_f = np.array_equal(out_c, out_f) and np.array_equal(edges_c, edges_f)
    ok_ssf = np.array_equal(out_c, out_f2)
    print(f"shape {n_rows}x{n_cols} dt={np.dtype(dtype).name}")
    print(f"  C-contig(prod) {t_c*1e3:8.2f}ms | F-contig(both+copy) {t_f*1e3:8.2f}ms ({t_c/t_f:.2f}x) ident={ok_f}"
          f" | F-ss-only(+copy) {t_ssf*1e3:8.2f}ms ({t_c/t_ssf:.2f}x) ident={ok_ssf}")


if __name__ == "__main__":
    for nc in (300, 1000, 4000):
        run(2407, nc, dtype=np.float32)
        run(2407, nc, dtype=np.float64)
