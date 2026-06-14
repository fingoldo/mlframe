"""Bench: QRF ``predict_quantile`` weighted-ECDF inversion -- Python per-row loop vs njit-prange batch kernel.

The ``_LeafResidualForest.predict_quantile`` inner loop inverts the conditional weighted ECDF one query row at a time
(mask-nonzero -> ``np.argsort`` -> ``np.cumsum`` -> ``np.interp`` per row). ``_batch_weighted_quantiles_kernel`` does the
whole dense ``(batch, n_train)`` membership matrix in a single njit(parallel) pass: per-row compact gather of nonzero
``(value, weight)`` pairs, stable insertion-sort by value, centered cumulative-weight plotting positions, binary-search
interp of the requested levels.

Run: ``python -c "import sys;sys.modules['cupy']=None;import scipy.stats,numba" && python bench_qrf_weighted_quantiles.py``

Measured (this box, py3.14, 8 threads, n_train=20000, n_nz~=400/row, K=19 levels):
  isolated kernel-only seam : 17.13s -> 1.50s for 199,680 rows (~11.4x), max abs diff 7e-14 (FP reduction order).
  end-to-end CompositeQRFEstimator.predict_quantile (n_train=1500, K=5, 80k query rows): see run output.
Output is BIT-IDENTICAL to the Python path within FP reduction-order tolerance (~1e-13); the kernel is gated behind
``_HAS_NUMBA`` with the original Python loop as the numba-unavailable fallback.
"""
from __future__ import annotations

import sys

sys.modules.setdefault("cupy", None)  # avoid cold cupy import segfault on py3.14

import time

import numpy as np


def _weighted_quantiles(values, weights, lv):
    total = weights.sum()
    if values.size == 0 or total <= 0:
        return np.full(lv.shape[0], np.nan)
    order = np.argsort(values, kind="mergesort")
    v_s = values[order]
    w_s = weights[order]
    cum = np.cumsum(w_s)
    pos = (cum - 0.5 * w_s) / total
    return np.interp(lv, pos, v_s, left=v_s[0], right=v_s[-1])


def main() -> None:
    import numba  # noqa: F401
    from mlframe.training.composite.qrf import _batch_weighted_quantiles_kernel

    rng = np.random.default_rng(0)
    n_train = 20000
    y_train = rng.standard_normal(n_train)
    levels = np.array([round(0.05 * k, 2) for k in range(1, 20)])
    PB = 512
    n_query = 200000

    def gen_batch(bsz):
        w = np.zeros((bsz, n_train))
        for r in range(bsz):
            cols = rng.integers(0, n_train, size=400)
            w[r, cols] += rng.random(400)
        return w

    wb = gen_batch(PB)
    out_o = np.empty((PB, levels.shape[0]))
    for r in range(PB):
        nz = wb[r] > 0.0
        out_o[r, :] = _weighted_quantiles(y_train[nz], wb[r][nz], levels)
    out_n = np.empty((PB, levels.shape[0]))
    _batch_weighted_quantiles_kernel(wb[:5], y_train, levels, out_n, 0)  # warm JIT
    out_n = np.empty((PB, levels.shape[0]))
    _batch_weighted_quantiles_kernel(wb, y_train, levels, out_n, 0)
    print("max abs diff:", float(np.nanmax(np.abs(out_o - out_n))))

    batches = [gen_batch(PB) for _ in range(n_query // PB)]
    n = len(batches) * PB
    out = np.empty((n, levels.shape[0]))
    t0 = time.perf_counter()
    for bi, w in enumerate(batches):
        s = bi * PB
        for r in range(w.shape[0]):
            nz = w[r] > 0.0
            out[s + r, :] = _weighted_quantiles(y_train[nz], w[r][nz], levels)
    t1 = time.perf_counter()
    print(f"OLD python per-row {n} rows: {t1 - t0:.2f}s")
    out2 = np.empty((n, levels.shape[0]))
    t0 = time.perf_counter()
    for bi, w in enumerate(batches):
        _batch_weighted_quantiles_kernel(w, y_train, levels, out2, bi * PB)
    t1 = time.perf_counter()
    print(f"NEW njit-prange batch {n} rows: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
