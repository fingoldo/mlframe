"""Bench S45 / W4b: alternative implementations for the per-bin reduction
loop in ``_median_residual_fit`` and ``_quantile_residual_fit``.

Three candidate variants for the median-residual binned reduction:

* ``v1_pyloop``         -- the production baseline; for each bin ``i`` build a boolean mask ``bin_idx == i``, then ``np.median(y[mask])``.
* ``v2_sort_split``     -- single ``np.argsort(bin_idx)``, ``np.searchsorted`` to find bin boundaries, then iterate over slices computing a contiguous-slice median (no Python boolean mask materialisation).
* ``v3_pandas_groupby`` -- ``pd.Series(y).groupby(bin_idx).median()`` -- pandas vectorised hash-groupby + sort-based median; one pass over the data, no per-bin Python loop.

Same three for the quantile-residual case (median + 25 / 75 IQR percentiles).

Output: prints a table of (variant, n, n_bins, ms) plus a JSON dump under
``_results/bench_median_quantile_residual_<timestamp>.json``.

Run:
    D:/ProgramData/anaconda3/python.exe -m mlframe.training._benchmarks.bench_median_quantile_residual
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import numba as _nb
    _HAS_NB = True
except Exception:
    _nb = None
    _HAS_NB = False

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "_results"
_RESULTS.mkdir(exist_ok=True)


def _median_v1_pyloop(y: np.ndarray, bin_idx: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.full(n_bins, np.median(y), dtype=np.float64)
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.any():
            out[i] = float(np.median(y[mask]))
    return out


def _median_v2_sort_split(y: np.ndarray, bin_idx: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.full(n_bins, np.median(y), dtype=np.float64)
    order = np.argsort(bin_idx, kind="stable")
    sorted_bins = bin_idx[order]
    sorted_y = y[order]
    starts = np.searchsorted(sorted_bins, np.arange(n_bins), side="left")
    ends = np.searchsorted(sorted_bins, np.arange(n_bins), side="right")
    for i in range(n_bins):
        if ends[i] > starts[i]:
            out[i] = float(np.median(sorted_y[starts[i] : ends[i]]))
    return out


if _HAS_NB:
    @_nb.njit(cache=True)
    def _median_v4_njit_kernel(sorted_y, starts, ends, n_bins, global_med):
        out = np.full(n_bins, global_med, dtype=np.float64)
        for i in range(n_bins):
            lo = starts[i]
            hi = ends[i]
            sz = hi - lo
            if sz > 0:
                # sort the slice in-place via copy + np.partition for median.
                slc = sorted_y[lo:hi].copy()
                slc.sort()
                if sz % 2 == 1:
                    out[i] = slc[sz // 2]
                else:
                    out[i] = 0.5 * (slc[sz // 2 - 1] + slc[sz // 2])
        return out

    @_nb.njit(cache=True)
    def _quantile_v4_njit_kernel(sorted_y, starts, ends, n_bins, global_med, global_iqr):
        meds = np.full(n_bins, global_med, dtype=np.float64)
        iqrs = np.full(n_bins, global_iqr, dtype=np.float64)
        sizes = np.zeros(n_bins, dtype=np.int64)
        for i in range(n_bins):
            lo = starts[i]
            hi = ends[i]
            sz = hi - lo
            sizes[i] = sz
            if sz >= 4:
                slc = sorted_y[lo:hi].copy()
                slc.sort()
                # linear-interpolation percentile (numpy default).
                if sz % 2 == 1:
                    med = slc[sz // 2]
                else:
                    med = 0.5 * (slc[sz // 2 - 1] + slc[sz // 2])
                # 25% / 75% via linear interpolation between adjacent positions.
                p25_pos = 0.25 * (sz - 1)
                p75_pos = 0.75 * (sz - 1)
                lo25 = int(p25_pos)
                hi25 = min(lo25 + 1, sz - 1)
                f25 = p25_pos - lo25
                q25 = slc[lo25] * (1.0 - f25) + slc[hi25] * f25
                lo75 = int(p75_pos)
                hi75 = min(lo75 + 1, sz - 1)
                f75 = p75_pos - lo75
                q75 = slc[lo75] * (1.0 - f75) + slc[hi75] * f75
                meds[i] = med
                iqr = q75 - q25
                if iqr > 1e-6:
                    iqrs[i] = iqr
        return meds, iqrs, sizes


def _median_v4_njit_sort(y: np.ndarray, bin_idx: np.ndarray, n_bins: int) -> np.ndarray:
    if not _HAS_NB:
        return _median_v1_pyloop(y, bin_idx, n_bins)
    order = np.argsort(bin_idx, kind="stable")
    sorted_bins = bin_idx[order]
    sorted_y = np.ascontiguousarray(y[order])
    starts = np.searchsorted(sorted_bins, np.arange(n_bins), side="left").astype(np.int64)
    ends = np.searchsorted(sorted_bins, np.arange(n_bins), side="right").astype(np.int64)
    global_med = float(np.median(y))
    return _median_v4_njit_kernel(sorted_y, starts, ends, n_bins, global_med)


def _quantile_v4_njit_sort(y: np.ndarray, bin_idx: np.ndarray, n_bins: int):
    if not _HAS_NB:
        return _quantile_v1_pyloop(y, bin_idx, n_bins)
    order = np.argsort(bin_idx, kind="stable")
    sorted_bins = bin_idx[order]
    sorted_y = np.ascontiguousarray(y[order])
    starts = np.searchsorted(sorted_bins, np.arange(n_bins), side="left").astype(np.int64)
    ends = np.searchsorted(sorted_bins, np.arange(n_bins), side="right").astype(np.int64)
    global_med = float(np.median(y))
    global_iqr = max(float(np.subtract(*np.percentile(y, [75, 25]))), 1e-6)
    return _quantile_v4_njit_kernel(sorted_y, starts, ends, n_bins, global_med, global_iqr)


def _median_v3_pandas_groupby(y: np.ndarray, bin_idx: np.ndarray, n_bins: int) -> np.ndarray:
    global_med = float(np.median(y))
    grp = pd.Series(y).groupby(bin_idx, sort=True).median()
    out = np.full(n_bins, global_med, dtype=np.float64)
    out[grp.index.to_numpy()] = grp.to_numpy()
    return out


def _quantile_v1_pyloop(y: np.ndarray, bin_idx: np.ndarray, n_bins: int):
    global_med = float(np.median(y))
    global_iqr = max(float(np.subtract(*np.percentile(y, [75, 25]))), 1e-6)
    meds = np.full(n_bins, global_med, dtype=np.float64)
    iqrs = np.full(n_bins, global_iqr, dtype=np.float64)
    sizes = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        mask = bin_idx == i
        n = int(mask.sum())
        sizes[i] = n
        if n >= 4:
            bin_y = y[mask]
            meds[i] = float(np.median(bin_y))
            iqr = float(np.subtract(*np.percentile(bin_y, [75, 25])))
            iqrs[i] = iqr if iqr > 1e-6 else global_iqr
    return meds, iqrs, sizes


def _quantile_v2_sort_split(y: np.ndarray, bin_idx: np.ndarray, n_bins: int):
    global_med = float(np.median(y))
    global_iqr = max(float(np.subtract(*np.percentile(y, [75, 25]))), 1e-6)
    meds = np.full(n_bins, global_med, dtype=np.float64)
    iqrs = np.full(n_bins, global_iqr, dtype=np.float64)
    order = np.argsort(bin_idx, kind="stable")
    sorted_bins = bin_idx[order]
    sorted_y = y[order]
    starts = np.searchsorted(sorted_bins, np.arange(n_bins), side="left")
    ends = np.searchsorted(sorted_bins, np.arange(n_bins), side="right")
    sizes = (ends - starts).astype(np.int64)
    for i in range(n_bins):
        n = int(sizes[i])
        if n >= 4:
            sl = sorted_y[starts[i] : ends[i]]
            meds[i] = float(np.median(sl))
            iqr = float(np.subtract(*np.percentile(sl, [75, 25])))
            iqrs[i] = iqr if iqr > 1e-6 else global_iqr
    return meds, iqrs, sizes


def _quantile_v3_pandas_groupby(y: np.ndarray, bin_idx: np.ndarray, n_bins: int):
    global_med = float(np.median(y))
    global_iqr = max(float(np.subtract(*np.percentile(y, [75, 25]))), 1e-6)
    meds = np.full(n_bins, global_med, dtype=np.float64)
    iqrs = np.full(n_bins, global_iqr, dtype=np.float64)
    sizes = np.zeros(n_bins, dtype=np.int64)
    ser = pd.Series(y)
    gb = ser.groupby(bin_idx, sort=True)
    qs = gb.quantile([0.25, 0.5, 0.75]).unstack()
    counts = gb.count()
    idx = qs.index.to_numpy()
    q25 = qs[0.25].to_numpy()
    q50 = qs[0.5].to_numpy()
    q75 = qs[0.75].to_numpy()
    keep = counts.to_numpy() >= 4
    sizes[idx] = counts.to_numpy()
    meds[idx[keep]] = q50[keep]
    raw_iqr = q75[keep] - q25[keep]
    iqrs[idx[keep]] = np.where(raw_iqr > 1e-6, raw_iqr, global_iqr)
    return meds, iqrs, sizes


def _time_call(fn, *args, repeat: int = 5) -> float:
    fn(*args)  # warm
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        dt = (time.perf_counter() - t0) * 1000.0
        if dt < best:
            best = dt
    return best


def _make_inputs(n: int, n_bins: int, seed: int = 17):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float64)
    base = rng.standard_normal(n).astype(np.float64)
    edges = np.quantile(base, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    n_bins_eff = edges.size - 1
    bin_idx = np.clip(np.searchsorted(edges[1:-1], base, side="right"), 0, n_bins_eff - 1)
    return y, bin_idx.astype(np.intp), n_bins_eff


def run() -> dict:
    sizes = [(100_000, 10), (100_000, 20), (1_000_000, 10), (1_000_000, 20)]
    out = {"median": [], "quantile": []}
    for n, n_bins in sizes:
        y, bin_idx, n_bins_eff = _make_inputs(n, n_bins)

        t_v1 = _time_call(_median_v1_pyloop, y, bin_idx, n_bins_eff)
        t_v2 = _time_call(_median_v2_sort_split, y, bin_idx, n_bins_eff)
        t_v3 = _time_call(_median_v3_pandas_groupby, y, bin_idx, n_bins_eff)
        t_v4 = _time_call(_median_v4_njit_sort, y, bin_idx, n_bins_eff)
        ref = _median_v1_pyloop(y, bin_idx, n_bins_eff)
        for name, fn in (("v2", _median_v2_sort_split), ("v3", _median_v3_pandas_groupby), ("v4", _median_v4_njit_sort)):
            arr = fn(y, bin_idx, n_bins_eff)
            np.testing.assert_allclose(arr, ref, rtol=1e-12, err_msg=f"median {name} n={n} nb={n_bins_eff}")
        print(f"MED  n={n:>8d} nb={n_bins_eff:>2d}  v1={t_v1:7.2f}ms  v2={t_v2:7.2f}ms  v3={t_v3:7.2f}ms  v4={t_v4:7.2f}ms  (v4/v1: {t_v1 / t_v4:.2f}x)")
        out["median"].append({"n": n, "n_bins": n_bins_eff, "v1_ms": round(t_v1, 3), "v2_ms": round(t_v2, 3), "v3_ms": round(t_v3, 3), "v4_ms": round(t_v4, 3)})

        t_q1 = _time_call(_quantile_v1_pyloop, y, bin_idx, n_bins_eff)
        t_q2 = _time_call(_quantile_v2_sort_split, y, bin_idx, n_bins_eff)
        t_q3 = _time_call(_quantile_v3_pandas_groupby, y, bin_idx, n_bins_eff)
        t_q4 = _time_call(_quantile_v4_njit_sort, y, bin_idx, n_bins_eff)
        ref_q = _quantile_v1_pyloop(y, bin_idx, n_bins_eff)
        for name, fn in (("v2", _quantile_v2_sort_split), ("v3", _quantile_v3_pandas_groupby), ("v4", _quantile_v4_njit_sort)):
            arr = fn(y, bin_idx, n_bins_eff)
            np.testing.assert_allclose(arr[0], ref_q[0], rtol=1e-12, err_msg=f"qmed {name} n={n}")
            np.testing.assert_allclose(arr[1], ref_q[1], rtol=1e-12, err_msg=f"qiqr {name} n={n}")
            np.testing.assert_array_equal(arr[2], ref_q[2], err_msg=f"qsize {name} n={n}")
        print(f"QUAN n={n:>8d} nb={n_bins_eff:>2d}  v1={t_q1:7.2f}ms  v2={t_q2:7.2f}ms  v3={t_q3:7.2f}ms  v4={t_q4:7.2f}ms  (v4/v1: {t_q1 / t_q4:.2f}x)")
        out["quantile"].append({"n": n, "n_bins": n_bins_eff, "v1_ms": round(t_q1, 3), "v2_ms": round(t_q2, 3), "v3_ms": round(t_q3, 3), "v4_ms": round(t_q4, 3)})

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = _RESULTS / f"bench_median_quantile_residual_{ts}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults: {path}")
    return out


if __name__ == "__main__":
    run()
