"""iter106 A/B bench: rolling_zero_crossings @10M.

OLD (numpy multi-pass per segment: np.sign + slice products + boolean + sum.axis=1 + astype)
vs NEW (single @njit(parallel) prange kernel folding all passes into one per-window count).

Run:
    PYTHONPATH=src python src/mlframe/feature_engineering/_benchmarks/bench_rolling_zero_crossings_iter106.py
"""
from __future__ import annotations

import sys
import time

sys.modules.setdefault("cupy", None)
import scipy.stats  # noqa: F401,E402  pre-import to dodge py3.14 cold-import segfault
import numba  # noqa: F401,E402
import numpy as np  # noqa: E402

from mlframe.feature_engineering import windowed_shape as ws  # noqa: E402
from mlframe.feature_engineering.grouped import per_group_sliding_window  # noqa: E402


def _old_zero_crossings(values, group_ids, window_K=20, center="zero", fill_value=np.nan):
    if center not in {"zero", "median", "mean"}:
        raise ValueError(center)
    out = np.full(values.size, fill_value, dtype=np.float64)
    for _si, wins, write_idx in per_group_sliding_window(values, group_ids, window_K=window_K):
        if center == "median":
            c = np.median(wins, axis=1, keepdims=True)
        elif center == "mean":
            c = wins.mean(axis=1, keepdims=True)
        else:
            c = 0.0
        s = wins - c
        s_sign = np.sign(s)
        cross = (s_sign[:, 1:] * s_sign[:, :-1]) < 0
        out[write_idx] = cross.sum(axis=1).astype(np.float64)
    return out


def main():
    n = 10_000_000
    rng = np.random.default_rng(0)
    values = rng.standard_normal(n)
    group_ids = np.zeros(n, dtype=np.int64)  # single group; worst case for kernel work
    K = 20

    for center in ("zero", "median", "mean"):
        # identity
        old = _old_zero_crossings(values[:200000], group_ids[:200000], window_K=K, center=center)
        new = ws.rolling_zero_crossings(values[:200000], group_ids[:200000], window_K=K, center=center)
        fin = ~np.isnan(old)
        assert np.array_equal(old[fin], new[fin]), f"identity FAIL center={center}"
        print(f"identity OK center={center}")

    # warm new kernel at full size
    ws.rolling_zero_crossings(values[:1000], group_ids[:1000], window_K=K, center="zero")

    for center in ("zero", "median", "mean"):
        told, tnew = [], []
        for _ in range(5):
            t0 = time.perf_counter(); _old_zero_crossings(values, group_ids, window_K=K, center=center); told.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); ws.rolling_zero_crossings(values, group_ids, window_K=K, center=center); tnew.append(time.perf_counter() - t0)
        mo, mn = min(told), min(tnew)
        print(f"center={center:6s} OLD min={mo:.3f}s NEW min={mn:.3f}s speedup={mo/mn:.2f}x  (old={[f'{x:.2f}' for x in told]} new={[f'{x:.2f}' for x in tnew]})")


if __name__ == "__main__":
    main()
