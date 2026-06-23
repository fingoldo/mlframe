"""Bench: cusum_features Page-Hinkley walk -- pure-Python scalar loop vs numba njit.

Target: stationarity.cusum_features._walk inner loop (branch-dependent CUSUM
state with resets). A genuinely serial recurrence over the full series; the
pure-Python loop pays Python-frame overhead per element. The njit kernel
performs the IDENTICAL float64 arithmetic in the IDENTICAL order (same
nanmean-as-sum/count, same max/min/threshold branches) so it is bit-identical.

Run:
    python -m mlframe.feature_engineering._benchmarks.bench_cusum_walk_njit
"""

from __future__ import annotations

import time

import numpy as np

from mlframe.feature_engineering.stationarity import cusum_features


def _old_cusum(values, threshold=None, *, group_ids=None, drift=0.0):
    """Verbatim copy of the PRE-optimization cusum_features (pure-Python loop)."""
    from mlframe.feature_engineering.grouped import iter_group_segments

    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if threshold is None:
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            threshold = 1.0
        else:
            mad = float(np.median(np.abs(finite - np.median(finite))))
            threshold = 5.0 * mad * 1.4826 if mad > 0 else 1.0

    out_pos = np.zeros(n, dtype=np.float64)
    out_neg = np.zeros(n, dtype=np.float64)
    out_since = np.zeros(n, dtype=np.float64)
    out_count = np.zeros(n, dtype=np.float64)

    def _walk(idx_seg):
        m = idx_seg.size
        if m == 0:
            return
        seg = arr[idx_seg]
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0
        pos = 0.0
        neg = 0.0
        rows_since = 0.0
        n_resets = 0
        for i in range(m):
            x = seg[i]
            if not np.isfinite(x):
                out_pos[idx_seg[i]] = pos
                out_neg[idx_seg[i]] = neg
                out_since[idx_seg[i]] = rows_since
                out_count[idx_seg[i]] = n_resets
                rows_since += 1
                continue
            dev = x - seg_mean
            pos = max(0.0, pos + dev - drift)
            neg = min(0.0, neg + dev + drift)
            triggered = (pos > threshold) or (neg < -threshold)
            if triggered:
                pos = 0.0
                neg = 0.0
                rows_since = 0.0
                n_resets += 1
            else:
                rows_since += 1
            out_pos[idx_seg[i]] = pos
            out_neg[idx_seg[i]] = neg
            out_since[idx_seg[i]] = rows_since
            out_count[idx_seg[i]] = n_resets

    if group_ids is None:
        _walk(np.arange(n))
    else:
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            _walk(sort_idx[s:e])

    return {
        "cusum_pos": out_pos,
        "cusum_neg": out_neg,
        "rows_since_reset": out_since,
        "n_resets_in_window": out_count,
    }


def _bench(fn, *args, n_iter=20):
    best = float("inf")
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    keys = ("cusum_pos", "cusum_neg", "rows_since_reset", "n_resets_in_window")
    for n in (2000, 50_000, 1_000_000):
        x = rng.standard_normal(n).cumsum()
        # sprinkle some NaNs to exercise the non-finite branch
        x[rng.integers(0, n, size=max(1, n // 50))] = np.nan

        new = cusum_features(x)
        old = _old_cusum(x)
        ident = all(np.array_equal(new[k], old[k], equal_nan=True) for k in keys)
        max_abs = max(np.nanmax(np.abs(new[k] - old[k])) for k in keys) if not ident else 0.0

        cusum_features(x[:100])  # warm njit

        t_old = _bench(lambda: _old_cusum(x))
        t_new = _bench(lambda: cusum_features(x))
        print(f"n={n:>9} OLD={t_old*1e3:9.3f}ms NEW={t_new*1e3:9.3f}ms "
              f"speedup={t_old/t_new:5.2f}x identical={ident} max_abs={max_abs:.2e}")


if __name__ == "__main__":
    main()
