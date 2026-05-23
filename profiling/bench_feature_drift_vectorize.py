"""Bench vectorized vs per-col-loop feature_distribution_drift.

c0122 iter182: compute_feature_distribution_drift runs once per fit at
~83ms cumtime / 7ms self. Per-col loop extracts 3 column views per
column (train, val, test) -- for polars this is 3 Series builds + 3
to_numpy conversions per col. Vectorising via df[cols].mean() /
df[cols].std() reduces to 6 total ops (regardless of column count).

Run: ``python profiling/bench_feature_drift_vectorize.py``
"""
import time
import math
import numpy as np
import pandas as pd


def drift_loop(train_df, val_df, test_df, cols):
    """Current per-col-loop form."""
    per_feature = {}
    for col in cols:
        train_vals = train_df[col].to_numpy().astype(np.float64)
        train_vals = train_vals[np.isfinite(train_vals)]
        if train_vals.size < 2:
            continue
        train_mean = float(np.mean(train_vals))
        train_std = float(np.std(train_vals))
        if train_std <= 0.0:
            per_feature[col] = {"train_mean": train_mean, "train_std": 0.0,
                                "val_z": float("nan"), "test_z": float("nan")}
            continue
        def _z_for(other_df):
            if other_df is None:
                return float("nan")
            other = other_df[col].to_numpy().astype(np.float64)
            other = other[np.isfinite(other)]
            if other.size < 2:
                return float("nan")
            return float((np.mean(other) - train_mean) / train_std)
        per_feature[col] = {
            "train_mean": train_mean, "train_std": train_std,
            "val_z": _z_for(val_df), "test_z": _z_for(test_df),
        }
    return per_feature


def drift_vec(train_df, val_df, test_df, cols):
    """Vectorised: 6 ops total (train mean/std, val mean, test mean),
    per-col loop is pure dict-build."""
    # Pandas .mean()/.std() with default skipna=True replicates
    # np.mean(arr[isfinite(arr)]) semantics for typical float cols.
    # ddof=0 matches np.std default.
    train_means = train_df[cols].mean()
    train_stds = train_df[cols].std(ddof=0)
    val_means = val_df[cols].mean() if val_df is not None else None
    test_means = test_df[cols].mean() if test_df is not None else None
    per_feature = {}
    for col in cols:
        train_mean = float(train_means[col])
        train_std = float(train_stds[col])
        if not math.isfinite(train_std) or train_std <= 0.0:
            if math.isfinite(train_mean):
                per_feature[col] = {"train_mean": train_mean, "train_std": 0.0,
                                    "val_z": float("nan"), "test_z": float("nan")}
            continue
        val_z = float((val_means[col] - train_mean) / train_std) if val_means is not None and math.isfinite(val_means[col]) else float("nan")
        test_z = float((test_means[col] - train_mean) / train_std) if test_means is not None and math.isfinite(test_means[col]) else float("nan")
        per_feature[col] = {
            "train_mean": train_mean, "train_std": train_std,
            "val_z": val_z, "test_z": test_z,
        }
    return per_feature


def bench(label, fn, train_df, val_df, test_df, cols, n_iter=20):
    fn(train_df, val_df, test_df, cols); fn(train_df, val_df, test_df, cols)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(train_df, val_df, test_df, cols)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e3


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for n, K in [(10_000, 10), (100_000, 10), (100_000, 30), (200_000, 30)]:
        cols = [f"f{i}" for i in range(K)]
        train_df = pd.DataFrame(rng.standard_normal((n, K)), columns=cols)
        val_df = pd.DataFrame(rng.standard_normal((n // 5, K)), columns=cols)
        test_df = pd.DataFrame(rng.standard_normal((n // 5, K)), columns=cols)
        t_loop = bench("loop", drift_loop, train_df, val_df, test_df, cols)
        t_vec = bench("vec", drift_vec, train_df, val_df, test_df, cols)
        # Equivalence sanity
        l = drift_loop(train_df, val_df, test_df, cols)
        v = drift_vec(train_df, val_df, test_df, cols)
        # Max abs z-diff
        max_diff = 0.0
        for col in cols:
            for k in ("val_z", "test_z"):
                lv = l[col].get(k, float("nan")); vv = v[col].get(k, float("nan"))
                if math.isfinite(lv) and math.isfinite(vv):
                    max_diff = max(max_diff, abs(lv - vv))
        print(f"n={n:>7} K={K}: loop={t_loop:6.2f}ms vec={t_vec:6.2f}ms ({t_loop/t_vec:.2f}x) z_diff={max_diff:.2e}")
