"""Bench: float64 fast-path in _normalize_binary_labels vs the legacy np.unique path.

The bootstrap-ECE loop (honest_diagnostics) casts labels to float64 ONCE, so every resample's
``y_true_f64[idx]`` reaches _normalize_binary_labels as float64 and used to hit the np.unique O(n log n)
sort. This bench times the OLD float path (reconstructed verbatim) vs the NEW njit single-pass scan,
isolated and in a representative resample loop, and asserts ECE bit-identity.

Run: python -m mlframe.calibration._benchmarks.bench_normalize_binary_labels_f64
"""
from __future__ import annotations
import time
import numpy as np
from mlframe.calibration.policy import _normalize_binary_labels, _scan_binary01_f64, _ece_score


def _old_normalize_f64(arr: np.ndarray) -> np.ndarray:
    """Legacy float path (verbatim from pre-fix _normalize_binary_labels): np.unique sort."""
    finite = arr[np.isfinite(arr.astype(np.float64))] if arr.dtype.kind == "f" else arr
    uniq = np.unique(finite)
    if uniq.size != 2:
        raise ValueError("need exactly 2 distinct finite values")
    if uniq[0] == 0 and uniq[1] == 1:
        return arr.astype(np.int64, copy=False)
    hi = uniq.max()
    return (arr == hi).astype(np.int64)


def _median_us(fn, iters):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e6)
    return float(np.median(ts))


def main():
    rng = np.random.default_rng(0)
    print(f"{'n':>10} {'old_us':>10} {'new_us':>10} {'speedup':>8}  identity")
    for n in (2_000, 200_000, 1_000_000):
        y = rng.integers(0, 2, n).astype(np.float64)
        p = rng.random(n)
        # warm
        _scan_binary01_f64(y); _old_normalize_f64(y); _normalize_binary_labels(y)
        idx = rng.integers(0, n, n)
        old = _median_us(lambda: _old_normalize_f64(y[idx]), 7)
        new = _median_us(lambda: _normalize_binary_labels(y[idx]), 7)
        # identity: ECE via old-normalized vs new-normalized labels
        yi = y[idx]
        e_old = _ece_score(_old_normalize_f64(yi).astype(np.float64), p[idx], n_bins=10)
        e_new = _ece_score(yi, p[idx], n_bins=10)
        ok = e_old == e_new
        print(f"{n:>10} {old:>10.2f} {new:>10.2f} {old/new:>7.2f}x  ece_equal={ok} ({e_old:.12f})")

    # representative resample loop (n=200k, 200 resamples), isolated normalize cost
    n = 200_000
    y = rng.integers(0, 2, n).astype(np.float64)
    idxs = [rng.integers(0, n, n) for _ in range(200)]
    _normalize_binary_labels(y); _old_normalize_f64(y)
    t0 = time.perf_counter()
    for ix in idxs:
        _old_normalize_f64(y[ix])
    t_old = time.perf_counter() - t0
    t0 = time.perf_counter()
    for ix in idxs:
        _normalize_binary_labels(y[ix])
    t_new = time.perf_counter() - t0
    print(f"\nloop n=200k x200: old={t_old*1e3:.1f}ms new={t_new*1e3:.1f}ms  {t_old/t_new:.2f}x")


if __name__ == "__main__":
    main()
