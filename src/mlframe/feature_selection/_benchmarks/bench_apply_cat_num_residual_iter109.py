"""iter109 A/B bench: apply_cat_num_residual per-row Python loop vs pd.factorize vectorization @10M.

The count/freq replay paths (apply_count_encoding / apply_frequency_encoding) were vectorized with pd.factorize
(O(n) hashtable, ~6-8x over the per-row dict.get loop); the cat x num residual replay was left as a scalar
``for i in range(len(cats))`` loop with a dict.get + float arithmetic per row. At 10M rows that is 10M Python
iterations. This bench measures the OLD scalar loop vs the NEW factorize-gather, both isolated, and asserts
bit-identical output.

Run: python -m mlframe.feature_selection._benchmarks.bench_apply_cat_num_residual_iter109
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _old_apply(cats: np.ndarray, num_vals: np.ndarray, lookup: dict, global_mean: float) -> np.ndarray:
    finite = np.isfinite(num_vals)
    out = np.zeros(len(cats), dtype=np.float64)
    for i in range(len(cats)):
        if not finite[i]:
            out[i] = 0.0
        else:
            cell_mean = float(lookup.get(cats[i], global_mean))
            out[i] = num_vals[i] - cell_mean
    return out


def _new_apply(cats: np.ndarray, num_vals: np.ndarray, lookup: dict, global_mean: float) -> np.ndarray:
    finite = np.isfinite(num_vals)
    if cats.size == 0:
        return np.empty(0, dtype=np.float64)
    codes, uniques = pd.factorize(cats)
    cell = np.array([float(lookup.get(u, global_mean)) for u in uniques], dtype=np.float64)
    gathered = cell[codes]
    return np.where(finite, num_vals - gathered, 0.0).astype(np.float64, copy=False)


def _make_data(n: int, n_cats: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    cat_ids = rng.integers(0, n_cats, size=n)
    cats = np.array([f"c{c}" for c in cat_ids], dtype=object)
    num_vals = rng.normal(size=n).astype(np.float64)
    # ~2% NaN in num to exercise the finite mask.
    num_vals[rng.random(n) < 0.02] = np.nan
    global_mean = float(np.nanmean(num_vals))
    # lookup covers most cats; leave a few unseen to hit global_mean fallback.
    lookup = {f"c{c}": float(rng.normal()) for c in range(n_cats) if c % 7 != 0}
    return cats, num_vals, lookup, global_mean


def main():
    n = 10_000_000
    n_cats = 5000
    cats, num_vals, lookup, gm = _make_data(n, n_cats)

    # identity on a small slice + on the full data
    o_small = _old_apply(cats[:10000], num_vals[:10000], lookup, gm)
    nw_small = _new_apply(cats[:10000], num_vals[:10000], lookup, gm)
    assert np.array_equal(o_small, nw_small, equal_nan=True), "small-slice mismatch"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input

    # warm new path
    _new_apply(cats[:1000], num_vals[:1000], lookup, gm)

    best_old = best_new = float("inf")
    for _ in range(3):
        t = time.perf_counter(); o = _old_apply(cats, num_vals, lookup, gm); best_old = min(best_old, time.perf_counter() - t)
        t = time.perf_counter(); nw = _new_apply(cats, num_vals, lookup, gm); best_new = min(best_new, time.perf_counter() - t)
    assert np.array_equal(o, nw, equal_nan=True), "full-data mismatch"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input
    print(f"n={n} n_cats={n_cats}")
    print(f"OLD per-row loop:      {best_old*1000:9.1f} ms")
    print(f"NEW factorize-gather:  {best_new*1000:9.1f} ms")
    print(f"speedup: {best_old/best_new:.2f}x   bit-identical: True")


if __name__ == "__main__":
    main()
