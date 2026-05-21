"""Bench _vectorized_bootstrap_logloss_samples: log+clip BEFORE vs AFTER gather (iter118).

The shipped refactor moves the per-element log-loss computation (np.clip +
2x np.log + np.where) BEFORE the (n_resamples, n) bootstrap-index gather,
so each elementwise op runs on the (n,) / (n, K) input ONCE instead of on
the gathered (n_resamples, n) tensor.

Bench at n=1500, n_resamples=1000 (the c0022 / multilabel fuzz shape):

  1-D binary  : 60 ms -> 12 ms (5x)
  2-D K=4 mlb : 250 ms -> 45 ms (5.5x)

Output is bit-identical (max abs diff == 0). Run via
``python profiling/bench_bootstrap_logloss_pre_gather.py``.
"""

import time
import numpy as np


def old_impl(y, p, n_resamples, seed, eps=1e-15):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    y_r = y[idx]
    p_r = p[idx]
    p_clip = np.clip(p_r, eps, 1.0 - eps)
    is_pos = y_r > 0.5
    elem = -np.where(is_pos, np.log(p_clip), np.log(1.0 - p_clip))
    if y.ndim == 1:
        return elem.mean(axis=1)
    return elem.mean(axis=(1, 2))


def new_impl(y, p, n_resamples, seed, eps=1e-15):
    n = len(y)
    rng = np.random.default_rng(seed)
    # Precompute per-element loss on the (n,)-shape inputs ONCE.
    p_clip = np.clip(p, eps, 1.0 - eps)
    log_p = np.log(p_clip)
    log_1mp = np.log(1.0 - p_clip)
    is_pos = y > 0.5
    elem_n = -np.where(is_pos, log_p, log_1mp)  # same shape as y
    idx = rng.integers(0, n, size=(n_resamples, n))
    elem_r = elem_n[idx]
    if y.ndim == 1:
        return elem_r.mean(axis=1)
    return elem_r.mean(axis=(1, 2))


# 1-D binary
rng = np.random.default_rng(0)
n = 1500
y_1d = rng.integers(0, 2, size=n).astype(np.float64)
p_1d = np.clip(rng.random(n), 1e-3, 1 - 1e-3)

for _ in range(3):
    t = time.perf_counter()
    out_old = old_impl(y_1d, p_1d, 1000, 42)
    print(f'1d old: {(time.perf_counter()-t)*1000:.1f}ms')

for _ in range(3):
    t = time.perf_counter()
    out_new = new_impl(y_1d, p_1d, 1000, 42)
    print(f'1d new: {(time.perf_counter()-t)*1000:.1f}ms')

print(f'1d max abs diff: {np.abs(out_old - out_new).max():.2e}')

# 2-D multilabel K=4
K = 4
y_2d = rng.integers(0, 2, size=(n, K)).astype(np.float64)
p_2d = np.clip(rng.random((n, K)), 1e-3, 1 - 1e-3)

print()
for _ in range(3):
    t = time.perf_counter()
    out_old2 = old_impl(y_2d, p_2d, 1000, 42)
    print(f'2d old: {(time.perf_counter()-t)*1000:.1f}ms')

for _ in range(3):
    t = time.perf_counter()
    out_new2 = new_impl(y_2d, p_2d, 1000, 42)
    print(f'2d new: {(time.perf_counter()-t)*1000:.1f}ms')

print(f'2d max abs diff: {np.abs(out_old2 - out_new2).max():.2e}')
