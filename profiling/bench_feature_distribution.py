"""Bench analyze_feature_distribution at production shape (200k rows, ~15 numeric).

c0139 iter114 (2026-05-21) profile: 0.49s cumtime on a single call. Used to
identify the leakage loop as the dominant detector (~145 ms of 270 ms total)
and to validate the corrwith vectorisation that landed in the same iter
(post-fix: ~190 ms full, leakage ~65 ms).

Also documents the bench-attempt-rejected numeric-loop vectorisation: a
naive (N, F) block + nanmean/nanstd replacement of the per-column low-var
loop was slightly SLOWER, see the comment at
_target_distribution_analyzer.py:`# bench-attempt-rejected (2026-05-21)`.

Run: ``python profiling/bench_feature_distribution.py``
"""

from __future__ import annotations
import time
import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd

from mlframe.training._target_distribution_analyzer import analyze_feature_distribution


def make_synthetic(n_rows: int = 200_000, n_numeric: int = 15, n_cat: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    cols = {}
    for j in range(n_numeric):
        x = rng.standard_normal(n_rows).astype(np.float64)
        if j == 5:
            x[:] = 0.0  # low-var test
        # Sprinkle some NaNs
        if j in (2, 7):
            mask = rng.random(n_rows) < 0.05
            x[mask] = np.nan
        cols[f"num_{j}"] = x
    for j in range(n_cat):
        cols[f"cat_{j}"] = rng.choice(['a', 'b', 'c'], size=n_rows)
    df = pd.DataFrame(cols)
    y = (rng.standard_normal(n_rows) + 0.1 * df['num_0']).to_numpy()  # one feature with light leak
    return df, y


df, y = make_synthetic()
print(f'X shape: {df.shape}, n_numeric=15, n_cat=3, n_nan_cols=2')

# Warmup
_ = analyze_feature_distribution(df.head(5000), y[:5000])

# Bench
for _ in range(3):
    t = time.perf_counter()
    rep = analyze_feature_distribution(df, y)
    print(f'  {time.perf_counter() - t:.3f}s  pathologies={rep.pathologies}')

# Bench per-detector by toggling caps
print('\nWith leakage_corr_threshold=1.1 (skip leakage detection):')
for _ in range(3):
    t = time.perf_counter()
    rep = analyze_feature_distribution(df, y, leakage_corr_threshold=1.1)
    print(f'  {time.perf_counter() - t:.3f}s')

print('\nNo y (skip leakage):')
for _ in range(3):
    t = time.perf_counter()
    rep = analyze_feature_distribution(df, y=None)
    print(f'  {time.perf_counter() - t:.3f}s')

print('\nredundancy_max_numeric_features=0 (skip redundancy):')
for _ in range(3):
    t = time.perf_counter()
    rep = analyze_feature_distribution(df, y, redundancy_max_numeric_features=0)
    print(f'  {time.perf_counter() - t:.3f}s')
