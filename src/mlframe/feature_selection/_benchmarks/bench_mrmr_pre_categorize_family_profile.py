"""cProfile pass over MRMR's pre-categorization block (2026-07-09 audit follow-up).

The audit found ~40 default-ON pre-FE univariate/pairwise generator families running on the full raw,
unquantized frame BEFORE discretization -- but its own recommendation required profiling to attribute
cost per family before any gating change (a blind gate risks losing real recall for families that turn
out cheap). This script profiles one fit's pre-categorize block (everything before the "categorizing
dataset..." log line) at a representative wide-p shape and reports the top hotspots by cumulative time,
so any follow-up gating decision is measurement-backed, not guessed.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_mrmr_pre_categorize_family_profile``
"""
from __future__ import annotations

import cProfile
import pstats
from io import StringIO

import numpy as np
import pandas as pd


def _make_dataset(n_rows: int, n_cols: int, n_informative: int, n_categorical: int, seed: int):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n_rows) for i in range(n_cols - n_categorical)})
    for j in range(n_categorical):
        X[f"cat{j}"] = rng.integers(0, 12, size=n_rows).astype(np.int64)
    coefs = rng.normal(size=n_informative)
    y = X.iloc[:, :n_informative].to_numpy() @ coefs + 0.3 * rng.standard_normal(n_rows)
    return X, pd.Series(y, name="y")


def main():
    N_ROWS = 60_000  # representative wide-p shape at a tractable profiling n; family-level RELATIVE cost
    N_COLS = 420     # is expected to hold reasonably across n since most families are O(n) or O(n*p).
    N_CATEGORICAL = 20
    N_INFORMATIVE = 6

    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _make_dataset(N_ROWS, N_COLS, N_INFORMATIVE, N_CATEGORICAL, seed=0)
    print(f"Dataset: {N_ROWS} rows x {N_COLS} cols ({N_CATEGORICAL} categorical, {N_INFORMATIVE} informative)")

    m = MRMR(n_jobs=4, n_workers=1, random_seed=42, verbose=0, fe_max_steps=0, dcd_enable=False)

    pr = cProfile.Profile()
    pr.enable()
    m.fit(X, y)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    report = s.getvalue()
    print(report)

    print("\n--- filtered to mlframe feature_engineering / FE-family source lines ---")
    s2 = StringIO()
    ps2 = pstats.Stats(pr, stream=s2).sort_stats("cumulative")
    ps2.print_stats(r"mlframe.*feature_selection.*filters")
    print(s2.getvalue()[:8000])


if __name__ == "__main__":
    main()
