"""A/B bench: MRMR + cat-FE permutation test with full-N vs subsampled
permutation null on 1M-row inputs.

Wave 13b. ``CatFEConfig.permutation_subsample`` lets the caller cap the
sample count fed to ``_count_nfailed_joint_indep_prange`` -- ii_obs
stays on the full N rows; only the per-shuffle MI is computed on a
subset. Stat-vs-runtime trade-off; documented on the config field.

Bench measures end-to-end MRMR.fit wall on the multiplicative-synergy
fixture (1M rows, 3 numeric + 5 categorical) under three configs:
  - full-N permutation (default, baseline)
  - permutation_subsample=100_000
  - permutation_subsample=50_000

Usage:
    python -m mlframe.profiling.bench_mrmr_subsample
"""

from __future__ import annotations

import statistics
import time
import warnings

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig


def _build_synthetic_data(n: int = 1_000_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4, n).astype(np.int32)
    b = rng.integers(0, 4, n).astype(np.int32)
    c = rng.integers(0, 8, n).astype(np.int32)
    d = rng.normal(size=n).astype(np.float64)
    e = rng.normal(size=n).astype(np.float64)
    # y depends on a*b (synergy) + d
    y_raw = (a == b).astype(np.int32) + (d > 0).astype(np.int32) * 2
    df = pd.DataFrame({
        "a": pd.Categorical(a.astype(str)),
        "b": pd.Categorical(b.astype(str)),
        "c": pd.Categorical(c.astype(str)),
        "d": d, "e": e,
    })
    y = pd.Series(y_raw, name="target")
    return df, y


def bench_one(label: str, cat_fe_cfg: CatFEConfig, df, y, n_repeat: int = 1) -> None:
    # Clear cache between runs (otherwise the 2nd repeat hits the
    # process-wide _FIT_CACHE and reports 0 ms).
    MRMR._FIT_CACHE.clear()
    times = []
    for _ in range(n_repeat):
        MRMR._FIT_CACHE.clear()
        m = MRMR(
            full_npermutations=20,
            baseline_npermutations=20,
            fe_max_steps=0,
            cat_fe_config=cat_fe_cfg,
            verbose=0,
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            m.fit(df, y)
            times.append(time.perf_counter() - t0)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    print(f"  {label:<55} {mean:>8.2f} s +/- {std:>5.2f} s")


def main() -> None:
    print("Building synthetic data (n=1_000_000)...")
    df, y = _build_synthetic_data(n=1_000_000)
    print("done.\n")

    print(f"# MRMR.fit wall on 1M rows, full_npermutations=20\n")
    bench_one(
        "FULL (permutation_subsample=None)",
        CatFEConfig(enable=True, full_npermutations=20),
        df, y,
    )
    bench_one(
        "SUBSAMPLE 100_000",
        CatFEConfig(enable=True, full_npermutations=20, permutation_subsample=100_000),
        df, y,
    )
    bench_one(
        "SUBSAMPLE 50_000",
        CatFEConfig(enable=True, full_npermutations=20, permutation_subsample=50_000),
        df, y,
    )


if __name__ == "__main__":
    main()
