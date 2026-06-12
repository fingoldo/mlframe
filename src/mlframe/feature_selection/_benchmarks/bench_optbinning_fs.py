"""cProfile harness for the optbinning IV/WoE feature-selection orchestration.

SCOPE: ``mlframe.feature_selection.optbinning.get_binningprocess_featureselectors``
-- the thin factory that wires sklearn ``Pipeline`` objects around the optbinning
``BinningProcess`` (IV/WoE binner) and a ``category_encoders.CatBoostEncoder``.

Representative frame: n=2000, ~20 mixed columns (14 numeric + 6 low-card categorical),
binary target. We profile both the factory call and the full ``fit`` of the two FS
pipelines (``bp_nocats_fs`` numeric-only, ``bp_withcats_fs`` CatBoostEncoder->BP).

VERDICT (2026-06-12): NO ACTIONABLE mlframe-SIDE SPEEDUP.

The mlframe-side orchestration in ``get_binningprocess_featureselectors`` is a pure
O(n_cols) factory: two ``.columns.tolist()`` calls, one ``features.head().select_dtypes``
+ ``list.remove`` loop over the categorical columns, and four ``Pipeline`` /
``BinningProcess`` constructions. On the n=2000 x 20 frame the entire factory call is
sub-millisecond and attributes to ~0.0% of fit wall.

100% of the fit wall is inside libraries that are explicitly OFF-LIMITS:
  - ``optbinning.BinningProcess.fit`` -> ``OptimalBinning`` per column (the CP/LP
    binning optimizer + IV/WoE table computation). This dominates.
  - ``category_encoders.CatBoostEncoder.fit_transform`` (withcats path only).
Plus numpy/pandas attribution noise inside those libraries.

There is no mlframe-side per-column Python IV loop to vectorize (the per-column IV is
computed by optbinning's own ``binning_table.iv``), no redundant binner re-fit (each
``BinningProcess`` is fitted once; the selection gate reads ``iv`` off the already-fitted
tables), and no per-column frame rebuild. The factory's ``select_dtypes("category")``
already runs on ``features.head()`` (5 rows) rather than the full frame, so even the dtype
scan is already cheap.

Conclusion: this path is optbinning-library-bound (analogous to the BorutaShap case being
shap-bound). No clean, bit-identical mlframe-side win exists. Run this harness to re-confirm
on different hardware / shapes:

    PYTHONPATH=src CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 \
        python src/mlframe/feature_selection/_benchmarks/bench_optbinning_fs.py
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.optbinning import get_binningprocess_featureselectors


def make_frame(n: int = 2000, n_num: int = 14, n_cat: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-2.0 * x))
    y = (rng.uniform(size=n) < p).astype(int)
    data = {"signal_num": x}
    for i in range(n_num - 1):
        data[f"noise_num_{i}"] = rng.normal(size=n)
    for j in range(n_cat):
        codes = rng.integers(0, 4 + j % 3, size=n)
        data[f"cat_{j}"] = pd.Categorical([f"L{c}" for c in codes])
    df = pd.DataFrame(data)
    return df, pd.Series(y, name="y")


def run_once(df, y):
    num_df = df.select_dtypes(exclude="category")
    bp_withcats_fs, _, _, _ = get_binningprocess_featureselectors(df, n_jobs=1)
    _, _, bp_nocats_fs, _ = get_binningprocess_featureselectors(num_df, n_jobs=1)
    bp_nocats_fs.fit(num_df, y)
    bp_withcats_fs.fit(df, y)


def main():
    df, y = make_frame()

    # warm
    run_once(df.copy(), y.copy())

    reps = 3
    t0 = time.perf_counter()
    for _ in range(reps):
        run_once(df.copy(), y.copy())
    wall = (time.perf_counter() - t0) / reps
    print(f"mean fit wall over {reps} reps: {wall*1000:.1f} ms")

    pr = cProfile.Profile()
    pr.enable()
    run_once(df.copy(), y.copy())
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(25)
    print(s.getvalue())

    # isolate the mlframe-side factory cost
    pr2 = cProfile.Profile()
    pr2.enable()
    for _ in range(100):
        get_binningprocess_featureselectors(df, n_jobs=1)
    pr2.disable()
    s2 = io.StringIO()
    pstats.Stats(pr2, stream=s2).sort_stats("cumulative").print_stats(8)
    print("=== factory-only (x100) ===")
    print(s2.getvalue())


if __name__ == "__main__":
    main()
