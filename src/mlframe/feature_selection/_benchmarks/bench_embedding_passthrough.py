"""cProfile harness for the MRMR embedding / free-text passthrough feature.

Profiles ``MRMR.fit`` on a frame carrying an embedding-vector column + a free-text column alongside numeric features, isolating the cost of the new detection +
narrowing path (``detect_passthrough_columns`` + the column-subset narrow). Run::

    PYTHONPATH=src python -m mlframe.feature_selection._benchmarks.bench_embedding_passthrough

VERDICT (2026-06-12, n=20000, 6 numeric + 1 embedding(8) + 1 text): the passthrough machinery is NOT a hotspot. ``detect_passthrough_columns`` is sampling-bounded
(<= 50 cells/column regardless of frame height) and the narrow is a zero-copy column subselection; both are O(ncols) and disappear into cProfile attribution noise
(<0.5% of fit cumtime, dwarfed by categorize_dataset + the MI screen). No actionable speedup -- the detector's sampling cap is the right design and adding numba /
vectorisation to a <1ms once-per-fit pass would be pure over-engineering (see CLAUDE.md "When to skip the ladder").
"""
from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np
import pandas as pd


def _make_frame(n=20000, n_num=6, emb_dim=8, seed=0):
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    cols = {f"num_{i}": (sig if i == 0 else rng.normal(size=n)) + rng.normal(scale=0.1, size=n) for i in range(n_num)}
    cols["emb"] = [rng.normal(size=emb_dim).astype(np.float32) for _ in range(n)]
    cols["descr"] = ["a moderately long free-text description row number %d here" % i for i in range(n)]
    y = (sig > 0).astype(int)
    return pd.DataFrame(cols), y


def _run_fit(df, y):
    from mlframe.feature_selection.filters.mrmr import MRMR

    m = MRMR(max_runtime_mins=0.5, fe_max_steps=0, random_seed=0)
    m.fit(df, y)
    return m


def main():
    df, y = _make_frame()
    _run_fit(df.head(2000), y[:2000])  # warm numba / imports

    pr = cProfile.Profile()
    pr.enable()
    m = _run_fit(df, y)
    pr.disable()

    print(f"passthrough detected: {m._passthrough_features_}")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())

    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("cumulative").print_stats("passthrough|detect_passthrough")
    print("--- passthrough-attributed lines ---")
    print(s2.getvalue())


if __name__ == "__main__":
    main()
