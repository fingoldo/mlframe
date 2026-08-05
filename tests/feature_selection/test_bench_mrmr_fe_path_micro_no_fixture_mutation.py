"""FS_BENCHMARKS_B-1: bench_e8_tonumeric_vs_asarray's fast path must not mutate the shared df
fixture in place, else repeated timed calls skip the NaN scrub and inflate the reported speedup."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlframe.feature_selection._benchmarks.bench_mrmr_fe_path_micro import _tonumeric_fast_path, bench_e8_tonumeric_vs_asarray


def test_tonumeric_fast_path_does_not_mutate_source_df():
    """Calling the real production fast path repeatedly must scrub NaNs every time, proving it
    never mutates its df argument through a numpy view."""
    rng = np.random.default_rng(7)
    n, p = 500, 8
    M = rng.standard_normal((n, p))
    mask = rng.random((n, p)) < 0.05
    M[mask] = np.nan
    df = pd.DataFrame(M, columns=[f"c{i}" for i in range(p)])
    df_before = df.copy(deep=True)

    first = _tonumeric_fast_path(df)
    assert not np.isnan(first).any()
    pd.testing.assert_frame_equal(df, df_before)

    second = _tonumeric_fast_path(df)
    assert not np.isnan(second).any(), "second call found no NaNs to scrub, meaning the first call mutated the shared df fixture in place"


def test_bench_e8_tonumeric_vs_asarray_runs_and_matches_old_new():
    """Smoke-check the actual benchmark entrypoint still returns a sane, internally consistent result."""
    result = bench_e8_tonumeric_vs_asarray(n=200, p=6, nan_frac=0.05)
    assert result["lever"] == "E8_tonumeric_vs_asarray"
    assert result["old_ms"] > 0
    assert result["new_ms"] > 0
