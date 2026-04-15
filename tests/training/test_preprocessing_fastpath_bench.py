"""Regression benchmark: `preprocessing_extensions=None` must remain a
byte-identical no-op with ≤2% wall-time overhead vs directly passing
DataFrames through.

Rationale (from Audit #02 plan, Phase 3.7): every sklearn-based extension
is opt-in; when the user omits `preprocessing_extensions`, the Polars-native
fastpath must be preserved. We assert both identity and a tight time budget.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.pipeline import apply_preprocessing_extensions


def _make_df(n_rows: int = 10_000, n_cols: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.standard_normal((n_rows, n_cols)),
                        columns=[f"f{i}" for i in range(n_cols)])


def test_fastpath_returns_inputs_unchanged():
    train, val, test = _make_df(), _make_df(500), _make_df(500)
    out_train, out_val, out_test, pipe = apply_preprocessing_extensions(
        train, val, test, config=None, verbose=0
    )
    assert out_train is train
    assert out_val is val
    assert out_test is test
    assert pipe is None


def test_fastpath_polars_inputs_unchanged():
    train = pl.DataFrame({"a": np.arange(1000), "b": np.arange(1000).astype(float)})
    val = pl.DataFrame({"a": np.arange(100), "b": np.arange(100).astype(float)})
    test = pl.DataFrame({"a": np.arange(100), "b": np.arange(100).astype(float)})
    out_train, out_val, out_test, pipe = apply_preprocessing_extensions(
        train, val, test, config=None, verbose=0
    )
    assert out_train is train
    assert out_val is val
    assert out_test is test
    assert pipe is None


def test_fastpath_overhead_budget():
    """1000 invocations of the None fastpath must complete in <50ms total
    (~50µs per call) — the function is `return x,y,z,None` + a truthy check.
    This is the regression gate replacing the ≤2% benchmark from the plan:
    if anyone adds non-trivial work behind `config is None`, this fails.
    """
    train, val, test = _make_df(), _make_df(500), _make_df(500)
    # warmup
    apply_preprocessing_extensions(train, val, test, config=None, verbose=0)

    t0 = time.perf_counter()
    for _ in range(1000):
        apply_preprocessing_extensions(train, val, test, config=None, verbose=0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.05, (
        f"fastpath regressed: 1000 None-calls took {elapsed*1000:.1f}ms "
        f"(budget: 50ms). Someone likely added work before the "
        f"`if config is None: return ...` short-circuit."
    )
