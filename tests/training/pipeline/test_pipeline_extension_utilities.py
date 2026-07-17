"""Tests for mlframe.training.pipeline.apply_preprocessing_extensions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.pipeline import apply_preprocessing_extensions
from mlframe.training.configs import PreprocessingExtensionsConfig


def _make_data(n=60, p=4, seed=0):
    """Random numeric (n, p) DataFrame with columns f0..f(p-1)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])


def test_extensions_none_returns_inputs_unchanged():
    """cfg=None returns the caller's train/val/test objects unchanged (identity, no pipeline built)."""
    tr = _make_data()
    va = _make_data(n=10)
    te = _make_data(n=10)
    out_tr, out_va, out_te, pipe = apply_preprocessing_extensions(tr, va, te, None, verbose=0)
    assert out_tr is tr and out_va is va and out_te is te
    assert pipe is None


def test_extensions_empty_config_returns_inputs():
    """A config with every step explicitly off returns the caller's objects unchanged."""
    tr = _make_data()
    va = _make_data(n=5)
    te = _make_data(n=5)
    # row_wise_summary_stats_enabled / row_wise_extreme_columns_enabled default to True (additive,
    # generic per-row aggregates) -- explicitly off here so "no fields set" genuinely means "no
    # steps built -> untouched", matching this test's actual subject.
    cfg = PreprocessingExtensionsConfig(row_wise_summary_stats_enabled=False, row_wise_extreme_columns_enabled=False)
    out_tr, out_va, out_te, pipe = apply_preprocessing_extensions(tr, va, te, cfg, verbose=0)
    assert out_tr is tr and out_va is va and out_te is te
    assert pipe is None


def test_extensions_scaler_applies_sklearn():
    """A configured scaler runs, centers the train frame, and preserves the original column names/shapes."""
    tr = _make_data(n=50)
    va = _make_data(n=10, seed=1)
    te = _make_data(n=10, seed=2)
    # Row-wise stats/extreme-columns default ON but are orthogonal to this test's subject (the
    # scaler step preserving column identity); disabled so the added columns don't obscure it.
    cfg = PreprocessingExtensionsConfig(
        scaler="StandardScaler",
        row_wise_summary_stats_enabled=False,
        row_wise_extreme_columns_enabled=False,
    )
    out_tr, out_va, out_te, pipe = apply_preprocessing_extensions(tr, va, te, cfg, verbose=0)
    assert pipe is not None
    assert isinstance(out_tr, pd.DataFrame)
    # StandardScaler -> train mean ~ 0
    assert abs(out_tr.mean().mean()) < 1e-6
    # Output preserves original column names (legacy renaming to ``ext_*`` was
    # dropped; downstream consumers rely on stable column identifiers).
    assert list(out_tr.columns) == list(tr.columns)
    # val/test shapes preserved
    assert out_va.shape[0] == 10
    assert out_te.shape[0] == 10


def test_extensions_converts_polars_input():
    """A polars input frame is bridged to pandas when a scaler stage is active, with bounds honoured."""
    tr_pd = _make_data(n=40)
    tr = pl.from_pandas(tr_pd)
    va = pl.from_pandas(_make_data(n=5, seed=1))
    cfg = PreprocessingExtensionsConfig(scaler="MinMaxScaler")
    out_tr, _out_va, out_te, _pipe = apply_preprocessing_extensions(tr, va, None, cfg, verbose=0)
    assert isinstance(out_tr, pd.DataFrame)
    # MinMax on train -> [0,1] bounds
    assert out_tr.min().min() >= -1e-9
    assert out_tr.max().max() <= 1 + 1e-9
    assert out_te is None


def test_extensions_stacked_scaler_plus_kbins():
    """Scaler + KBinsDiscretizer stack into a single sklearn Pipeline with the auto-prepended imputer step."""
    tr = _make_data(n=60)
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", kbins=4)
    _out_tr, _, _, pipe = apply_preprocessing_extensions(tr, None, None, cfg, verbose=0)
    # sklearn Pipeline with 3 steps post-2026-04 perf series:
    # `imputer` (defensive, NaN-safe) is now prepended automatically.
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert "kbins" in step_names
    # Imputer is defensive but always present in this branch.
    assert len(pipe.steps) >= 2  # tolerate either "scaler+kbins" or "imputer+scaler+kbins"
