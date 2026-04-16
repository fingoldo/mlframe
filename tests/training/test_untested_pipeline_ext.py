"""Tests for mlframe.training.pipeline.apply_preprocessing_extensions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.pipeline import apply_preprocessing_extensions
from mlframe.training.configs import PreprocessingExtensionsConfig


def _make_data(n=60, p=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])


def test_extensions_none_returns_inputs_unchanged():
    tr = _make_data()
    va = _make_data(n=10)
    te = _make_data(n=10)
    out_tr, out_va, out_te, pipe = apply_preprocessing_extensions(tr, va, te, None, verbose=0)
    assert out_tr is tr and out_va is va and out_te is te
    assert pipe is None


def test_extensions_empty_config_returns_inputs():
    tr = _make_data()
    va = _make_data(n=5)
    te = _make_data(n=5)
    # No fields set -> no steps built -> untouched
    cfg = PreprocessingExtensionsConfig()
    out_tr, out_va, out_te, pipe = apply_preprocessing_extensions(tr, va, te, cfg, verbose=0)
    assert out_tr is tr and out_va is va and out_te is te
    assert pipe is None


def test_extensions_scaler_applies_sklearn():
    tr = _make_data(n=50)
    va = _make_data(n=10, seed=1)
    te = _make_data(n=10, seed=2)
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler")
    out_tr, out_va, out_te, pipe = apply_preprocessing_extensions(tr, va, te, cfg, verbose=0)
    assert pipe is not None
    assert isinstance(out_tr, pd.DataFrame)
    # StandardScaler -> train mean ~ 0
    assert abs(out_tr.mean().mean()) < 1e-6
    # Columns renamed to ext_i
    assert all(c.startswith("ext_") for c in out_tr.columns)
    # val/test shapes preserved
    assert out_va.shape[0] == 10
    assert out_te.shape[0] == 10


def test_extensions_converts_polars_input():
    tr_pd = _make_data(n=40)
    tr = pl.from_pandas(tr_pd)
    va = pl.from_pandas(_make_data(n=5, seed=1))
    cfg = PreprocessingExtensionsConfig(scaler="MinMaxScaler")
    out_tr, out_va, out_te, pipe = apply_preprocessing_extensions(tr, va, None, cfg, verbose=0)
    assert isinstance(out_tr, pd.DataFrame)
    # MinMax on train -> [0,1] bounds
    assert out_tr.min().min() >= -1e-9
    assert out_tr.max().max() <= 1 + 1e-9
    assert out_te is None


def test_extensions_stacked_scaler_plus_kbins():
    tr = _make_data(n=60)
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", kbins=4)
    out_tr, _, _, pipe = apply_preprocessing_extensions(tr, None, None, cfg, verbose=0)
    # sklearn Pipeline with 2 steps
    assert len(pipe.steps) == 2
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert "kbins" in step_names
