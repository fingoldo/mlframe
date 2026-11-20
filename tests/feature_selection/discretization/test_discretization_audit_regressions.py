"""Regression tests for the discretization audit: null categorical codes, all-NA ints, chunked-uniform NaN chunks, silent CUDA fallback, code dtype widening."""

from __future__ import annotations

import logging
import types

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters.discretization as disc_mod
from mlframe.feature_selection.filters.discretization import categorize_dataset
from mlframe.feature_selection.filters.discretization._discretization_cuda import discretize_2d_array_cuda_row_chunked
from mlframe.feature_selection.filters.discretization._discretization_edges import discretize_sklearn
from mlframe.utils.log_throttle import reset_throttle_counts


def test_categorize_dataset_polars_null_string_gets_own_code():
    """A polars string column holding a null must give the null its own code instead of crashing on a NaN-to-int cast."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "s": ["x", None, "y", "x"]})
    data, cols, nbins = categorize_dataset(df, missing_strategy="separate_bin")
    assert cols == ["a", "s"]
    s_codes = data[:, 1]
    assert s_codes[1] != s_codes[0] and s_codes[1] != s_codes[2]
    assert s_codes[0] == s_codes[3]
    assert int(nbins[1]) == 3


def test_categorize_dataset_all_na_nullable_int_column_does_not_crash():
    """An all-NA nullable Int64 column makes ``Series.min()`` return ``pd.NA``; the integer-range check must skip it."""
    df = pd.DataFrame({"a": pd.array([None, None, None, None], dtype="Int64"), "b": [1.0, 2.0, 3.0, 4.0]})
    data, cols, _nbins = categorize_dataset(df, missing_strategy="separate_bin")
    assert cols == ["a", "b"]
    assert data.shape == (4, 2)


def test_row_chunked_uniform_all_nan_first_chunk_keeps_column_range(monkeypatch):
    """A column whose first row-chunk is all NaN must still be binned by its true range, not collapsed by a NaN-propagating running min."""
    shim = types.SimpleNamespace(
        int8=np.int8, float64=np.float64, asarray=np.asarray, asnumpy=np.asarray, nanmin=np.nanmin, nanmax=np.nanmax,
        fmin=np.fmin, fmax=np.fmax, minimum=np.minimum, maximum=np.maximum, where=np.where, nan_to_num=np.nan_to_num,
        clip=np.clip, isnan=np.isnan,
    )
    monkeypatch.setitem(__import__("sys").modules, "cupy", shim)
    n = 25000
    col = np.linspace(0.0, 1.0, n)
    col[:10000] = np.nan
    arr = np.column_stack([col, np.linspace(0.0, 1.0, n)])
    with np.errstate(all="ignore"):
        out = discretize_2d_array_cuda_row_chunked(arr, n_bins=10, method="uniform", dtype=np.int8, quantile_subsample_rows=100, free_bytes=0)
    assert (out[:10000, 0] == 10).all()
    assert out[10000:, 0].min() == 0 and out[10000:, 0].max() == 9
    assert out[:, 1].max() == 9


def test_cuda_fastpath_failure_is_warned_not_debug(monkeypatch, caplog):
    """A CUDA fast-path exception that sends the frame to the slower CPU path must be logged at WARNING."""
    reset_throttle_counts()
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(size=(2000, 300)).astype(np.float32)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    monkeypatch.setattr(disc_mod._DISCRETIZE_SPEC, "choose", lambda **kw: "cuda")
    monkeypatch.setattr("pyutilz.core.pythonlib.is_cuda_available", lambda: True)
    monkeypatch.setattr("mlframe.feature_selection.filters._fe_gpu_vram.fe_gpu_has_vram_cushion", lambda bytes_needed: True)

    def _boom(*a, **kw):
        """Raise like a failing device kernel."""
        raise RuntimeError("simulated device fault")

    monkeypatch.setattr(disc_mod, "discretize_2d_array_cuda", _boom)
    with caplog.at_level(logging.WARNING):
        out = disc_mod.discretize_2d_array(arr, n_bins=10, method="quantile", dtype=np.int8)
    assert out.shape == arr.shape
    assert any("CUDA fastpath failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_discretize_sklearn_widens_int8_codes_for_many_bins():
    """With more than 128 bins the int8 request must be widened, not wrapped negative."""
    arr = np.linspace(0.0, 1.0, 5000)
    out = discretize_sklearn(arr, n_bins=200, method="uniform", dtype=np.int8)
    assert out.min() >= 0
    assert out.max() > 127


def test_count_freq_residual_recipe_builders_canonicalize_float_keys():
    """Recipe builders must key the lookup like the apply side does, so a float-keyed raw lookup still matches integer data."""
    from mlframe.feature_selection.filters.engineered_recipes import (
        apply_recipe,
        build_cat_num_residual_recipe,
        build_count_encoded_recipe,
        build_frequency_encoded_recipe,
    )

    X = pd.DataFrame({"c": [1, 2, 3], "v": [10.0, 20.0, 30.0]})
    cnt = apply_recipe(build_count_encoded_recipe(name="n", src_name="c", lookup={1.0: 3, 2.0: 5}), X)
    assert list(np.asarray(cnt)) == [3, 5, 0]
    frq = apply_recipe(build_frequency_encoded_recipe(name="f", src_name="c", lookup={1.0: 0.25, 2.0: 0.5}), X)
    assert list(np.asarray(frq)) == [0.25, 0.5, 0.0]
    res = apply_recipe(build_cat_num_residual_recipe(name="r", cat_name="c", num_name="v", lookup={1.0: 4.0, 2.0: 5.0}, global_mean=1.0, smoothing=1.0), X)
    assert list(np.asarray(res)) == [6.0, 15.0, 29.0]
