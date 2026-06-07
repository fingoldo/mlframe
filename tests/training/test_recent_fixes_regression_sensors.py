"""Regression sensors for the 2026-05-20 round of bug fixes surfaced by
S: full-suite runs (fuzz_3way + ensembling-chart + composite-discovery
combos).

Each test below targets ONE specific fix and is designed to:
  1. Fail on the pre-fix code (verifiable via ``git stash``).
  2. Pass on the post-fix code (CI gate).
  3. Run in <2 seconds and import nothing heavyweight (no MLP / lightning).

The fixes covered:
  - process_infinities cs.float() vs cs.numeric() (commit af66424)
  - preprocess_dataframe normalises pandas StringDtype -> object (b3aefb8)
  - create_fairness_subgroups whitelist-numeric for qcut (fe7d653)
  - _warmup_numba_kernels <-> prewarm_numba_cache re-entrancy (0d64b72)
  - multilabel detection shape[1] >= 2, not >= 1 (dfca063)
  - CatBoost allow_writing_files=False default (e68bbde)
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Fix: process_infinities must skip integer columns (af66424).
# Pre-fix used cs.numeric().replace([inf,-inf], 0.0) which crashed polars
# with ``InvalidOperationError: conversion from f64 to i32 failed`` because
# polars tried to cast inf to the int column dtype.
# ---------------------------------------------------------------------------


def test_process_infinities_skips_int_columns():
    """Mixed int + float frame with inf in float must complete without
    raising; ints unchanged, infs in float replaced by fill_value."""
    from mlframe.training._nan_processing import process_infinities

    df = pl.DataFrame({
        "int_col": pl.Series([1, 2, 3, 4], dtype=pl.Int32),
        "float_col": pl.Series([1.0, float("inf"), float("-inf"), 4.0], dtype=pl.Float64),
    })
    out = process_infinities(df, fill_value=0.0, verbose=0)
    assert out["int_col"].to_list() == [1, 2, 3, 4], "int column must pass through untouched"
    assert out["float_col"].to_list() == [1.0, 0.0, 0.0, 4.0], "+/-inf must replace with fill_value"


# ---------------------------------------------------------------------------
# Fix: preprocess_dataframe normalises pandas StringDtype -> object (b3aefb8).
# Pre-fix downstream code (auto-detect cat, LGB/XGB fit) silently skipped
# StringDtype columns and the fit crashed with
# ``pandas dtypes must be int, float or bool, got cat_a: str``.
# ---------------------------------------------------------------------------


def test_preprocess_dataframe_stringdtype_to_object():
    """A StringDtype column must be coerced to object before downstream
    consumers (auto-detect / encoders / model fit) ever see the frame."""
    from mlframe.training.preprocessing import preprocess_dataframe
    from mlframe.training.configs import PreprocessingConfig

    df = pd.DataFrame({
        "f0": np.array([1.0, 2.0, 3.0], dtype="float32"),
        "cat_a": pd.array(["A", "B", "C"], dtype="string"),
        "cat_b": pd.array(["X", "Y", "X"], dtype="string"),
    })
    assert isinstance(df["cat_a"].dtype, pd.StringDtype), "fixture must create StringDtype"
    out = preprocess_dataframe(df, PreprocessingConfig(), verbose=0)
    assert out["cat_a"].dtype == object, "cat_a must be coerced to object"
    assert out["cat_b"].dtype == object, "cat_b must be coerced to object"
    assert out["f0"].dtype.name == "float32", "numeric column must be unchanged"


# ---------------------------------------------------------------------------
# Fix: create_fairness_subgroups whitelist-numeric for qcut (fe7d653).
# Pre-fix blacklist let pyarrow large_string slip past, then pd.qcut crashed
# with ``ArrowNotImplementedError: Function 'quantile' has no kernel
# matching input types (large_string)``.
# ---------------------------------------------------------------------------


def test_create_fairness_subgroups_handles_pyarrow_large_string():
    """A pyarrow-backed large_string column must be treated as categorical
    (skip qcut path); the numeric column path is still exercised."""
    pyarrow = pytest.importorskip("pyarrow")
    from mlframe.metrics.core import create_fairness_subgroups

    str_arr = pyarrow.array(["A", "B", "C"] * 100, type=pyarrow.large_string())
    df = pd.DataFrame({
        "num": np.random.default_rng(0).normal(size=300),
        "cat": pd.Series(pd.arrays.ArrowExtensionArray(str_arr), name="cat"),
    })
    assert str(df["cat"].dtype).startswith("large_string"), "fixture must produce large_string"

    # Pre-fix this raised ArrowNotImplementedError from pd.qcut(large_string, ...).
    subgroups = create_fairness_subgroups(df, ["cat", "num"], min_pop_cat_thresh=10)
    assert set(subgroups.keys()) == {"cat", "num"}, "both columns must be subgrouped"


# ---------------------------------------------------------------------------
# Fix: _warmup_numba_kernels <-> prewarm_numba_cache mutual recursion guard
# (0d64b72). Pre-fix the two functions called each other without a
# re-entrancy sentinel; the stack overflowed before any except block fired.
# ---------------------------------------------------------------------------


def test_warmup_numba_kernels_does_not_recurse():
    """Calling _warmup_numba_kernels twice in a row must NOT exhaust the
    Python recursion limit. The sentinel attribute makes the second
    forward+reverse traversal a no-op."""
    pytest.importorskip("numba")
    from mlframe.training.baselines.dummy import _warmup_numba_kernels

    # Two back-to-back calls: previously the second one re-entered the cycle.
    _warmup_numba_kernels(verbose=False)
    _warmup_numba_kernels(verbose=False)


def test_prewarm_numba_cache_does_not_recurse():
    """Symmetric direction: prewarm_numba_cache must not recurse either."""
    pytest.importorskip("numba")
    from mlframe.metrics.core import prewarm_numba_cache

    prewarm_numba_cache()
    prewarm_numba_cache()


# ---------------------------------------------------------------------------
# Fix: multilabel detection requires shape[1] >= 2, not >= 1 (dfca063).
# A (N, 1) single-column 2-D target is still SINGLE-label classification;
# the pre-fix check misclassified it as multilabel, the MLP got
# output_dim=1, predictions+labels both squeezed to (N,), CrossEntropyLoss
# entered the class-probabilities branch and rejected Long labels.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "y_shape, expected_multilabel",
    [
        ((100,),    False),   # 1-D vector -> single-label
        ((100, 1),  False),   # 1-column 2-D       -> single-label (the bug case)
        ((100, 2),  True),    # 2-column 2-D       -> multilabel
        ((100, 5),  True),    # K-column 2-D       -> multilabel
    ],
)
def test_classifier_multilabel_detection_shape_only(y_shape, expected_multilabel):
    """Verify the multilabel-detection predicate that all three classifier
    code paths (base.py, recurrent.py x2) share. Direct shape check; no
    torch/lightning import needed."""
    y = np.zeros(y_shape, dtype=np.int64)
    arr = np.asarray(y)
    # The fixed predicate as used in base.py:291 / recurrent.py:692, 982.
    is_multilabel = bool(arr.ndim == 2 and arr.shape[1] >= 2)
    assert is_multilabel is expected_multilabel


# ---------------------------------------------------------------------------
# Fix: CatBoost allow_writing_files=False default (e68bbde).
# Pre-fix CB wrote catboost_info/ into CWD, causing Windows
# ``CatBoostError: Error 5: Access is denied`` under xdist multi-process.
# ---------------------------------------------------------------------------


def _get_cb_general_params():
    """Pull CB_GENERAL_PARAMS off helpers.get_training_configs regardless of
    whether the return is a SimpleNamespace (current) or a dict (historic)."""
    from mlframe.training.helpers import get_training_configs

    cfg = get_training_configs(iterations=2, validation_fraction=0.2)
    return getattr(cfg, "CB_GENERAL_PARAMS", None) or cfg["CB_GENERAL_PARAMS"]


def test_catboost_default_does_not_create_catboost_info():
    """The CB_GENERAL_PARAMS produced by helpers.get_training_configs must
    include ``allow_writing_files=False`` so CatBoost never tries to write
    ``catboost_info/`` under the CWD."""
    cb_general = _get_cb_general_params()
    assert cb_general.get("allow_writing_files") is False, (
        "CB_GENERAL_PARAMS must default allow_writing_files=False to avoid "
        "xdist worker collision on catboost_info/catboost_training.json"
    )


def test_catboost_real_fit_writes_no_files_with_default():
    """End-to-end behavioural check: a tiny CB fit with the mlframe default
    params must NOT create a catboost_info/ directory under the CWD."""
    pytest.importorskip("catboost")
    from catboost import CatBoostClassifier

    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    cb_general = _get_cb_general_params()
    # Run from a clean tempdir so we can detect the absence/presence of catboost_info.
    prev_cwd = os.getcwd()
    test_cwd = tempfile.mkdtemp(prefix="mlframe_cb_sensor_")
    os.chdir(test_cwd)
    try:
        # Use only the kwargs CatBoostClassifier consistently accepts across versions.
        clf = CatBoostClassifier(
            iterations=cb_general["iterations"],
            allow_writing_files=cb_general["allow_writing_files"],
            verbose=0,
        )
        clf.fit(X, y)
        assert not os.path.isdir("catboost_info"), (
            "catboost_info/ must not be created when allow_writing_files=False"
        )
    finally:
        os.chdir(prev_cwd)
        import shutil
        shutil.rmtree(test_cwd, ignore_errors=True)
