"""Wave-5 FE / pipeline-extension regression tests.

Covers:
  A2-02  apply_preprocessing_extensions fastpath: polars in -> polars out, NO down-convert, when zero stages are active.
  A2-03  RBFSampler / Nystroem / dim-reducer random_state threaded from config.random_seed (reproducible).
  A2-04  _filter_to_numeric cross-split column parity via keep_cols.
  A2-13  deepcopy (not shallow copy.copy) fallback for the auto-tune config rewrite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.pipeline._pipeline_extensions import (
    _filter_to_numeric,
    _has_active_extension_stage,
    apply_preprocessing_extensions,
)
from mlframe.training.configs import PreprocessingExtensionsConfig


# ----------------------------- A2-02 ---------------------------------------


def test_a2_02_no_active_stage_returns_polars_unchanged() -> None:
    """Zero active stages: the polars frame must be returned UNCHANGED (same object, no pandas down-convert)."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    cfg = PreprocessingExtensionsConfig()  # all stages off by default
    assert _has_active_extension_stage(cfg) is False
    train, val, test, pipe = apply_preprocessing_extensions(df, None, None, cfg)
    assert pipe is None
    assert train is df, "fastpath must return the identical polars object (no conversion)"
    assert isinstance(train, pl.DataFrame)


def test_a2_02_active_stage_triggers_conversion() -> None:
    """A configured scaler stage activates the path -> pandas output."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 5.0, 6.0, 7.0]})
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler")
    assert _has_active_extension_stage(cfg) is True
    train, _, _, pipe = apply_preprocessing_extensions(df, None, None, cfg)
    assert pipe is not None
    assert not isinstance(train, pl.DataFrame)


# ----------------------------- A2-03 ---------------------------------------


def test_a2_03_nonlinear_random_state_reproducible_from_seed() -> None:
    """Two runs with the same config.random_seed produce identical RBFSampler output; a different seed differs."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 6)), columns=[f"f{i}" for i in range(6)])

    def _run(seed: int) -> np.ndarray:
        cfg = PreprocessingExtensionsConfig(nonlinear_features="RBFSampler", nonlinear_n_components=8, random_seed=seed)
        out, _, _, _ = apply_preprocessing_extensions(X.copy(), None, None, cfg)
        return out.to_numpy()

    a = _run(123)
    b = _run(123)
    c = _run(999)
    assert np.allclose(a, b), "same random_seed must give identical RBFSampler features"
    assert not np.allclose(a, c), "different random_seed must change RBFSampler features (seed is threaded, not hardcoded)"


# ----------------------------- A2-04 ---------------------------------------


def test_a2_04_filter_to_numeric_keep_cols_pins_cross_split() -> None:
    """keep_cols pins val/test to train's numeric column decision even when a column's per-split dtype differs."""
    train = pd.DataFrame({"x": [1.0, 2.0], "flag": [True, False], "txt": ["a", "b"]})
    train_f, dropped = _filter_to_numeric(train)
    keep = list(train_f.columns)
    assert "txt" in dropped and "x" in keep and "flag" in keep

    # val where 'x' is accidentally object dtype -- without keep_cols a per-split select_dtypes would drop it and diverge.
    val = pd.DataFrame({"x": ["3.0", "4.0"], "flag": [False, True], "txt": ["c", "d"]})
    val_f, _ = _filter_to_numeric(val, keep_cols=keep)
    assert list(val_f.columns) == keep, "val must keep exactly the train-decided columns"


def test_a2_04_apply_extensions_aligns_splits() -> None:
    """End-to-end: train/val/test emerge from the numeric gate with identical column sets."""
    rng = np.random.default_rng(1)
    cols = [f"f{i}" for i in range(5)]
    train = pd.DataFrame(rng.standard_normal((50, 5)), columns=cols)
    val = pd.DataFrame(rng.standard_normal((20, 5)), columns=cols)
    test = pd.DataFrame(rng.standard_normal((20, 5)), columns=cols)
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler")
    tr, va, te, _ = apply_preprocessing_extensions(train, val, test, cfg)
    assert tr.shape[1] == va.shape[1] == te.shape[1]


# ----------------------------- A2-13 ---------------------------------------


def test_a2_13_deepcopy_fallback_used_in_autotune() -> None:
    """When model_copy is unavailable, the auto-tune path deepcopies the config (not shallow copy.copy)."""
    import copy

    # A plain object config without model_copy, with a nested mutable field.
    class _PlainCfg:
        def __init__(self):
            self.dim_reducer = "PCA"
            self.dim_n_components = 50
            self.scaler = None
            self.binarization_threshold = None
            self.kbins = None
            self.polynomial_degree = None
            self.nonlinear_features = None
            self.tfidf_columns = None
            self.pysr_enabled = False
            self.verbose_logging = False
            self.memory_safety_max_bytes = None
            self.random_seed = 7
            self.nested = {"k": [1, 2, 3]}

    seen = {}
    real_deepcopy = copy.deepcopy

    def _spy_deepcopy(obj):
        seen["called"] = True
        return real_deepcopy(obj)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(copy, "deepcopy", _spy_deepcopy)
    try:
        cfg = _PlainCfg()
        X = pd.DataFrame(np.random.default_rng(0).standard_normal((30, 4)), columns=list("abcd"))
        # dim_n_components=50 >> n_features=4 -> clamp path mutates a copy of the config.
        apply_preprocessing_extensions(X, None, None, cfg)
    finally:
        monkey.undo()
    assert seen.get("called"), "auto-tune clamp must deepcopy the plain-object config, not shallow-copy it"
    assert cfg.dim_n_components == 50, "caller's config must NOT be mutated by the clamp"
