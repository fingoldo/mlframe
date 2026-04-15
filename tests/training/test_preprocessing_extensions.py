"""Tests for `apply_preprocessing_extensions` + `PreprocessingExtensionsConfig`.

These tests use pandas inputs directly (bypassing the suite) so they don't
transitively import torch/pytorch_lightning during collection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline import apply_preprocessing_extensions


@pytest.fixture
def small_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((60, 5)), columns=[f"x{i}" for i in range(5)])


def test_none_config_is_noop(small_df):
    val = small_df.iloc[:10].copy()
    a, b, c, p = apply_preprocessing_extensions(small_df, val, None, None)
    assert p is None
    assert a is small_df
    assert b is val
    assert c is None


def test_empty_config_is_noop(small_df):
    cfg = PreprocessingExtensionsConfig()  # all defaults = None
    a, _, _, p = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert p is None
    assert a is small_df


@pytest.mark.parametrize("scaler", [
    "StandardScaler", "RobustScaler", "MinMaxScaler", "MaxAbsScaler",
    "QuantileTransformer_uniform", "Normalizer_l2",
])
def test_scaler_variants_produce_expected_shape(small_df, scaler):
    cfg = PreprocessingExtensionsConfig(scaler=scaler)
    out, _, _, pipe = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert out.shape == small_df.shape
    assert pipe is not None


def test_pca_reduces_dimension(small_df):
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", dim_reducer="PCA", dim_n_components=3)
    out, _, _, _ = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert out.shape == (60, 3)


def test_polynomial_features_guard_triggers(small_df):
    cfg = PreprocessingExtensionsConfig(polynomial_degree=3, memory_safety_max_features=50)
    with pytest.raises(ValueError, match="memory_safety_max_features"):
        apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)


def test_binarization_and_kbins_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        PreprocessingExtensionsConfig(binarization_threshold=0.5, kbins=5)


def test_kbins_min_bins():
    with pytest.raises(ValueError, match="kbins"):
        PreprocessingExtensionsConfig(kbins=1)


def test_polynomial_min_degree():
    with pytest.raises(ValueError, match="polynomial_degree"):
        PreprocessingExtensionsConfig(polynomial_degree=1)


def test_umap_missing_raises_importerror(monkeypatch, small_df):
    import importlib.util as ilu
    orig = ilu.find_spec
    monkeypatch.setattr(ilu, "find_spec", lambda name: None if name == "umap" else orig(name))
    cfg = PreprocessingExtensionsConfig(dim_reducer="UMAP", dim_n_components=2)
    with pytest.raises(ImportError, match="umap-learn"):
        apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)


def test_binarizer_produces_binary(small_df):
    cfg = PreprocessingExtensionsConfig(binarization_threshold=0.0)
    out, _, _, _ = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert set(np.unique(out.values)) <= {0.0, 1.0}


def test_kbins_produces_integers(small_df):
    cfg = PreprocessingExtensionsConfig(kbins=4)
    out, _, _, _ = apply_preprocessing_extensions(small_df, None, None, cfg, verbose=0)
    assert out.values.max() < 4
    assert out.values.min() >= 0


def test_val_and_test_follow_train(small_df):
    val = small_df.iloc[:20].copy()
    test = small_df.iloc[20:30].copy()
    cfg = PreprocessingExtensionsConfig(scaler="StandardScaler", dim_reducer="PCA", dim_n_components=2)
    tr, va, te, pipe = apply_preprocessing_extensions(small_df, val, test, cfg, verbose=0)
    assert tr.shape == (60, 2)
    assert va.shape == (20, 2)
    assert te.shape == (10, 2)
