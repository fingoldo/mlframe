"""Regression tests for compute_model_input_fingerprint extension (P1).

Pre-fix the fingerprint was schema-only: target name / row count /
preprocessing config / pipeline config / model family / random_seed /
split indices were not folded in, so changing any of them silently
collided on the same model filename. Post-fix every changed input
yields a distinct hash and identical inputs yield an identical hash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.utils import compute_model_input_fingerprint


def _df() -> pd.DataFrame:
    return pd.DataFrame({
        "a": np.arange(10, dtype=np.float64),
        "b": np.arange(10, dtype=np.float32),
    })


def test_fingerprint_identical_inputs_match():
    h1, _ = compute_model_input_fingerprint(
        _df(), target_name="y",
        preprocessing_config={"scaler": "standard"},
        pipeline_config={"steps": ["a", "b"]},
        model_family="lgbm",
        random_seed=42,
        train_idx=np.arange(7),
        val_idx=np.arange(7, 10),
    )
    h2, _ = compute_model_input_fingerprint(
        _df(), target_name="y",
        preprocessing_config={"scaler": "standard"},
        pipeline_config={"steps": ["a", "b"]},
        model_family="lgbm",
        random_seed=42,
        train_idx=np.arange(7),
        val_idx=np.arange(7, 10),
    )
    assert h1 == h2


def test_fingerprint_row_count_change_invalidates():
    """Schema unchanged but row count differs -> different hash."""
    df_short = pd.DataFrame({"a": np.arange(10, dtype=np.float64)})
    df_long = pd.DataFrame({"a": np.arange(100, dtype=np.float64)})
    h1, _ = compute_model_input_fingerprint(df_short, target_name="y")
    h2, _ = compute_model_input_fingerprint(df_long, target_name="y")
    assert h1 != h2


def test_fingerprint_target_name_change_invalidates():
    df = _df()
    h1, _ = compute_model_input_fingerprint(df, target_name="y_a")
    h2, _ = compute_model_input_fingerprint(df, target_name="y_b")
    assert h1 != h2


def test_fingerprint_preprocessing_config_change_invalidates():
    df = _df()
    h1, _ = compute_model_input_fingerprint(
        df, preprocessing_config={"scaler": "standard"},
    )
    h2, _ = compute_model_input_fingerprint(
        df, preprocessing_config={"scaler": "minmax"},
    )
    assert h1 != h2


def test_fingerprint_pipeline_config_change_invalidates():
    df = _df()
    h1, _ = compute_model_input_fingerprint(df, pipeline_config={"k": 1})
    h2, _ = compute_model_input_fingerprint(df, pipeline_config={"k": 2})
    assert h1 != h2


def test_fingerprint_random_seed_change_invalidates():
    df = _df()
    h1, _ = compute_model_input_fingerprint(df, random_seed=1)
    h2, _ = compute_model_input_fingerprint(df, random_seed=2)
    assert h1 != h2


def test_fingerprint_split_indices_change_invalidates():
    df = _df()
    h1, _ = compute_model_input_fingerprint(df, train_idx=np.arange(7), val_idx=np.arange(7, 10))
    h2, _ = compute_model_input_fingerprint(df, train_idx=np.arange(8), val_idx=np.arange(8, 10))
    assert h1 != h2
