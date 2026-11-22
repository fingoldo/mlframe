"""Cluster 40 regression: float32 fallback keeps integer column labels (audit 2026-11-20, finding 40-F27)."""

import numpy as np
import pandas as pd

from mlframe.training.neural.data import TorchDataModule


def test_per_column_fallback_accepts_integer_column_labels():
    """A numpy-origin frame (integer labels) with one non-numeric column converts without TypeError (40-F27)."""
    df = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: ["x", "y", "z"], 2: [4.0, 5.0, 6.0]})
    dm = TorchDataModule(train_features=df)
    dm.train_features = df
    dm._convert_features_dtype(["train_features"])
    out = dm.train_features
    assert list(out.columns) == [0, 1, 2]
    assert out[0].dtype == np.float32 and out[2].dtype == np.float32
    assert out[1].tolist() == ["x", "y", "z"]
