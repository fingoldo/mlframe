"""Cluster 32 regression: a null group key gets its group's mean on the polars path, as on the pandas path (audit 2026-11-20, finding 32-F8)."""

import numpy as np
import polars as pl
import pytest

from mlframe.training.baselines import _dummy_baseline_compute as dbc


def test_null_key_rows_use_null_group_mean_on_polars_path():
    """Train has a null group; the val null-key row must receive that group's mean, not the global mean (32-F8)."""
    train_X = pl.DataFrame({"g": ["a", "a", None, None]})
    val_X = pl.DataFrame({"g": ["a", None]})
    test_X = pl.DataFrame({"g": ["a", None]})
    y = np.array([1.0, 1.0, 0.0, 0.0])
    train_pred, val_pred, _test_pred, _ = dbc._per_group_predict_polars(train_X, val_X, test_X, y, "g")
    assert val_pred[1] == pytest.approx(train_pred[2])
    assert val_pred[1] != pytest.approx(y.mean())
