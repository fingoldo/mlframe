"""Audit 2026-11-20 finding 28-F6: with fix_infinities=False a polars frame holding +inf is reported by the guard, not passed on silently."""

import logging

import numpy as np
import polars as pl

from mlframe.training.configs import PreprocessingConfig
from mlframe.training.preprocessing import preprocess_dataframe


def test_polars_inf_with_fix_infinities_false_is_reported(caplog):
    """The guard logs an error naming the inf problem, so the batched path cannot hide it."""
    df = pl.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})
    config = PreprocessingConfig(fillna_value=None, fix_infinities=False, ensure_float32_dtypes=False, remove_constant_columns=True)
    with caplog.at_level(logging.ERROR):
        preprocess_dataframe(df, config, verbose=0)
    assert any("np.inf" in rec.getMessage() for rec in caplog.records)
