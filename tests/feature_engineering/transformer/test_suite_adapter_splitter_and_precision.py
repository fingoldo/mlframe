"""ShortlistTransformerAdapter: an explicit OOF splitter, full-precision inputs, and a 1-row fit that does not raise."""

import logging

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_engineering.transformer._suite_adapter import ShortlistTransformerAdapter, _to_2d_numeric


def _record_splitter(seen):
    def compute(X_train, y_train, X_query, splitter=None, *, seed=0):
        seen.append(splitter)
        n = X_train.shape[0] if X_query is None else X_query.shape[0]
        return pl.DataFrame({"f": np.zeros(n)})

    return compute


def test_the_callers_splitter_is_used_for_the_oof_features():
    seen = []
    ts = TimeSeriesSplit(n_splits=3)
    X, y = pd.DataFrame({"a": np.arange(12.0)}), np.arange(12) % 2
    ShortlistTransformerAdapter(_record_splitter(seen), splitter=ts).fit(X, y).fit_transform(X, y)
    assert seen and seen[-1] is ts, "a time-aware splitter passed by the caller must reach the wrapped transformer"


def test_the_shuffled_fallback_is_announced(caplog):
    """The sibling contract makes the splitter required because shuffled KFold leaks on temporal rows; a fallback must say so."""
    seen = []
    X, y = pd.DataFrame({"a": np.arange(12.0)}), np.arange(12) % 2
    with caplog.at_level(logging.WARNING):
        ShortlistTransformerAdapter(_record_splitter(seen)).fit(X, y).fit_transform(X, y)
    assert any("SHUFFLED KFold" in r.getMessage() for r in caplog.records)


def test_large_magnitude_inputs_keep_their_resolution():
    """float32 kept ~7 significant digits, so epoch-microseconds lost everything below ~1e8."""
    epoch_us = np.array([1_700_000_000_000_001.0, 1_700_000_000_000_002.0])
    arr = _to_2d_numeric(epoch_us)
    assert arr.dtype == np.float64
    assert arr[1, 0] - arr[0, 0] == pytest.approx(1.0)


def test_a_one_row_fit_takes_the_out_of_sample_path():
    """KFold(2).split on one sample raised from inside fit_transform."""
    seen = []
    X, y = pd.DataFrame({"a": [1.0]}), np.array([1])
    out = ShortlistTransformerAdapter(_record_splitter(seen)).fit(X, y).fit_transform(X, y)
    assert len(out) == 1
    assert seen == [None], "the 1-row fit must go through transform(), which passes no splitter"
