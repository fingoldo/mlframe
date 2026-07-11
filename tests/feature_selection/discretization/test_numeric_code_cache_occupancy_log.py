"""Regression: ``categorize_dataset`` emits a DEBUG-level occupancy snapshot of the process-wide
numeric-code cache (``_NUMERIC_CODE_CACHE``) on every call.

The cache (``_discretization_dataset.py``, ``_NUMERIC_CODE_CACHE`` + ``_NUMERIC_CODE_CACHE_MAX_BYTES``)
was already properly bounded (512 MB default, LRU-evicted via ``popitem``) -- confirmed by reading the
implementation (``_discretize_2d_array_col_cached``'s while-loop evicts until back under the byte cap).
This module adds ONLY the missing occupancy visibility: one DEBUG log line per ``categorize_dataset``
call reporting entry count + bytes vs the cap, so a maintainer diagnosing memory pressure can see cache
growth/eviction without reading the module globals directly.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.discretization._discretization_dataset import (
    categorize_dataset,
    clear_numeric_code_cache,
)


def test_categorize_dataset_logs_cache_occupancy(caplog):
    clear_numeric_code_cache()
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 4)), columns=["a", "b", "c", "d"])

    logger_name = "mlframe.feature_selection.filters.discretization._discretization_dataset"
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        categorize_dataset(X, n_bins=4)

    occupancy_lines = [r for r in caplog.records if "numeric-code cache occupancy" in r.message]
    assert occupancy_lines, "expected a DEBUG log line reporting numeric-code cache occupancy"
    msg = occupancy_lines[-1].message
    assert "entries" in msg and "bytes" in msg and "cap" in msg


def test_categorize_dataset_occupancy_log_reflects_growth(caplog):
    """After caching several columns, the reported entry count must be > 0."""
    clear_numeric_code_cache()
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((300, 6)), columns=list("abcdef"))

    logger_name = "mlframe.feature_selection.filters.discretization._discretization_dataset"
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        categorize_dataset(X, n_bins=4)

    occupancy_lines = [r for r in caplog.records if "numeric-code cache occupancy" in r.message]
    assert occupancy_lines
    # The log call's positional args carry (n_entries, n_bytes, cap_bytes).
    last = occupancy_lines[-1]
    n_entries = last.args[0]
    assert n_entries >= 1, f"expected at least 1 cached column entry after categorize_dataset; got {n_entries}"
    clear_numeric_code_cache()
