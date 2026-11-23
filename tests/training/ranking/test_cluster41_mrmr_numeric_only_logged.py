"""Cluster 41 regression: group-aware MRMR logs the categorical columns it skips (audit 2026-11-20, finding 41-F25)."""

import logging

import numpy as np
import pandas as pd

from mlframe.training.ranking import _ranker_fs


def test_skipped_categorical_columns_are_logged(caplog):
    """A categorical column is excluded from the numeric-only path and the exclusion is logged (41-F25)."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({"num": rng.normal(size=n), "num2": rng.normal(size=n), "cat": pd.Categorical(rng.choice(["a", "b"], n))})
    y = rng.integers(0, 2, n).astype(float)
    groups = np.repeat(np.arange(6), 10)
    with caplog.at_level(logging.WARNING, logger=_ranker_fs.logger.name):
        _ranker_fs.group_aware_mrmr_select(X, y, groups, bins=4)
    assert any("skipped 1 non-numeric" in r.getMessage() for r in caplog.records)
