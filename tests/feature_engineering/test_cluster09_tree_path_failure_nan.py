"""Cluster 09 regression: a failed path extraction gives NaN path features, not constant 1.0 (audit 2026-11-20, finding 09-F17).

An empty path list evaluates every path as matched, so a failure looked like full agreement. The failed fold now emits NaN for the path and
match-count columns and logs a WARNING.
"""

import logging
from unittest.mock import patch

import numpy as np

from mlframe.feature_engineering.transformer import tree_path_boolean as tpb


def test_extraction_failure_gives_nan_path_columns(caplog):
    """With path extraction forced to fail, the path and n_matched columns are NaN and a WARNING is logged (09-F17)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] + rng.normal(size=60) * 0.1
    Xq = rng.normal(size=(10, 3))
    with patch.object(tpb, "_extract_top_paths", side_effect=RuntimeError("boom")), caplog.at_level(logging.WARNING):
        out = tpb.compute_tree_path_boolean_features(X, y, Xq, None, seed=0, task="regression", n_paths=4)
    arr = out.to_numpy()
    assert np.isnan(arr[:, :5]).all()
    assert any(r.levelno == logging.WARNING and "path extraction failed" in r.getMessage() for r in caplog.records)
