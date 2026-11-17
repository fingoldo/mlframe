"""RFF median-distance bandwidth is offset-invariant, a non-finite RFF block is logged, and an entity-scoped lagged diff refuses to replay without its entity column."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _random_fourier_features_fe as rff
from mlframe.feature_selection.filters._ratio_delta_fe import apply_lagged_diff
from mlframe.utils.log_throttle import reset_throttle_counts


def test_median_pairwise_distance_is_offset_invariant_in_float32():
    """A large offset on float32 columns must not change the bandwidth heuristic."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(300, 4)).astype(np.float32)
    shifted = (base.astype(np.float64) + 1e4).astype(np.float32)
    ref = rff._median_pairwise_distance(base.astype(np.float64))
    got = rff._median_pairwise_distance(shifted)
    assert got == pytest.approx(ref, rel=2e-2)


def test_nonfinite_rff_block_is_logged(caplog):
    """A source block holding NaN yields no features and says why."""
    reset_throttle_counts()
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 5.0]})
    with caplog.at_level(logging.WARNING, logger=rff.logger.name):
        enc, payload = rff.generate_random_fourier_features_block(X, ["a", "b"], m=4)
    assert enc.shape[1] == 0 and payload == {}
    assert any("skipped" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_apply_lagged_diff_raises_when_the_entity_column_is_missing():
    """An entity-scoped recipe must not silently replay as a global-order diff."""
    X = pd.DataFrame({"t": [3, 1, 2, 4], "v": [1.0, 2.0, 3.0, 4.0]})
    recipe = {"time_col": "t", "value_col": "v", "period": 1, "entity_cols": ("e",)}
    with pytest.raises(KeyError, match="entity"):
        apply_lagged_diff(X, recipe)
