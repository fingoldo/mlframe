"""The surrogate seeder logs (throttled, WARNING) when lightgbm is missing or its fit fails, instead of returning silently."""

import logging
import sys

import numpy as np

from mlframe.feature_selection.filters import _surrogate_interaction_seeder as S
from mlframe.utils.log_throttle import reset_throttle_counts


def _data():
    """Small ordinal screening matrix and binary target."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 5, size=(200, 3)), rng.integers(0, 2, size=200)


def test_missing_lightgbm_warns(monkeypatch, caplog):
    """An ImportError of lightgbm yields no booster and a WARNING."""
    reset_throttle_counts()
    monkeypatch.setitem(sys.modules, "lightgbm", None)
    X, y = _data()
    with caplog.at_level(logging.WARNING):
        bst, score = S._fit_surrogate_and_oof(X, y, is_classification=True, n_estimators=5, max_depth=2, num_leaves=4, learning_rate=0.1,
                                              min_data_in_leaf=5, n_threads=1, random_seed=0, shuffle_y=False)
    assert bst is None and np.isnan(score)
    assert any("lightgbm is unavailable" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_failed_lightgbm_fit_warns(monkeypatch, caplog):
    """A lightgbm.train failure yields no booster and a throttled WARNING."""
    import lightgbm as lgb

    reset_throttle_counts()

    def boom(*a, **k):
        """Always fail."""
        raise RuntimeError("boom")

    monkeypatch.setattr(lgb, "train", boom)
    X, y = _data()
    with caplog.at_level(logging.WARNING):
        bst, _ = S._fit_surrogate_and_oof(X, y, is_classification=True, n_estimators=5, max_depth=2, num_leaves=4, learning_rate=0.1,
                                          min_data_in_leaf=5, n_threads=1, random_seed=0, shuffle_y=False)
    assert bst is None
    assert any("LightGBM fit failed" in r.getMessage() for r in caplog.records)
