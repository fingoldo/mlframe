"""Cluster 29 regression: bagging without any randomisation warns that every bag is identical (audit 2026-11-20, finding 29-F41).

With n_bags > 1 but randomize_order=False and randomize_start=False, each bag replays the same greedy path. A WARNING now says so. With
randomisation on, no such warning is logged.
"""

import logging

import numpy as np

from mlframe.votenrank.hill_climb import hill_climb_ensemble


def _mse(y_true, y_pred):
    """Mean squared error used as the blend metric."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def _inputs():
    """Three noisy predictors of a shared target."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    preds = [y + rng.normal(size=60) * s for s in (0.1, 0.4, 0.8)]
    return preds, y


def test_bagging_without_randomisation_warns(caplog):
    """n_bags=3 with both randomisation flags off logs a WARNING (29-F41)."""
    preds, y = _inputs()
    with caplog.at_level(logging.WARNING):
        hill_climb_ensemble(preds, y, _mse, maximize=False, n_bags=3, randomize_order=False, randomize_start=False)
    assert any("identical" in r.getMessage() for r in caplog.records)


def test_bagging_with_randomisation_is_silent(caplog):
    """Control: with randomised order the bagging warning is not logged (29-F41)."""
    preds, y = _inputs()
    with caplog.at_level(logging.WARNING):
        hill_climb_ensemble(preds, y, _mse, maximize=False, n_bags=3, randomize_order=True, random_state=1)
    assert not any("identical" in r.getMessage() for r in caplog.records)
