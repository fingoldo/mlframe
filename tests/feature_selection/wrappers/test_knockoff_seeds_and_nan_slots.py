"""Knockoff noise and swap coins use independent streams, and a NaN never marks the real slot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.wrappers._knockoffs import knockoff_importance


class _Recorder:
    """Importance-bearing model that records the matrix it was fitted on."""

    seen: list = []

    def fit(self, X, y):
        """Store the fitted matrix and give every column unit importance."""
        _Recorder.seen.append(np.asarray(X, dtype=np.float64))
        self.feature_importances_ = np.ones(np.asarray(X).shape[1])
        return self

    def predict(self, X):
        """Predict zeros."""
        return np.zeros(len(X))


def _fit(X, seed):
    """Run knockoff_importance with the recorder and return the fitted joint matrix."""
    _Recorder.seen = []
    knockoff_importance(_Recorder, X, np.zeros(len(X)), random_state=seed)
    return _Recorder.seen[-1]


def test_swap_coins_are_not_the_first_draws_of_the_shared_seed():
    """With 40 features the coin pattern must differ from default_rng(seed).random(p) < 0.5, which the shared seed used to produce."""
    rng = np.random.default_rng(1)
    p, seed = 40, 7
    X = pd.DataFrame(rng.normal(size=(200, p)), columns=[f"f{i}" for i in range(p)])
    joint = _fit(X, seed)
    first_is_real = np.array([np.allclose(joint[:, 2 * i], X.iloc[:, i].to_numpy()) for i in range(p)])
    old_coins = np.random.default_rng(seed).random(p) < 0.5
    assert not np.array_equal(~first_is_real, old_coins)


def test_nan_in_input_leaves_no_nan_in_either_slot():
    """NaNs are imputed in the real slot as in the knockoff slot, so the model cannot spot the real column by its gaps."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 6))
    X[rng.random(X.shape) < 0.1] = np.nan
    joint = _fit(X, 3)
    assert not np.isnan(joint).any()
