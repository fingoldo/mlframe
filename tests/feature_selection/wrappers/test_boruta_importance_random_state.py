"""The 'boruta' importance getter must shuffle its shadow columns with the random_state it is given."""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.wrappers._importance_methods import _boruta


class _RecordingModel:
    """Estimator that remembers the joint [X, shadow] design it was fitted on."""

    last_X = None

    def get_params(self, deep=True):
        """Return no parameters so sklearn.clone rebuilds the class."""
        return {}

    def set_params(self, **_):
        """Accept and ignore parameters."""
        return self

    def fit(self, X, y):
        """Store the design and expose a constant importance vector."""
        type(self).last_X = np.asarray(X).copy()
        self.feature_importances_ = np.ones(np.asarray(X).shape[1])
        return self


def test_boruta_shadow_shuffle_follows_random_state():
    """Two different random_state values give different shadow columns; the same value reproduces them."""
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.zeros(20)
    shadows = {}
    for seed in (1, 2, 1):
        _boruta(_RecordingModel(), X, y, None, random_state=seed)
        shadows.setdefault(seed, []).append(_RecordingModel.last_X[:, 2:].copy())
    assert not np.array_equal(shadows[1][0], shadows[2][0])
    assert np.array_equal(shadows[1][0], shadows[1][1])
