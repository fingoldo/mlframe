"""Audit 2026-11-20 finding 35-F49: the uncached transform-pair path forwards refit_supervised, like the memoised path does."""

from types import SimpleNamespace

import numpy as np

import mlframe.training.composite.ensemble as facade


def test_memo_none_path_forwards_refit_supervised(monkeypatch):
    """With no memo, the call reaches _transform_pair_via with refit_supervised as given."""
    seen = {}

    def spy(pp, X_train, X_holdout, *, y_train=None, refit_supervised=False):
        """Record the refit flag and return a placeholder pair."""
        seen["refit"] = refit_supervised
        return X_train, X_holdout

    monkeypatch.setattr(facade, "_transform_pair_via", spy)
    monkeypatch.setattr(facade, "_pp_is_fitted", lambda pp: True)
    pp = SimpleNamespace()
    X = np.zeros((3, 1))
    facade._transform_pair_cached(None, pp, X, X, y_train=np.zeros(3), refit_supervised=True)
    assert seen["refit"] is True
