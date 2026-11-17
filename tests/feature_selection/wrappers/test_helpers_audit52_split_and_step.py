"""Regression tests: positional row split with a duplicated index, and dichotomic_step forwarding to the scipy aliases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers import _helpers
from mlframe.feature_selection.wrappers._enums import OptimumSearch
from mlframe.feature_selection.wrappers._helpers import _pick_next_nfeatures, split_into_train_test


def test_split_by_feature_names_with_duplicated_row_index_is_positional():
    """Rows must be taken by position: a duplicated index label must not duplicate rows or leak them across the split."""
    X = pd.DataFrame(np.arange(40, dtype=float).reshape(10, 4), columns=list("abcd"), index=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y = pd.Series(np.arange(10), index=X.index)
    tr, te = np.array([0, 2, 4, 6]), np.array([1, 3, 5])
    X_tr, _y_tr, X_te, _y_te = split_into_train_test(X, y, tr, te, features_indices=["b", "d"])
    assert X_tr.shape == (4, 2) and X_te.shape == (3, 2)
    assert X_tr["b"].tolist() == X["b"].to_numpy()[tr].tolist()
    assert X_te["d"].tolist() == X["d"].to_numpy()[te].tolist()


def test_split_unknown_feature_name_raises_keyerror():
    """A feature name absent from X must raise, not be silently dropped."""
    X = pd.DataFrame(np.zeros((6, 2)), columns=["a", "b"])
    with pytest.raises(KeyError):
        split_into_train_test(X, np.zeros(6), np.array([0, 1]), np.array([2, 3]), features_indices=["a", "zzz"])


@pytest.mark.parametrize("method", [OptimumSearch.ScipyLocal, OptimumSearch.ScipyGlobal])
def test_scipy_search_methods_forward_dichotomic_step(monkeypatch, method):
    """dichotomic_step must reach the dichotomic suggester under the ScipyLocal / ScipyGlobal aliases."""
    seen = []

    def _spy(remaining, evaluated_scores_mean, n_total, epsilon=0.0, rng=None, step="auto"):
        """Record the step."""
        seen.append(step)
        return remaining[0]

    monkeypatch.setattr(_helpers, "_suggest_dichotomic", _spy)
    _pick_next_nfeatures(
        top_predictors_search_method=method, remaining=[1, 2, 3], n_total=3, evaluated_scores_mean={}, dichotomic_epsilon=0.0, dichotomic_step="auto",
        rng=None, Optimizer=None,
    )
    assert seen == ["auto"]


def test_split_by_feature_positions_with_duplicated_row_index():
    """Integer feature positions must also work (and stay positional) when the row index has duplicates."""
    X = pd.DataFrame(np.arange(40, dtype=float).reshape(10, 4), columns=list("abcd"), index=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y = np.arange(10)
    tr, te = np.array([0, 2, 4, 6]), np.array([1, 3, 5])
    X_tr, _, X_te, _ = split_into_train_test(X, y, tr, te, features_indices=np.array([1, 3]))
    assert X_tr.shape == (4, 2)
    assert X_tr["b"].tolist() == X["b"].to_numpy()[tr].tolist()
    assert X_te["d"].tolist() == X["d"].to_numpy()[te].tolist()
