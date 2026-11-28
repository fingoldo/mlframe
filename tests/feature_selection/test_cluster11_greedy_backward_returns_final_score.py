"""Cluster 11 regression: greedy_backward_elimination can return the kept set's score, also when nothing was removed (audit 2026-11-20, finding 11-F17)."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination


def test_return_score_is_the_survivors_score_even_with_an_empty_trace():
    """Two informative columns and min_features=2: nothing is removed, the trace is empty, yet a finite score comes back."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=["a", "b"])
    y = X["a"] * 2 + X["b"]
    kept, trace, score = greedy_backward_elimination(LinearRegression(), X, y.to_numpy(), r2_score, min_features=2, return_trace=True, return_score=True)
    assert kept == ["a", "b"] and trace == () and np.isfinite(score) and score > 0.9


def test_return_score_alone_appends_one_element():
    """Without return_trace the result is ``(kept, score)``; the default call still returns the bare list."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc"))
    y = X["a"].to_numpy()
    out = greedy_backward_elimination(LinearRegression(), X, y, r2_score, return_score=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(greedy_backward_elimination(LinearRegression(), X, y, r2_score), list)
