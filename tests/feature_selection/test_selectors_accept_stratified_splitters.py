"""Three selectors crashed on the splitter every classification user reaches for first.

`greedy_backward_elimination`, `iterative_zero_importance_pruning` and `stochastic_bandit_selection` each
declared a CV splitter argument and then called `cv.split(X)` -- or `cv.split(np.empty(n))` -- without the
target. `KFold` ignores `y`, so every existing test passed. `StratifiedKFold` requires it, and raised
`TypeError: split() missing 1 required positional argument: 'y'` on the first fold.

Found while building benchmark arms around these three: a stratified splitter is the natural choice for a
binary target, and none of the three could accept one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination
from mlframe.feature_selection.stochastic_bandit_selection import stochastic_bandit_selection
from mlframe.feature_selection.zero_importance_pruning import iterative_zero_importance_pruning


@pytest.fixture()
def bed() -> tuple:
    """A small binary bed with three informative columns among six, imbalanced enough to want stratification."""
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    logit = 1.4 * X["f0"] - 1.1 * X["f1"] + 0.9 * X["f2"] - 1.2
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int).to_numpy()
    return X, y


def _stratified() -> StratifiedKFold:
    """The splitter each selector used to reject."""
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


def test_greedy_backward_elimination_accepts_a_stratified_splitter(bed: tuple) -> None:
    """The elimination runs to completion and keeps at least one informative column."""
    X, y = bed
    kept = greedy_backward_elimination(LogisticRegression(max_iter=200), X, y, accuracy_score, cv=_stratified(), min_features=2)
    names = [str(c) for c in (kept.columns if hasattr(kept, "columns") else kept)]
    assert names, "elimination returned nothing"
    assert {"f0", "f1", "f2"} & set(names)


def test_zero_importance_pruning_accepts_a_stratified_splitter(bed: tuple) -> None:
    """The pruning runs to completion on a stratified split."""
    X, y = bed
    result = iterative_zero_importance_pruning(
        LogisticRegression(max_iter=200), X, y, accuracy_score, cv=_stratified(), max_rounds=3,
        importance_fn=lambda model, frame, target: np.abs(np.asarray(model.coef_)).ravel(),
    )
    assert len(result) >= 1 and set(result) <= set(X.columns), result


def test_stochastic_bandit_selection_accepts_a_stratified_splitter(bed: tuple) -> None:
    """The bandit runs to completion on a stratified split and returns a subset of the requested size."""
    X, y = bed
    chosen = stochastic_bandit_selection(LogisticRegression(max_iter=200), X, y, accuracy_score, subset_size=3, n_epochs=8, cv=_stratified(), random_state=0)
    assert len(chosen) == 3
    assert set(chosen) <= set(X.columns)
