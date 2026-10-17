"""The target-mix pre-check separates a target that still holds structure from one that is noise."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._target_mix_gate import target_has_structure


def _frame(n=6000, seed=0):
    """Six continuous columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"c{j}": rng.normal(size=n) for j in range(6)})


def test_pure_noise_has_no_structure():
    """A target independent of every column is skipped."""
    X = _frame()
    t = np.random.default_rng(1).normal(size=len(X))
    assert not target_has_structure(X, t)


def test_a_one_column_term_is_structure():
    """A sine of one column under noise is found by the marginal screen."""
    X = _frame()
    t = np.sin(3 * X["c2"].to_numpy()) + 0.5 * np.random.default_rng(2).normal(size=len(X))
    assert target_has_structure(X, t)


def test_a_pure_interaction_is_structure():
    """A product of two columns (no main effect) is found by the pair screen."""
    X = _frame()
    t = X["c1"].to_numpy() * X["c4"].to_numpy() + 0.5 * np.random.default_rng(3).normal(size=len(X))
    assert target_has_structure(X, t)
