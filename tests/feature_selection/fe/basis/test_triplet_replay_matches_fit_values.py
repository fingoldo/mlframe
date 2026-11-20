"""The triplet recipe must replay the generator's values: float64 axis at fit and params frozen from the mean-filled column."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_triplet_fe import generate_triplet_cross_basis_features
from mlframe.feature_selection.filters._orthogonal_triplet_recipes import _triplet_recipe_for_name
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _frame(nan_row: bool) -> pd.DataFrame:
    """Three standard-normal columns, optionally with one NaN in the first."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.standard_normal(500) for c in "abc"})
    if nan_row:
        X.loc[3, "a"] = np.nan
    return X


def _replay_gap(X: pd.DataFrame) -> float:
    """Largest absolute gap between the generated column and its recipe replay."""
    out = generate_triplet_cross_basis_features(X, [("a", "b", "c")], basis="hermite", max_degree=1)
    name = out.columns[0]
    rec = _triplet_recipe_for_name(name, X, "hermite")
    assert rec is not None
    generated = out[name].to_numpy().astype(np.float32).astype(np.float64)  # the replay emits float32
    return float(np.abs(np.asarray(apply_recipe(rec, X), dtype=np.float64) - generated).max())


def test_triplet_replay_matches_fit_values_on_finite_data() -> None:
    """A float32 fit axis drifted from the float64 replay axis on finite data."""
    assert _replay_gap(_frame(False)) < 1e-9


def test_triplet_replay_matches_fit_values_with_nan_leg() -> None:
    """A NaN leg froze NaN params, so the replayed column was all zeros."""
    assert _replay_gap(_frame(True)) < 1e-9
