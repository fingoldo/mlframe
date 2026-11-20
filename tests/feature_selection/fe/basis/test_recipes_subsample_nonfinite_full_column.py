"""A subsample-decided winner whose full column holds a NaN must not be emitted with NaN-valued params."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe import hybrid_orth_mi_fe_with_recipes


def test_subsample_decision_drops_winner_with_nan_in_full_column() -> None:
    """The NaN sat outside the decision subsample, so the full-n rebuild produced an all-NaN column."""
    n, sub, seed = 4000, 1500, 42
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = (X["a"].to_numpy() ** 2 > 1.0).astype(int)
    kept = np.sort(np.random.default_rng(seed).choice(n, size=sub, replace=False))
    outside = int(np.setdiff1d(np.arange(n), kept)[0])
    X.loc[outside, "a"] = np.nan
    X_aug, _scores, recipes = hybrid_orth_mi_fe_with_recipes(X, y, subsample_n=sub, subsample_seed=seed, degrees=(2,), basis="hermite")
    appended = [c for c in X_aug.columns if c not in X.columns]
    assert not [c for c in appended if c.startswith("a__")]
    assert len(recipes) == len(appended)
    assert all(np.isfinite(X_aug[c].to_numpy()).all() for c in appended)
