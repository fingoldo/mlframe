"""The noise-floor cut compares levels, so a falling re-ranked null cannot make the cut keep every feature.

The plateau rule compares increments. A null curve that falls as features are added has large positive increments
relative to the envelope, so the real curve's flat tail looks like signal and the cut keeps all features. The band rule
reads the same curves as levels and stops where the real curve stops improving.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from mlframe.feature_selection.wrappers._noise_floor import (
    noise_floor_band,
    noise_floor_plateau,
    select_features_noise_floor,
)


def _falling_null_flat_real():
    """A real curve that reaches its plateau at N=3 and a re-ranked-style null that falls with N across 20 draws."""
    n_grid = [1, 2, 3, 5, 8, 12, 20, 30, 43]
    real = np.array([0.66, 0.70, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82])
    rng = np.random.default_rng(0)
    base = np.linspace(0.75, 0.52, len(n_grid))
    perm = base[None, :] + rng.normal(scale=0.005, size=(20, len(n_grid)))
    return n_grid, real, perm


def test_band_cut_stops_at_the_plateau_when_the_null_falls():
    """On a falling null the band rule stops at the plateau; the plateau rule keeps every feature past it."""
    n_grid, real, perm = _falling_null_flat_real()
    assert noise_floor_band(n_grid, real, perm, pct=95.0) == 3
    assert noise_floor_plateau(n_grid, real, perm, pct=95.0)[0] == 43


def test_band_cut_is_the_default_stop_rule():
    """Callers get the band rule without asking for it; the increment rule is an explicit opt-out."""
    default = inspect.signature(select_features_noise_floor).parameters["stop_rule"].default
    assert default == "band"


def test_unknown_stop_rule_is_rejected():
    """A misspelled stop rule raises instead of silently running one of the two."""
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(60, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] > 0).astype(int))
    with pytest.raises(ValueError, match="stop_rule"):
        select_features_noise_floor(
            lambda: RandomForestClassifier(n_estimators=3, random_state=0), X, y, list(X.columns), n_grid=[1, 2], cv=2, n_perm=20, stop_rule="sometimes",
        )


@pytest.mark.slow
def test_reranked_null_on_pure_noise_keeps_few_features():
    """End-to-end: 40 pure-noise features ranked by correlation with a random label keep at most 3 under the re-ranked null.

    With the plateau rule this case keeps 8 features; the band rule keeps 2.
    """
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 40)), columns=[f"f{i}" for i in range(40)])
    y = pd.Series((rng.random(n) < 0.5).astype(int))
    corr = X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]))
    ranking = list(corr.sort_values(ascending=False).index)
    out = select_features_noise_floor(
        lambda: RandomForestClassifier(n_estimators=15, max_depth=4, random_state=0, n_jobs=1),
        X, y, ranking, n_grid=[1, 2, 3, 5, 8, 12, 20, 30, 43], cv=StratifiedKFold(3, shuffle=True, random_state=0),
        n_perm=20, rerank="permutation",
    )
    assert out["n_star"] <= 3
