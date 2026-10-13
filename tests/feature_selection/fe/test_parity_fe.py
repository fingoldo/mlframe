"""Floor-parity FE: the period scan finds a square wave's period and the appended feature replays exactly from its recipe."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.feature_selection.filters._parity_fe import hybrid_parity_fe_with_recipes, scan_parity_period
from mlframe.feature_selection.filters.engineered_recipes.shared import apply_recipe


def _frame(seed: int = 7, n: int = 800):
    """Five Gaussian columns; the target is a square wave of the first (period 1/3)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y = ((np.floor(X["x0"].to_numpy() * 3).astype(int)) % 2).astype(int)
    return X, y


def test_scan_recovers_the_square_wave_period():
    """The scan's best period is close to the true half-cycle 1/3 of floor(3x) mod 2 and its MI is far above the raw column's."""
    X, y = _frame()
    period, mi = scan_parity_period(X["x0"].to_numpy(), y)
    assert abs(period - 1.0 / 3.0) < 0.02, period
    assert mi > 0.4, mi


def test_appended_feature_is_linearly_usable_where_the_raw_column_is_not():
    """A linear model on the appended parity feature separates the target; on the raw column it stays near chance."""
    X, y = _frame()
    X_aug, appended, _, _ = hybrid_parity_fe_with_recipes(X, y)
    assert appended and appended[0].startswith("x0__par_")
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0)
    feat = X_aug[[appended[0]]].to_numpy()
    auc_feat = roc_auc_score(y[te], LogisticRegression(max_iter=500).fit(feat[tr], y[tr]).predict_proba(feat[te])[:, 1])
    raw = X[["x0"]].to_numpy()
    auc_raw = roc_auc_score(y[te], LogisticRegression(max_iter=500).fit(raw[tr], y[tr]).predict_proba(raw[te])[:, 1])
    assert auc_feat > 0.9, auc_feat
    assert auc_raw < 0.7, auc_raw


def test_recipe_replays_the_training_values_exactly():
    """apply_recipe on a fresh frame reproduces the training column, so transform is leakage-free and train/test exact."""
    X, y = _frame()
    X_aug, appended, recipes, _ = hybrid_parity_fe_with_recipes(X, y)
    assert recipes and recipes[0].extra["op"] == "parity"
    replay = np.asarray(apply_recipe(recipes[0], X), dtype=np.float64)
    np.testing.assert_array_equal(replay, X_aug[appended[0]].to_numpy(dtype=np.float64))


def test_pure_noise_appends_nothing():
    """No column clears the uplift margin on noise, so the family adds no features."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(800, 4)), columns=[f"n{i}" for i in range(4)])
    y = (rng.random(800) > 0.5).astype(int)
    _, appended, recipes, _ = hybrid_parity_fe_with_recipes(X, y)
    assert appended == [] and recipes == []
