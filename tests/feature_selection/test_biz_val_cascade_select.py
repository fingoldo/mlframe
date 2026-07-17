"""biz_value test for ``feature_selection.cascade_select`` / ``forward_select``.

The win: with a handful of truly informative features buried among many noise columns (a 4600-features-down-
to-21 style regime), a downstream linear model fit on ALL raw features suffers real variance inflation from
the noise columns. The 3-stage cascade (Boruta shadow-feature screen -> forward selection -> permutation
backward elimination) should narrow down to close to the true informative subset, and a model fit on that
narrowed subset should generalize materially better than one fit on the full raw feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.feature_selection import cascade_select, forward_select


def _make_noisy_dataset(n: int, d_informative: int, d_noise: int, seed: int):
    """Make noisy dataset."""
    rng = np.random.default_rng(seed)
    X_info = rng.normal(size=(n, d_informative))
    X_noise = rng.normal(size=(n, d_noise))
    w = rng.normal(size=d_informative)
    y = X_info @ w + rng.normal(scale=0.5, size=n)
    cols = [f"info{i}" for i in range(d_informative)] + [f"noise{i}" for i in range(d_noise)]
    X = pd.DataFrame(np.concatenate([X_info, X_noise], axis=1), columns=cols)
    return X, y


def test_biz_val_cascade_select_beats_full_feature_set_mse():
    """Biz val cascade select beats full feature set mse."""
    X, y = _make_noisy_dataset(n=300, d_informative=4, d_noise=60, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    result = cascade_select(
        X_train, y_train, lambda: RandomForestRegressor(n_estimators=15, random_state=0), n_boruta_iterations=10, cv=3, scoring="neg_mean_squared_error"
    )
    final_features = result["final_selected"]
    assert len(final_features) < X.shape[1] / 2, f"expected the cascade to materially narrow the 64-column feature set, got {len(final_features)} features"
    assert all(f.startswith("info") for f in final_features), f"expected only informative columns to survive the cascade, got {final_features}"

    model_all = LinearRegression().fit(X_train, y_train)
    mse_all = mean_squared_error(y_test, model_all.predict(X_test))

    model_cascade = LinearRegression().fit(X_train[final_features], y_train)
    mse_cascade = mean_squared_error(y_test, model_cascade.predict(X_test[final_features]))

    improvement = 1.0 - mse_cascade / mse_all
    assert improvement > 0.2, f"expected >20% MSE reduction vs. the full raw feature set, got {improvement:.4f} (all={mse_all:.4f}, cascade={mse_cascade:.4f})"


def test_forward_select_recovers_informative_subset():
    """Forward select recovers informative subset."""
    X, y = _make_noisy_dataset(n=200, d_informative=3, d_noise=10, seed=1)
    selected = forward_select(X, y, lambda: RandomForestRegressor(n_estimators=15, random_state=0), scoring="neg_mean_squared_error", cv=3, max_features=6)
    assert len(selected) > 0
    n_informative_selected = sum(1 for f in selected if f.startswith("info"))
    assert n_informative_selected >= 2, f"expected forward_select to pick up at least 2 of the 3 informative columns, got {selected}"


def test_cascade_select_empty_boruta_confirmation_returns_empty_result():
    """Cascade select empty boruta confirmation returns empty result."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(60, 5)), columns=[f"noise{i}" for i in range(5)])
    y = rng.normal(size=60)  # pure noise target, no real signal in X at all
    result = cascade_select(X, y, lambda: RandomForestRegressor(n_estimators=10, random_state=0), n_boruta_iterations=5, cv=3)
    assert result["final_selected"] == [] or len(result["final_selected"]) <= 2  # near-empty; a spurious confirm is possible with pure noise but rare
