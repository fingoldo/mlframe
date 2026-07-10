"""biz_value test for ``preprocessing.rare_count_pruning`` (``collapse_rare_categories``, ``drop_rare_features``).

The win (3rd_mercedes-benz-greener-manufacturing.md): on a small-N dataset, a categorical column with many
rare (near-unique) values gives a tree model a near-infinite number of trivial single-row splits to overfit
to -- pure noise memorization that hurts held-out generalization. Collapsing rare values into a single
"other" bucket removes that overfitting surface while preserving the informative, well-populated categories.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlframe.preprocessing.rare_count_pruning import collapse_rare_categories, drop_rare_features


def _make_small_n_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    # a few genuinely informative, well-populated categories...
    informative_cat = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.35, 0.25])
    cat_effect = {"A": 2.0, "B": -1.0, "C": 0.5}
    # ...plus a high-cardinality column dominated by near-unique (rare) values carrying NO real signal.
    noise_cat = rng.choice([f"rare_{i}" for i in range(n // 2)], size=n)

    y = np.array([cat_effect[c] for c in informative_cat]) + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"informative": informative_cat, "noisy_high_card": noise_cat})
    return df, y


def test_biz_val_collapse_rare_categories_reduces_overfitting_on_small_n():
    df, y = _make_small_n_dataset(n=400, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)

    def _encode(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.get_dummies(frame)

    model_raw = RandomForestRegressor(n_estimators=100, random_state=0)
    model_raw.fit(_encode(X_train), y_train)
    X_test_encoded_raw = _encode(X_test).reindex(columns=_encode(X_train).columns, fill_value=0)
    mse_raw = mean_squared_error(y_test, model_raw.predict(X_test_encoded_raw))

    X_train_collapsed = collapse_rare_categories(X_train, ["noisy_high_card"], min_count=5)
    X_test_collapsed = collapse_rare_categories(X_test, ["noisy_high_card"], min_count=5)
    model_collapsed = RandomForestRegressor(n_estimators=100, random_state=0)
    train_encoded = _encode(X_train_collapsed)
    model_collapsed.fit(train_encoded, y_train)
    X_test_encoded_collapsed = _encode(X_test_collapsed).reindex(columns=train_encoded.columns, fill_value=0)
    mse_collapsed = mean_squared_error(y_test, model_collapsed.predict(X_test_encoded_collapsed))

    assert mse_collapsed < mse_raw, f"expected collapsing rare high-cardinality categories to reduce overfitting-driven test MSE, got collapsed={mse_collapsed:.4f} raw={mse_raw:.4f}"


def test_collapse_rare_categories_exact_behavior():
    df = pd.DataFrame({"cat": ["a", "a", "a", "b", "c", "d"]})
    out = collapse_rare_categories(df, ["cat"], min_count=2, other_label="OTHER")
    assert list(out["cat"]) == ["a", "a", "a", "OTHER", "OTHER", "OTHER"]


def test_drop_rare_features_flags_sparse_binary_indicator():
    n = 200
    df = pd.DataFrame({"common": np.concatenate([np.ones(100), np.zeros(100)]), "sparse": np.concatenate([np.ones(10), np.zeros(190)])})
    dropped = drop_rare_features(df, min_total_count=20)
    assert "sparse" in dropped
    assert "common" not in dropped
