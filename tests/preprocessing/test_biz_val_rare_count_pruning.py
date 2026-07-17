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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

    assert mse_collapsed < mse_raw, (
        f"expected collapsing rare high-cardinality categories to reduce overfitting-driven test MSE, got collapsed={mse_collapsed:.4f} raw={mse_raw:.4f}"
    )


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


def _make_target_aware_dataset(n: int, seed: int):
    """A rare-but-genuinely-informative category ("special") plus many rare-and-uninformative noise categories.

    "special" occurs below the naive min_count floor but its target rate (0.9) is wildly different from the
    0.1 baseline -- a real, distinctive rare signal. The ``noise_i`` categories occur at similar counts but
    carry no signal (target rate matches baseline) -- genuinely safe to collapse.
    """
    rng = np.random.default_rng(seed)
    groups = [f"noise_{i}" for i in range(n // 8)]
    cat = rng.choice(groups, size=n).astype(object)
    y = (rng.random(n) < 0.1).astype(float)
    special_idx = rng.choice(n, size=50, replace=False)
    cat[special_idx] = "special"
    y[special_idx] = (rng.random(50) < 0.9).astype(float)
    df = pd.DataFrame({"cat": cat})
    return df, y, special_idx


def _split_preserving_special(df: pd.DataFrame, y: np.ndarray, special_idx: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    test_special = special_idx[:25]
    train_special = special_idx[25:]
    rest = np.setdiff1d(np.arange(len(df)), special_idx)
    rng.shuffle(rest)
    test_rest = rest[:1000]
    train_rest = rest[1000:]
    test_idx = np.concatenate([test_special, test_rest])
    train_idx = np.concatenate([train_special, train_rest])
    return df.iloc[train_idx], y[train_idx], df.iloc[test_idx], y[test_idx]


def _fit_and_auc(X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray) -> float:
    train_enc = pd.get_dummies(X_train)
    test_enc = pd.get_dummies(X_test).reindex(columns=train_enc.columns, fill_value=0)
    clf = LogisticRegression(max_iter=200)
    clf.fit(train_enc, y_train)
    p = clf.predict_proba(test_enc)[:, 1]
    return float(roc_auc_score(y_test, p))


def test_biz_val_collapse_rare_categories_target_aware_preserves_informative_rare_category():
    df, y, special_idx = _make_target_aware_dataset(n=4000, seed=1)
    X_train, y_train, X_test, y_test = _split_preserving_special(df, y, special_idx, seed=2)

    X_train_naive = collapse_rare_categories(X_train, ["cat"], min_count=60)
    X_test_naive = collapse_rare_categories(X_test, ["cat"], min_count=60)
    auc_naive = _fit_and_auc(X_train_naive, y_train, X_test_naive, y_test)

    X_train_aware = collapse_rare_categories(X_train, ["cat"], min_count=60, y=y_train, target_aware=True)
    X_test_aware = collapse_rare_categories(X_test, ["cat"], min_count=60, y=y_test, target_aware=True)
    auc_aware = _fit_and_auc(X_train_aware, y_train, X_test_aware, y_test)

    assert "special" not in X_train_naive["cat"].unique(), (
        "naive count-only collapse is expected to fold the informative rare category into the catch-all bucket"
    )
    assert "special" in X_train_aware["cat"].unique(), "target-aware collapse must preserve a rare-but-genuinely-informative category"
    # measured: auc_naive=0.500 (informative rare category destroyed, indistinguishable from chance),
    # auc_aware=0.651 -- threshold set ~10% below the measured aware value, well above the naive floor.
    assert auc_aware >= 0.58, f"expected target-aware collapse to retain the informative rare category's signal, got auc_aware={auc_aware:.4f}"
    assert auc_aware > auc_naive + 0.05, f"expected target-aware AUC to clearly beat naive count-only collapse, got aware={auc_aware:.4f} naive={auc_naive:.4f}"


def test_collapse_rare_categories_target_aware_default_off_is_bit_identical():
    """target_aware=False (the default) must reproduce the prior count-only behavior exactly -- no y needed."""
    df = pd.DataFrame({"cat": ["a", "a", "a", "b", "c", "d"]})
    out_default = collapse_rare_categories(df, ["cat"], min_count=2, other_label="OTHER")
    out_explicit_off = collapse_rare_categories(df, ["cat"], min_count=2, other_label="OTHER", target_aware=False)
    assert list(out_default["cat"]) == list(out_explicit_off["cat"]) == ["a", "a", "a", "OTHER", "OTHER", "OTHER"]


def test_collapse_rare_categories_target_aware_requires_y():
    df = pd.DataFrame({"cat": ["a", "a", "a", "b", "c", "d"]})
    try:
        collapse_rare_categories(df, ["cat"], min_count=2, target_aware=True)
        assert False, "expected ValueError when target_aware=True without y"
    except ValueError:
        pass
