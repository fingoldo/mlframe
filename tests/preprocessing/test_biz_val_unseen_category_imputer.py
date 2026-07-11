"""biz_value test for ``preprocessing.unseen_category_imputer.UnseenCategoryImputer``.

Source: 9th_home-credit-default-risk.md -- "Replace categories not found in train... replace all of these
with things previously encountered. Test set has no XNA genders..." A downstream target/frequency encoder
falls back to the GLOBAL mean for a category it never saw at train time (a sentinel-bucket outcome, since the
sentinel itself has no train-side target statistic). Mapping the unseen value to the train MODE category
instead gives it that category's real target statistic, which is a much better proxy whenever the unseen
category is close in behavior to the dominant train category (a realistic real-world case: a rare/typo
variant of a common category) -- this test confirms mode-replacement beats naive global-mean fallback on such
a synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.preprocessing.unseen_category_imputer import UnseenCategoryImputer


def _make_skewed_category_data(n_train: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    cat_means = {"A": 0.2, "B": 0.5, "C": 0.8, "D": 0.5}
    train_cat = rng.choice(list(cat_means), n_train, p=[0.1, 0.1, 0.7, 0.1])  # C (mean 0.8) is the train mode.
    train_y = np.array([cat_means[c] for c in train_cat]) + rng.normal(scale=0.05, size=n_train)
    train_df = pd.DataFrame({"cat": train_cat})

    unseen_true_mean = 0.78  # unseen categories behave like a variant of the dominant category C.
    test_cat = np.array(rng.choice(["A", "B", "C", "D", "UNSEEN1", "UNSEEN2"], n_test, p=[0.05, 0.05, 0.5, 0.05, 0.2, 0.15]), dtype=object)
    true_mean_map = dict(cat_means, UNSEEN1=unseen_true_mean, UNSEEN2=unseen_true_mean)
    test_y = np.array([true_mean_map[c] for c in test_cat]) + rng.normal(scale=0.05, size=n_test)
    test_df = pd.DataFrame({"cat": test_cat})
    return train_df, train_y, test_df, test_y


def test_biz_val_unseen_category_imputer_beats_global_mean_fallback():
    train_df, train_y, test_df, test_y = _make_skewed_category_data(n_train=3000, n_test=800, seed=3)

    enc = train_df.assign(y=train_y).groupby("cat")["y"].mean()
    global_mean = float(train_y.mean())

    pred_sentinel = np.array([enc.get(c, global_mean) for c in test_df["cat"]])
    rmse_sentinel = float(np.sqrt(np.mean((pred_sentinel - test_y) ** 2)))

    imputer = UnseenCategoryImputer(columns=["cat"]).fit(train_df)
    test_mapped = imputer.transform(test_df)
    pred_mode = np.array([enc.get(c, global_mean) for c in test_mapped["cat"]])
    rmse_mode = float(np.sqrt(np.mean((pred_mode - test_y) ** 2)))

    assert rmse_mode < rmse_sentinel * 0.75, f"expected mode-replacement to beat global-mean-fallback by >=25% RMSE, got mode={rmse_mode:.4f} sentinel={rmse_sentinel:.4f}"


def test_unseen_category_imputer_maps_rare_and_unseen_to_train_mode():
    train_df = pd.DataFrame({"cat": ["A"] * 90 + ["B"] * 5 + ["C"] * 5})
    imputer = UnseenCategoryImputer(columns=["cat"], min_count=10).fit(train_df)
    assert imputer.mode_["cat"] == "A"

    test_df = pd.DataFrame({"cat": ["A", "B", "C", "UNSEEN"]})
    out = imputer.transform(test_df)
    # B and C fall below min_count=10 -> also mapped to mode, alongside the genuinely unseen value.
    assert out["cat"].tolist() == ["A", "A", "A", "A"]


def test_unseen_category_imputer_leaves_known_frequent_categories_untouched():
    train_df = pd.DataFrame({"cat": ["A"] * 50 + ["B"] * 50})
    imputer = UnseenCategoryImputer(columns=["cat"]).fit(train_df)
    test_df = pd.DataFrame({"cat": ["A", "B", "A", "B"]})
    out = imputer.transform(test_df)
    assert out["cat"].tolist() == ["A", "B", "A", "B"]
