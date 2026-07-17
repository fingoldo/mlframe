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

    assert rmse_mode < rmse_sentinel * 0.75, (
        f"expected mode-replacement to beat global-mean-fallback by >=25% RMSE, got mode={rmse_mode:.4f} sentinel={rmse_sentinel:.4f}"
    )


def _make_non_dominant_category_data(n_train: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    cat_means = {"A": 0.8, "B": 0.5, "C": 0.2, "D": 0.15}  # A (mean 0.8) is the dominant train mode.
    train_cat = rng.choice(list(cat_means), n_train, p=[0.7, 0.1, 0.1, 0.1])
    train_val = np.array([cat_means[c] for c in train_cat]) + rng.normal(scale=0.03, size=n_train)  # companion numeric column, correlated with y.
    train_y = np.array([cat_means[c] for c in train_cat]) + rng.normal(scale=0.05, size=n_train)
    train_df = pd.DataFrame({"cat": train_cat, "val": train_val})

    unseen_true_mean = 0.48  # unseen category behaves like the NON-dominant known category B (0.5), far from mode A (0.8).
    test_cat = np.array(rng.choice(["A", "B", "C", "D", "UNSEEN"], n_test, p=[0.1, 0.1, 0.1, 0.1, 0.6]), dtype=object)
    true_mean_map = dict(cat_means, UNSEEN=unseen_true_mean)
    test_val = np.array([true_mean_map[c] for c in test_cat]) + rng.normal(scale=0.03, size=n_test)
    test_y = np.array([true_mean_map[c] for c in test_cat]) + rng.normal(scale=0.05, size=n_test)
    test_df = pd.DataFrame({"cat": test_cat, "val": test_val})
    return train_df, train_y, test_df, test_y


def test_biz_val_unseen_category_imputer_nearest_beats_mode_for_non_dominant_unseen_category():
    train_df, train_y, test_df, test_y = _make_non_dominant_category_data(n_train=3000, n_test=800, seed=7)
    enc = train_df.assign(y=train_y).groupby("cat")["y"].mean()

    mode_imputer = UnseenCategoryImputer(columns=["cat"]).fit(train_df)
    test_mode = mode_imputer.transform(test_df)
    pred_mode = np.array([enc[c] for c in test_mode["cat"]])
    rmse_mode = float(np.sqrt(np.mean((pred_mode - test_y) ** 2)))

    nearest_imputer = UnseenCategoryImputer(columns=["cat"], similarity_mode="nearest", value_column="val").fit(train_df)
    test_nearest = nearest_imputer.transform(test_df)
    pred_nearest = np.array([enc[c] for c in test_nearest["cat"]])
    rmse_nearest = float(np.sqrt(np.mean((pred_nearest - test_y) ** 2)))

    assert rmse_nearest < rmse_mode * 0.6, (
        f"expected nearest-replacement to beat mode-replacement by >=40% RMSE, got nearest={rmse_nearest:.4f} mode={rmse_mode:.4f}"
    )


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


def test_unseen_category_imputer_default_transform_unaffected_by_fallback_stats_flag():
    # track_fallback_stats is a pure diagnostic add-on: default (False) must produce bit-identical output,
    # and fallback_stats_ must stay empty, regardless of whether transform() is even called.
    train_df, _, test_df, _ = _make_skewed_category_data(n_train=500, n_test=200, seed=5)

    baseline = UnseenCategoryImputer(columns=["cat"]).fit(train_df)
    out_baseline = baseline.transform(test_df)

    tracked_off = UnseenCategoryImputer(columns=["cat"], track_fallback_stats=False).fit(train_df)
    out_tracked_off = tracked_off.transform(test_df)

    pd.testing.assert_frame_equal(out_baseline, out_tracked_off)
    assert tracked_off.fallback_stats_ == {}


def test_biz_val_unseen_category_imputer_fallback_stats_flags_drifted_column():
    # Production blind spot: silent fallback gives no visibility into HOW OFTEN a column's unseen-value
    # rate is elevated. A rising fallback rate on a column signals category drift (schema change, new
    # category rollout, upstream bug) worth investigating -- this test proves the opt-in tracker actually
    # separates a drifted column from a stable one by a wide, alertable margin.
    rng = np.random.default_rng(11)
    n = 2000

    train_df = pd.DataFrame(
        {
            "stable": ["A"] * 1000 + ["B"] * 600 + ["C"] * 400,
            "drifted": ["A"] * 1000 + ["B"] * 600 + ["C"] * 400,
        }
    )
    stable_test = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    drifted_test = rng.choice(["A", "B", "C", "NEW1", "NEW2", "NEW3"], n, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    test_df = pd.DataFrame({"stable": stable_test, "drifted": drifted_test})

    imputer = UnseenCategoryImputer(columns=["stable", "drifted"], track_fallback_stats=True).fit(train_df)
    imputer.transform(test_df)

    stable_rate = imputer.fallback_stats_["stable"]["fallback_rate"]
    drifted_rate = imputer.fallback_stats_["drifted"]["fallback_rate"]

    assert stable_rate < 0.02, f"expected the stable column's fallback rate near 0, got {stable_rate:.4f}"
    assert drifted_rate > 0.35, f"expected the drifted column's fallback rate to clearly exceed an alert threshold, got {drifted_rate:.4f}"
    assert drifted_rate > stable_rate * 15, (
        f"expected drift to be starkly distinguishable from the stable baseline, got drifted={drifted_rate:.4f} stable={stable_rate:.4f}"
    )
