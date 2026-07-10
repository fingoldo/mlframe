"""biz_value + unit tests for ``preprocessing.category_support.train_test_support_screen``.

The win: on a synthetic column where train/test category SETS are fully disjoint (jaccard=0, like distinct
user-ID batches per period) but each category's FREQUENCY within its own split still correlates with the
target, the screener correctly recommends frequency encoding over the raw column — and switching to
frequency encoding measurably preserves predictive signal on test that a train-fit target-mean encoding of
the raw category loses entirely (every test category is unseen in train -> falls back to the constant global
mean -> zero correlation with target on test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mlframe.preprocessing.category_support import train_test_support_screen


def _make_disjoint_category_data(n_train: int, n_test: int, n_cats_per_split: int, seed: int):
    """train uses category ids [0, n_cats_per_split), test uses [n_cats_per_split, 2*n_cats_per_split) --
    fully disjoint sets. Each category's row count within its own split follows a Zipf-like power law, and
    the target is driven by log(rarity) = -log(category frequency within its split), so the RARITY signal is
    shared across splits even though no specific category id transfers."""
    rng = np.random.default_rng(seed)

    def _draw_split(n_rows, cat_offset):
        zipf_weights = 1.0 / np.arange(1, n_cats_per_split + 1)
        zipf_weights /= zipf_weights.sum()
        cats = rng.choice(np.arange(n_cats_per_split) + cat_offset, size=n_rows, p=zipf_weights)
        freq = pd.Series(cats).map(pd.Series(cats).value_counts()).to_numpy()
        rarity = -np.log(freq / n_rows)
        y_prob = 1.0 / (1.0 + np.exp(-(1.5 * (rarity - rarity.mean()) + 0.3 * rng.standard_normal(n_rows))))
        y = (rng.random(n_rows) < y_prob).astype(np.float64)
        return cats, freq, y

    train_cats, train_freq, y_train = _draw_split(n_train, cat_offset=0)
    test_cats, test_freq, y_test = _draw_split(n_test, cat_offset=n_cats_per_split)

    train_df = pd.DataFrame({"cat_col": train_cats, "y": y_train})
    test_df = pd.DataFrame({"cat_col": test_cats, "y": y_test})
    return train_df, test_df, train_freq, test_freq


def test_train_test_support_screen_flags_disjoint_column_for_frequency_encoding():
    train_df, test_df, _, _ = _make_disjoint_category_data(3000, 3000, n_cats_per_split=150, seed=0)
    report = train_test_support_screen(train_df, test_df, categorical_cols=["cat_col"])

    row = report.iloc[0]
    assert row["column"] == "cat_col"
    assert row["jaccard_overlap"] == 0.0
    assert row["n_test_only_categories"] == row["n_test_categories"]
    assert row["frac_test_rows_unseen"] == 1.0
    assert row["recommendation"] == "frequency_encode"


def test_train_test_support_screen_keeps_raw_for_stable_column():
    rng = np.random.default_rng(1)
    n = 2000
    shared_cats = rng.integers(0, 20, size=n)
    train_df = pd.DataFrame({"stable_col": shared_cats[: n // 2]})
    test_df = pd.DataFrame({"stable_col": shared_cats[n // 2 :]})
    report = train_test_support_screen(train_df, test_df, categorical_cols=["stable_col"])
    assert report.iloc[0]["recommendation"] == "keep_raw"


def test_train_test_support_screen_drops_near_unique_id_column():
    n = 500
    train_df = pd.DataFrame({"uuid_col": [f"train_{i}" for i in range(n)]})
    test_df = pd.DataFrame({"uuid_col": [f"test_{i}" for i in range(n)]})
    report = train_test_support_screen(train_df, test_df, categorical_cols=["uuid_col"])
    assert report.iloc[0]["recommendation"] == "drop"


def test_train_test_support_screen_multiple_columns_and_default_column_selection():
    train_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "only_in_train": [7, 8, 9]})
    test_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    report = train_test_support_screen(train_df, test_df)
    assert set(report["column"]) == {"a", "b"}


def test_biz_val_frequency_encoding_preserves_signal_that_raw_target_mean_encoding_loses():
    train_df, test_df, train_freq, test_freq = _make_disjoint_category_data(4000, 4000, n_cats_per_split=200, seed=42)

    report = train_test_support_screen(train_df, test_df, categorical_cols=["cat_col"])
    assert report.iloc[0]["recommendation"] == "frequency_encode"

    # Raw encoding: train-fit target-mean-per-category, applied to test. Every test category is unseen in
    # train (jaccard=0 by construction) so every test row falls back to the train global mean -> constant
    # -> zero correlation with y_test by construction. Assert this failure mode directly (sanity check the
    # scenario is real, not assumed).
    train_target_mean = train_df.groupby("cat_col")["y"].mean()
    raw_encoded_test = test_df["cat_col"].map(train_target_mean).fillna(train_df["y"].mean())
    assert raw_encoded_test.nunique() == 1, "sanity: every test category should be unseen in train, forcing a constant fallback"
    corr_raw, _ = spearmanr(raw_encoded_test, test_df["y"])
    assert np.isnan(corr_raw) or abs(corr_raw) < 1e-9

    # Frequency encoding computed independently per split (no train->test category identity needed):
    # rarity signal transfers because the GENERATIVE PROCESS (Zipf frequency -> target) is shared, even
    # though the specific category ids are not.
    corr_freq_train, _ = spearmanr(train_freq, train_df["y"])
    corr_freq_test, _ = spearmanr(test_freq, test_df["y"])
    assert abs(corr_freq_train) > 0.15, f"sanity: frequency should correlate with target on train, got {corr_freq_train}"
    assert abs(corr_freq_test) > 0.15, (
        f"frequency encoding should preserve predictive signal on test where raw target-mean encoding "
        f"collapses to a constant (corr={corr_freq_test:.3f} vs raw corr~0)"
    )
