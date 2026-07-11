"""biz_value test for ``evaluation.distribution_matching_subset_search.distribution_matching_subset_search``.

Source: 1st_lanl-earthquake-prediction.md -- "sampling 10 full earthquakes multiple times (up to 10k times) on
train, comparing the average KS statistic of all selected features on the sampled earthquakes to the feature
distributions in full test." On a synthetic where each block's true relationship (slope) depends on that
block's own feature regime, a downstream model trained on a KS-distribution-matched block subset should
generalize far better to a target distribution than one trained on a naive (unmatched) random block subset --
mismatched blocks carry a systematically WRONG relationship for the target regime, not just extra noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from mlframe.evaluation.distribution_matching_subset_search import distribution_matching_subset_search


def _make_regime_dependent_blocks(n_blocks_total: int, rows_per_block: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks_total):
        offset = rng.uniform(-5, 5)
        beta = 2.0 + 0.5 * offset  # regime-specific slope, correlated with the block's own x location.
        x = rng.normal(loc=offset, scale=1.0, size=rows_per_block)
        y = beta * x + rng.normal(scale=0.3, size=rows_per_block)
        for i in range(rows_per_block):
            rows.append({"block": b, "x": x[i], "y": y[i]})
    return pd.DataFrame(rows)


def test_biz_val_matched_subset_beats_naive_subset_on_target_regime():
    train_df = _make_regime_dependent_blocks(n_blocks_total=30, rows_per_block=40, seed=0)

    rng = np.random.default_rng(0)
    target_offset = 3.0
    target_beta = 2.0 + 0.5 * target_offset
    target_x = rng.normal(loc=target_offset, scale=1.0, size=500)
    target_y_true = target_beta * target_x + rng.normal(scale=0.3, size=500)
    target_df = pd.DataFrame({"x": target_x})

    result = distribution_matching_subset_search(train_df, target_df, block_col="block", feature_cols=["x"], n_blocks=5, n_trials=300, random_state=0)

    naive_blocks = np.random.default_rng(1).choice(train_df["block"].unique(), size=5, replace=False)

    matched_train = train_df[train_df["block"].isin(result["best_blocks"])]
    naive_train = train_df[train_df["block"].isin(naive_blocks)]

    model_matched = Ridge().fit(matched_train[["x"]], matched_train["y"])
    model_naive = Ridge().fit(naive_train[["x"]], naive_train["y"])

    rmse_matched = float(mean_squared_error(target_y_true, model_matched.predict(target_df[["x"]])) ** 0.5)
    rmse_naive = float(mean_squared_error(target_y_true, model_naive.predict(target_df[["x"]])) ** 0.5)

    assert rmse_matched < rmse_naive * 0.5, f"expected the distribution-matched subset to beat a naive random subset by >=50% RMSE on the target regime, got matched={rmse_matched:.4f} naive={rmse_naive:.4f}"
    assert result["best_score"] < 0.3, f"expected the best-found subset's mean KS statistic to indicate a genuinely close match, got {result['best_score']:.4f}"


def test_distribution_matching_subset_search_rejects_n_blocks_exceeding_available():
    import pytest

    train_df = pd.DataFrame({"block": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    target_df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        distribution_matching_subset_search(train_df, target_df, block_col="block", n_blocks=5, n_trials=10)


def test_distribution_matching_subset_search_all_scores_shape():
    train_df = pd.DataFrame({"block": np.repeat(np.arange(10), 5), "x": np.random.default_rng(0).normal(size=50)})
    target_df = pd.DataFrame({"x": np.random.default_rng(1).normal(size=50)})
    result = distribution_matching_subset_search(train_df, target_df, block_col="block", n_blocks=3, n_trials=20, random_state=0)
    assert result["all_scores"].shape == (20,)
    assert len(result["best_blocks"]) == 3
