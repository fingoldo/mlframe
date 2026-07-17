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
    """Helper that make regime dependent blocks."""
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
    """Matched subset beats naive subset on target regime."""
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

    assert rmse_matched < rmse_naive * 0.5, (
        f"expected the distribution-matched subset to beat a naive random subset by >=50% RMSE on the target regime, got matched={rmse_matched:.4f} naive={rmse_naive:.4f}"
    )
    assert result["best_score"] < 0.3, f"expected the best-found subset's mean KS statistic to indicate a genuinely close match, got {result['best_score']:.4f}"


def test_distribution_matching_subset_search_rejects_n_blocks_exceeding_available():
    """Distribution matching subset search rejects n blocks exceeding available."""
    import pytest

    train_df = pd.DataFrame({"block": [1, 2, 3], "x": [1.0, 2.0, 3.0]})
    target_df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        distribution_matching_subset_search(train_df, target_df, block_col="block", n_blocks=5, n_trials=10)


def test_distribution_matching_subset_search_all_scores_shape():
    """Distribution matching subset search all scores shape."""
    train_df = pd.DataFrame({"block": np.repeat(np.arange(10), 5), "x": np.random.default_rng(0).normal(size=50)})
    target_df = pd.DataFrame({"x": np.random.default_rng(1).normal(size=50)})
    # all_scores.shape == (n_trials,) is only guaranteed for search_strategy="random" -- greedy_swap spends
    # part of the budget on single-block scoring, so its all_scores length can be < n_trials (see docstring).
    result = distribution_matching_subset_search(train_df, target_df, block_col="block", n_blocks=3, n_trials=20, random_state=0, search_strategy="random")
    assert result["all_scores"].shape == (20,)
    assert len(result["best_blocks"]) == 3


def test_biz_val_greedy_swap_beats_random_at_equal_budget():
    """A/B: "greedy_swap" vs "random" at the SAME total evaluation budget (n_trials), same seed, same synthetic.

    Measured (20 seeds x n_trials in {50, 100, 300, 1000}, scratch A/B script): greedy_swap's mean best-found
    KS statistic was 11.7-31.1% lower than random's, and downstream RMSE 11.1-36.1% lower, at EVERY tested
    budget -- a decisive, consistent win, which is why "greedy_swap" became the default search_strategy. This
    test pins a single representative budget (n_trials=100) with thresholds set below the measured margins.
    """
    train_df = _make_regime_dependent_blocks(n_blocks_total=30, rows_per_block=40, seed=0)

    rng = np.random.default_rng(0)
    target_offset = 3.0
    target_beta = 2.0 + 0.5 * target_offset
    target_x = rng.normal(loc=target_offset, scale=1.0, size=500)
    target_y_true = target_beta * target_x + rng.normal(scale=0.3, size=500)
    target_df = pd.DataFrame({"x": target_x})

    n_trials = 100
    result_random = distribution_matching_subset_search(
        train_df, target_df, block_col="block", feature_cols=["x"], n_blocks=5, n_trials=n_trials, random_state=0, search_strategy="random"
    )
    result_greedy = distribution_matching_subset_search(
        train_df, target_df, block_col="block", feature_cols=["x"], n_blocks=5, n_trials=n_trials, random_state=0, search_strategy="greedy_swap"
    )

    def rmse_for(best_blocks):
        """Helper that rmse for."""
        matched_train = train_df[train_df["block"].isin(best_blocks)]
        model = Ridge().fit(matched_train[["x"]], matched_train["y"])
        return float(mean_squared_error(target_y_true, model.predict(target_df[["x"]])) ** 0.5)

    rmse_random = rmse_for(result_random["best_blocks"])
    rmse_greedy = rmse_for(result_greedy["best_blocks"])

    assert result_greedy["best_score"] <= result_random["best_score"] * 0.90, (
        f"expected greedy_swap's best_score to beat random's by >=10% at equal budget, "
        f"got greedy={result_greedy['best_score']:.4f} random={result_random['best_score']:.4f}"
    )
    assert rmse_greedy <= rmse_random * 0.95, (
        f"expected greedy_swap's downstream RMSE to beat random's by >=5% at equal budget, got greedy={rmse_greedy:.4f} random={rmse_random:.4f}"
    )


def _make_correlated_blocks(n_blocks: int, rows_per_block: int, block_offset: int, seed: int) -> pd.DataFrame:
    """ "good" blocks: x and y share a latent factor z, so they're strongly correlated -- like the target."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        z = rng.normal(size=rows_per_block)
        x = z + rng.normal(scale=0.1, size=rows_per_block)
        y = z + rng.normal(scale=0.1, size=rows_per_block)
        for i in range(rows_per_block):
            rows.append({"block": block_offset + b, "x": x[i], "y": y[i]})
    return pd.DataFrame(rows)


def _make_decorrelated_decoy_blocks(n_blocks: int, rows_per_block: int, block_offset: int, seed: int) -> pd.DataFrame:
    """ "decoy" blocks: x and y are drawn from INDEPENDENT latent factors -- each marginal individually looks
    just like the target's (same z + noise construction per column), but the joint/correlation structure is
    destroyed (x,y uncorrelated here vs strongly correlated in the target). Per-feature KS cannot tell these
    apart from the "good" blocks; only a joint/multivariate check can.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        zx = rng.normal(size=rows_per_block)
        zy = rng.normal(size=rows_per_block)
        x = zx + rng.normal(scale=0.1, size=rows_per_block)
        y = zy + rng.normal(scale=0.1, size=rows_per_block)
        for i in range(rows_per_block):
            rows.append({"block": block_offset + b, "x": x[i], "y": y[i]})
    return pd.DataFrame(rows)


def test_biz_val_distribution_matching_subset_search_joint_energy_catches_correlation_mismatch():
    """A block subset can match every feature's marginal (near-zero per-feature KS, same as a genuinely good
    subset) while having a completely different joint correlation structure -- per-feature KS is blind to
    this by construction (it only ever looks at one column at a time). ``joint_distance_mode="energy"`` scores
    the full standardized feature vector at once and should clearly separate the two, while the default
    (univariate-only) scoring should NOT -- proving the new mode catches a real gap the old one misses.
    """
    good_df = _make_correlated_blocks(n_blocks=10, rows_per_block=60, block_offset=0, seed=0)
    decoy_df = _make_decorrelated_decoy_blocks(n_blocks=10, rows_per_block=60, block_offset=1000, seed=1)

    rng = np.random.default_rng(2)
    target_z = rng.normal(size=1500)
    target_x = target_z + rng.normal(scale=0.1, size=1500)
    target_y = target_z + rng.normal(scale=0.1, size=1500)
    target_df = pd.DataFrame({"x": target_x, "y": target_y})

    # n_blocks == the exact number of blocks present in each frame -- "random" with n_trials=1 deterministically
    # scores that ONE full candidate set (no sampling ambiguity), letting us compare the two fixed candidates directly.
    def score(df: pd.DataFrame, joint_distance_mode):
        """Helper that score."""
        result = distribution_matching_subset_search(
            df,
            target_df,
            block_col="block",
            feature_cols=["x", "y"],
            n_blocks=10,
            n_trials=1,
            random_state=0,
            search_strategy="random",
            joint_distance_mode=joint_distance_mode,
        )
        return result["best_score"]

    good_univariate = score(good_df, None)
    decoy_univariate = score(decoy_df, None)
    good_joint = score(good_df, "energy")
    decoy_joint = score(decoy_df, "energy")

    # Measured: good_univariate=0.0362, decoy_univariate=0.0455 (nearly indistinguishable -- univariate-only
    # scoring WRONGLY treats the decorrelated decoy as an equally good match). good_joint=0.0385,
    # decoy_joint=0.1661 (~4.3x higher -- the joint mode CORRECTLY flags the decoy as a much worse match).
    assert decoy_univariate < good_univariate * 2.0, (
        f"expected the decoy's univariate (per-feature KS) score to be nearly as good as the genuine match's, "
        f"proving marginal-only checks are blind to the correlation mismatch -- got decoy={decoy_univariate:.4f} good={good_univariate:.4f}"
    )
    assert decoy_joint > good_joint * 2.0, (
        f"expected joint_distance_mode='energy' to score the decorrelated decoy at least 2x worse than the "
        f"genuine correlated match, catching the joint-structure mismatch that univariate KS missed -- "
        f"got decoy={decoy_joint:.4f} good={good_joint:.4f}"
    )


def test_distribution_matching_subset_search_joint_mode_default_is_bit_identical():
    """``joint_distance_mode`` is strictly opt-in -- omitting it (or passing ``None`` explicitly) must produce
    a bit-identical result to code that predates this parameter entirely.
    """
    train_df = _make_regime_dependent_blocks(n_blocks_total=20, rows_per_block=20, seed=3)
    target_df = pd.DataFrame({"x": np.random.default_rng(4).normal(size=200)})

    for search_strategy in ["random", "greedy_swap"]:
        r_default = distribution_matching_subset_search(
            train_df, target_df, block_col="block", feature_cols=["x"], n_blocks=5, n_trials=60, random_state=5, search_strategy=search_strategy
        )
        r_explicit_none = distribution_matching_subset_search(
            train_df,
            target_df,
            block_col="block",
            feature_cols=["x"],
            n_blocks=5,
            n_trials=60,
            random_state=5,
            search_strategy=search_strategy,
            joint_distance_mode=None,
        )
        assert r_default["best_score"] == r_explicit_none["best_score"]
        assert r_default["best_blocks"] == r_explicit_none["best_blocks"]
        assert np.array_equal(r_default["all_scores"], r_explicit_none["all_scores"])
