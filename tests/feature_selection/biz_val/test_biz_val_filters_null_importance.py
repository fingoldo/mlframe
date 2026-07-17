"""biz_value + unit tests for ``feature_selection.filters.null_importance_filter``.

The win: on a synthetic mix of 5 genuinely informative features and 30 pure-noise features, a raw
(unfiltered) importance-based selector keeps essentially ALL noise features (every feature a tree can split
on gets SOME nonzero importance), while ``null_importance_filter`` correctly rejects the large majority of
them (each noise feature's real importance rarely clears the 95th percentile of its own null distribution,
by the definition of a percentile test) while retaining every genuinely informative feature.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from mlframe.feature_selection.filters._null_importance import null_importance_filter


def _make_mixed_signal_noise_data(n: int, n_informative: int, n_noise: int, seed: int):
    """Make mixed signal noise data."""
    rng = np.random.default_rng(seed)
    X_informative = rng.standard_normal((n, n_informative))
    weights = rng.uniform(0.5, 1.5, size=n_informative)
    y = X_informative @ weights + 0.3 * rng.standard_normal(n)
    X_noise = rng.standard_normal((n, n_noise))
    X = np.column_stack([X_informative, X_noise])
    return X, y, n_informative, n_noise


def _rf_importance_fn(X, y):
    """Rf importance fn."""
    model = RandomForestRegressor(n_estimators=30, max_depth=6, n_jobs=-1, random_state=0)
    model.fit(X, y)
    return model.feature_importances_


def test_null_importance_filter_returns_expected_keys_and_shapes():
    """Null importance filter returns expected keys and shapes."""
    X, y, n_inf, n_noise = _make_mixed_signal_noise_data(400, 3, 5, seed=0)
    result = null_importance_filter(X, y, _rf_importance_fn, n_shuffles=8, random_state=0)
    n_features = n_inf + n_noise
    assert result["real_importance"].shape == (n_features,)
    assert result["null_importances"].shape == (8, n_features)
    assert result["threshold"].shape == (n_features,)
    assert result["keep_mask"].shape == (n_features,)


def test_null_importance_filter_keep_mask_is_boolean():
    """Null importance filter keep mask is boolean."""
    X, y, *_ = _make_mixed_signal_noise_data(300, 2, 3, seed=1)
    result = null_importance_filter(X, y, _rf_importance_fn, n_shuffles=5, random_state=0)
    assert result["keep_mask"].dtype == np.bool_


def test_biz_val_null_importance_filter_rejects_far_more_noise_than_raw_positive_threshold():
    """Biz val null importance filter rejects far more noise than raw positive threshold."""
    n_informative, n_noise = 5, 30
    X, y, _, _ = _make_mixed_signal_noise_data(2000, n_informative, n_noise, seed=42)

    result = null_importance_filter(X, y, _rf_importance_fn, n_shuffles=30, percentile=95.0, random_state=7)

    informative_keep = result["keep_mask"][:n_informative]
    noise_keep = result["keep_mask"][n_informative:]

    # every genuinely informative feature should clear its own null bar.
    assert informative_keep.all(), f"expected all {n_informative} informative features kept, got {informative_keep.sum()}"

    # naive baseline: "raw importance > 0" keeps essentially every feature a tree ever split on --
    # RandomForest with max_depth=6 over 35 features on n=2000 will find SOME split for nearly all of them.
    naive_keep_rate = float((result["real_importance"] > 0).mean())
    null_filtered_noise_keep_rate = float(noise_keep.mean())

    assert naive_keep_rate > 0.85, f"sanity: naive raw-importance>0 filter should keep almost everything, got rate={naive_keep_rate:.2f}"
    assert null_filtered_noise_keep_rate < naive_keep_rate - 0.5, (
        f"null-importance filtering should reject far more noise features than the naive positive-importance "
        f"baseline: null_filtered_noise_keep_rate={null_filtered_noise_keep_rate:.2f} naive_keep_rate={naive_keep_rate:.2f}"
    )
    # the null-filtered noise keep rate should be roughly consistent with the 5% false-positive rate implied
    # by a 95th-percentile threshold (generous margin for small-sample/correlated-noise variance).
    assert null_filtered_noise_keep_rate < 0.30, f"null-filtered noise keep rate should be low, got {null_filtered_noise_keep_rate:.2f}"


def test_biz_val_null_importance_filter_margin_score_ranks_by_true_signal_strength():
    # 3 tiers of genuinely informative features with clearly different signal strength, plus pure noise.
    # keep_mask alone can't distinguish a barely-clearing-the-bar feature from a towering one -- both are
    # just True/kept. margin_score should recover the true strong > medium > weak ordering, which the raw
    # real_importance alone is noisier at doing near the decision boundary (weak signal vs strongest noise).
    """Biz val null importance filter margin score ranks by true signal strength."""
    rng = np.random.default_rng(123)
    n = 2500
    n_noise = 20

    X_strong = rng.standard_normal((n, 3))
    X_medium = rng.standard_normal((n, 3))
    X_weak = rng.standard_normal((n, 3))
    X_noise = rng.standard_normal((n, n_noise))

    y = 2.5 * X_strong.sum(axis=1) + 0.8 * X_medium.sum(axis=1) + 0.15 * X_weak.sum(axis=1) + 0.5 * rng.standard_normal(n)
    X = np.column_stack([X_strong, X_medium, X_weak, X_noise])
    tier_labels = np.array(["strong"] * 3 + ["medium"] * 3 + ["weak"] * 3 + ["noise"] * n_noise)
    true_rank_strength = np.array([3] * 3 + [2] * 3 + [1] * 3 + [0] * n_noise)  # ground-truth ordering

    result = null_importance_filter(X, y, _rf_importance_fn, n_shuffles=30, percentile=95.0, random_state=11, return_margin_score=True)

    assert "margin_score" in result
    assert result["margin_score"].shape == (3 + 3 + 3 + n_noise,)

    # margin_score must rank the three signal tiers in the correct strong > medium > weak order (medians).
    strong_margin = result["margin_score"][tier_labels == "strong"]
    medium_margin = result["margin_score"][tier_labels == "medium"]
    weak_margin = result["margin_score"][tier_labels == "weak"]
    noise_margin = result["margin_score"][tier_labels == "noise"]

    assert np.median(strong_margin) > np.median(medium_margin) > np.median(weak_margin) > np.median(noise_margin), (
        f"expected strong > medium > weak > noise margin_score, got "
        f"strong={np.median(strong_margin):.2f} medium={np.median(medium_margin):.2f} "
        f"weak={np.median(weak_margin):.2f} noise={np.median(noise_margin):.2f}"
    )

    # quantitative threshold: rank correlation (Spearman) between margin_score and ground-truth tier strength
    # should be strong and comfortably beat a floor set below the measured value.
    from scipy.stats import spearmanr

    corr, _ = spearmanr(result["margin_score"], true_rank_strength)
    assert corr > 0.55, f"expected margin_score rank-correlation with true signal strength > 0.55, got {corr:.3f}"

    # opting in must not perturb the pre-existing keys: mask-only call is bit-identical to the non-opted call.
    baseline = null_importance_filter(X, y, _rf_importance_fn, n_shuffles=30, percentile=95.0, random_state=11)
    assert np.array_equal(baseline["real_importance"], result["real_importance"])
    assert np.array_equal(baseline["null_importances"], result["null_importances"])
    assert np.array_equal(baseline["threshold"], result["threshold"])
    assert np.array_equal(baseline["keep_mask"], result["keep_mask"])
    assert "margin_score" not in baseline
