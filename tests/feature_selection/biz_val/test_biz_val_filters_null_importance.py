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
    rng = np.random.default_rng(seed)
    X_informative = rng.standard_normal((n, n_informative))
    weights = rng.uniform(0.5, 1.5, size=n_informative)
    y = X_informative @ weights + 0.3 * rng.standard_normal(n)
    X_noise = rng.standard_normal((n, n_noise))
    X = np.column_stack([X_informative, X_noise])
    return X, y, n_informative, n_noise


def _rf_importance_fn(X, y):
    model = RandomForestRegressor(n_estimators=30, max_depth=6, n_jobs=-1, random_state=0)
    model.fit(X, y)
    return model.feature_importances_


def test_null_importance_filter_returns_expected_keys_and_shapes():
    X, y, n_inf, n_noise = _make_mixed_signal_noise_data(400, 3, 5, seed=0)
    result = null_importance_filter(X, y, _rf_importance_fn, n_shuffles=8, random_state=0)
    n_features = n_inf + n_noise
    assert result["real_importance"].shape == (n_features,)
    assert result["null_importances"].shape == (8, n_features)
    assert result["threshold"].shape == (n_features,)
    assert result["keep_mask"].shape == (n_features,)


def test_null_importance_filter_keep_mask_is_boolean():
    X, y, *_ = _make_mixed_signal_noise_data(300, 2, 3, seed=1)
    result = null_importance_filter(X, y, _rf_importance_fn, n_shuffles=5, random_state=0)
    assert result["keep_mask"].dtype == np.bool_


def test_biz_val_null_importance_filter_rejects_far_more_noise_than_raw_positive_threshold():
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
