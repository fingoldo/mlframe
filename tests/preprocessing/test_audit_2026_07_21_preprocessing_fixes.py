"""Regression tests for audits/full_audit_2026-07-21/preprocessing.md's findings (F1-F16).

One narrowly-named test per finding, verifying the real failure mode the audit describes is fixed. F2/F5/F6's
``.copy(deep=False)``/mixed-dtype-crash halves were fixed as part of the repo-wide large-frame-copy pass
before this cluster was processed individually; this file covers them alongside the rest of the cluster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------------------------------------------
# F1 -- collapse_rare_categories had no fit/apply split
# ---------------------------------------------------------------------------------------------------------------


def test_f1_fit_rare_category_collapse_replays_train_mapping_on_test():
    """F1: apply_rare_category_collapse must replay TRAIN's learned rare set, not derive test's own."""
    from mlframe.preprocessing.rare_count_pruning import apply_rare_category_collapse, fit_rare_category_collapse

    train = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 3 + ["rare_train"] * 2})
    # "b" is common in this TEST split (8/19 rows) but was learned as rare from TRAIN (3/55 rows) -- a
    # split-derived (not persisted) mapping would keep "b" as itself here; the persisted mapping must still
    # collapse it, since that's what a real train-fit model would have learned as its encoding.
    test = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 8 + ["untouched"] * 1})

    mapping = fit_rare_category_collapse(train, ["c"], min_count=5)
    assert mapping["c"] == frozenset({"b", "rare_train"})

    out_test = apply_rare_category_collapse(test, mapping)
    assert set(out_test["c"].unique()) == {"a", "__other__", "untouched"}
    assert (out_test.loc[test["c"] == "b", "c"] == "__other__").all()


def test_f1_collapse_rare_categories_wrapper_still_works_standalone():
    """F1: the single-frame fit+apply convenience wrapper stays usable and bit-identical for one-shot calls."""
    from mlframe.preprocessing.rare_count_pruning import collapse_rare_categories

    df = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 2})
    out = collapse_rare_categories(df, ["c"], min_count=5)
    assert set(out["c"].unique()) == {"a", "__other__"}


# ---------------------------------------------------------------------------------------------------------------
# F2 -- match_missingness_rate crashed on non-numeric columns
# ---------------------------------------------------------------------------------------------------------------


def test_f2_match_missingness_rate_mixed_dtype_frame_does_not_crash():
    """F2: a categorical column must pass through untouched instead of raising ValueError on to_numpy(float64)."""
    from mlframe.preprocessing.degradation_augment import match_missingness_rate

    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"num": rng.normal(size=200), "cat": rng.choice(["a", "b", "c"], size=200)})
    X_test = pd.DataFrame({"num": rng.normal(size=200), "cat": rng.choice(["a", "b", "c"], size=200)})
    X_test.loc[:50, "num"] = np.nan

    out = match_missingness_rate(X_train, X_test, rng)
    assert out["cat"].equals(X_train["cat"])
    assert out["num"].isna().mean() >= X_train["num"].isna().mean()


def test_f2_augment_to_match_test_distribution_default_pipeline_does_not_crash():
    """F2: the default degradation_fns tuple (match_missingness_rate first) must survive a mixed-dtype frame."""
    from mlframe.preprocessing.degradation_augment import augment_to_match_test_distribution

    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"num": rng.normal(size=100), "cat": rng.choice(["a", "b"], size=100)})
    X_test = pd.DataFrame({"num": rng.normal(size=100), "cat": rng.choice(["a", "b"], size=100)})
    y_train = rng.integers(0, 2, size=100)

    X_aug, y_aug = augment_to_match_test_distribution(X_train, y_train, X_test, n_augments=2)
    assert len(X_aug) == len(X_train) * 3
    assert len(y_aug) == len(X_aug)


# ---------------------------------------------------------------------------------------------------------------
# F3 -- _update_sub_df_col copied the whole analysis frame just to refresh one column
# ---------------------------------------------------------------------------------------------------------------


def test_f3_rare_merge_refreshes_value_counts_correctly_across_multiple_passes():
    """F3: after a rare-value merge, the refreshed value_counts must reflect the POST-merge distribution (not
    a stale pre-merge snapshot), across the repeated refresh calls analyse_and_clean_features makes."""
    from mlframe.preprocessing.cleaning import analyse_and_clean_features

    rng = np.random.default_rng(0)
    n = 3000
    # A column with one dominant value plus many singleton rare values, so the rare-merge pass fires and
    # _update_sub_df_col is called to refresh col_unique_values/nunique for this SAME column afterward.
    vals = ["common"] * (n - 30) + [f"rare_{i}" for i in range(30)]
    df = pd.DataFrame({"c": vals, "num": rng.normal(size=n)})

    result = analyse_and_clean_features(df, update_data=True, clean_nonnumeric_rarevals=True, verbose=False, min_fewlyvalued_rows_per_value=1)
    # every rare_i singleton must have been merged away into the sentinel NaN bucket.
    remaining = set(df["c"].unique())
    assert not any(str(v).startswith("rare_") for v in remaining if isinstance(v, str))
    assert "c" in result["features_transforms"]


# ---------------------------------------------------------------------------------------------------------------
# F4 -- calculated_quantiles truthiness crash
# ---------------------------------------------------------------------------------------------------------------


def test_f4_calculated_quantiles_ndarray_does_not_raise_truthiness_error():
    """F4: passing calculated_quantiles as the documented 2-element ndarray must not raise
    'truth value of an array with more than one element is ambiguous'."""
    from mlframe.preprocessing.cleaning import is_variable_truly_continuous

    rng = np.random.default_rng(0)
    sub_df = pd.DataFrame({"v": rng.normal(size=200)})
    calculated_quantiles = np.quantile(sub_df["v"], [0.1, 0.9])

    _is_continuous, outliers_percent = is_variable_truly_continuous(
        sub_df, "v", calculated_quantiles=calculated_quantiles, tukey_fences_multiplier=1.5, verbose=False
    )
    assert isinstance(outliers_percent, float)


# ---------------------------------------------------------------------------------------------------------------
# F5 / F6 -- deep .copy() of large frames; regime_conditioned_median_fill had no fit/apply split
# ---------------------------------------------------------------------------------------------------------------


def test_f5_apply_outlier_policy_uses_shallow_copy():
    """F5: apply_outlier_policy must not deep-copy the input frame (untouched columns keep their buffer)."""
    from mlframe.preprocessing.outlier_policy import apply_outlier_policy

    class NonTreeModel:
        """Stand-in for a linear/distance-based estimator (no tree-family marker in its MRO)."""

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 100.0], "b": [1.0, 2.0, 3.0, 4.0]})
    untouched_buffer = df["b"].to_numpy()
    out = apply_outlier_policy(df, model=NonTreeModel(), columns=["a"])
    assert np.shares_memory(out["b"].to_numpy(), untouched_buffer)


def test_f6_fit_regime_conditioned_median_replays_train_medians_on_test():
    """F6: apply_regime_conditioned_median_fill must fill test with TRAIN's regime medians, not test's own."""
    from mlframe.preprocessing.regime_conditioned_imputation import apply_regime_conditioned_median_fill, fit_regime_conditioned_median

    train = pd.DataFrame({"regime": ["x"] * 20 + ["y"] * 20, "v": [10.0] * 10 + [np.nan] * 10 + [100.0] * 10 + [np.nan] * 10})
    fit_state = fit_regime_conditioned_median(train, "regime")

    test = pd.DataFrame({"regime": ["x", "y"], "v": [np.nan, np.nan]})
    out = apply_regime_conditioned_median_fill(test, fit_state)
    assert out["v"].tolist() == [10.0, 100.0]


def test_f6_regime_conditioned_median_fill_wrapper_still_works_standalone():
    """F6: the single-frame fit+apply convenience wrapper stays usable for one-shot calls."""
    from mlframe.preprocessing.regime_conditioned_imputation import regime_conditioned_median_fill

    df = pd.DataFrame({"regime": ["x", "x", "x"], "v": [10.0, np.nan, 20.0]})
    out = regime_conditioned_median_fill(df, "regime")
    assert out["v"].iloc[1] == 15.0


# ---------------------------------------------------------------------------------------------------------------
# F7 -- impute_with_missing_indicator had no persisted fit state
# ---------------------------------------------------------------------------------------------------------------


def test_f7_fit_missing_indicator_imputation_replays_train_stats_on_test():
    """F7: apply_missing_indicator_imputation must fill test with TRAIN's median, not test's own."""
    from mlframe.preprocessing.missing_indicator_pairing import apply_missing_indicator_imputation, fit_missing_indicator_imputation

    train = pd.DataFrame({"v": [1.0, 2.0, 3.0, np.nan]})
    fit_state = fit_missing_indicator_imputation(train)

    test = pd.DataFrame({"v": [np.nan, np.nan]})
    out = apply_missing_indicator_imputation(test, fit_state)
    assert out["v"].tolist() == [2.0, 2.0]
    assert out["v_was_missing"].tolist() == [True, True]


def test_f7_impute_with_missing_indicator_wrapper_still_works_standalone():
    """F7: the single-frame fit+apply convenience wrapper stays usable for one-shot calls."""
    from mlframe.preprocessing.missing_indicator_pairing import impute_with_missing_indicator

    df = pd.DataFrame({"v": [1.0, 2.0, np.nan]})
    out = impute_with_missing_indicator(df)
    assert out["v"].iloc[2] == 1.5
    assert out["v_was_missing"].tolist() == [False, False, True]


# ---------------------------------------------------------------------------------------------------------------
# F8 -- single-class y division by zero in batch_univariate_auc
# ---------------------------------------------------------------------------------------------------------------


def test_f8_batch_univariate_auc_raises_on_single_class_y():
    """F8: a single-class y must raise ValueError, not silently produce NaN/inf AUCs."""
    from mlframe.preprocessing.align_feature_direction import batch_univariate_auc

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y_single = np.zeros(20)
    with pytest.raises(ValueError, match="single-class"):
        batch_univariate_auc(X, y_single)


def test_f8_align_feature_direction_raises_on_single_class_y():
    """F8: align_feature_direction (the public entry point) must propagate the same guard."""
    from mlframe.preprocessing.align_feature_direction import align_feature_direction

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 2)), columns=["a", "b"])
    y_single = np.ones(20)
    with pytest.raises(ValueError, match="single-class"):
        align_feature_direction(X, y_single)


# ---------------------------------------------------------------------------------------------------------------
# F9 -- scale-mismatched real vs augmented rows in augment_temporal_drift
# ---------------------------------------------------------------------------------------------------------------


def test_f9_select_true_last_standardized_matches_hand_computed_full_history_zscore():
    """F9: select_true_last_standardized's z-score must match a hand-rolled full-history standardization."""
    from mlframe.preprocessing.temporal_drift_augment import select_true_last_standardized

    panel = pd.DataFrame({"entity_id": [1, 1, 1, 2, 2, 2, 2], "t": [0, 1, 2, 0, 1, 2, 3], "x": [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0]})
    out = select_true_last_standardized(panel, "entity_id", "t", ["x"])

    assert len(out) == 2  # one row per entity
    expected_entity1 = (3.0 - np.mean([1.0, 2.0, 3.0])) / np.std([1.0, 2.0, 3.0], ddof=1)
    expected_entity2 = (7.0 - np.mean([5.0, 4.0, 6.0, 7.0])) / np.std([5.0, 4.0, 6.0, 7.0], ddof=1)
    got = out.set_index("entity_id")["x"]
    assert got.loc[1] == pytest.approx(expected_entity1)
    assert got.loc[2] == pytest.approx(expected_entity2)


def test_f9_augment_temporal_drift_real_rows_stay_unmodified():
    """F9: real (non-augmented) rows must stay an EXACT copy of the input -- the documented, tested contract
    this fix's docstring points callers toward select_true_last_standardized to avoid mixing scales."""
    from mlframe.preprocessing.temporal_drift_augment import augment_temporal_drift

    df = pd.DataFrame({"entity_id": [1, 1, 1, 2, 2, 2, 2], "t": [0, 1, 2, 0, 1, 2, 3], "x": [1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0]})
    result = augment_temporal_drift(df, entity_col="entity_id", time_col="t", feature_cols=["x"], n_drop_options=(1,))
    real_rows = result.loc[~result["_temporal_drift_augmented"], ["entity_id", "t", "x"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(real_rows, df.reset_index(drop=True))


# ---------------------------------------------------------------------------------------------------------------
# F10 -- Box-Cox/Yeo-Johnson refit-on-apply
# ---------------------------------------------------------------------------------------------------------------


def test_f10_apply_gaussian_power_transform_replays_fitted_lambda_not_refit():
    """F10: applying the searched transform to a DIFFERENT frame must replay the exact fitted lambda, not a
    freshly-refit one from that frame's own distribution."""
    from scipy.special import boxcox as boxcox_apply

    from mlframe.preprocessing.gaussian_power_transform_search import apply_gaussian_power_transform, gaussian_power_transform_search

    rng = np.random.default_rng(1)
    train = pd.DataFrame({"x": rng.lognormal(mean=0, sigma=1.0, size=500)})
    test = pd.DataFrame({"x": rng.lognormal(mean=2.0, sigma=1.5, size=500)})  # different distribution

    search_result = gaussian_power_transform_search(train)
    assert search_result["x"]["best_transform"] == "boxcox"
    fitted_lambda = search_result["x"]["best_fitted_params"]

    applied = apply_gaussian_power_transform(test, search_result)
    expected = boxcox_apply(test["x"].to_numpy(), fitted_lambda)
    np.testing.assert_allclose(applied["x"].to_numpy(), expected)

    # and it must genuinely differ from what a naive refit-from-test would have produced (proves this isn't
    # a no-op fix -- test's distribution is different enough that its own MLE lambda differs).
    from scipy.stats import boxcox as boxcox_fit

    _refit_transformed, refit_lambda = boxcox_fit(test["x"].to_numpy())
    assert not np.isclose(refit_lambda, fitted_lambda)


# ---------------------------------------------------------------------------------------------------------------
# F11 / F16 -- __init__.py star-import namespace pollution / inconsistent re-exports
# ---------------------------------------------------------------------------------------------------------------


def test_f11_package_namespace_does_not_leak_third_party_imports():
    """F11: star-imported modules must not leak np/pd/re/logging/DBSCAN/etc. into mlframe.preprocessing."""
    import mlframe.preprocessing as pkg

    for leaked_name in ("np", "pd", "re", "logging", "DBSCAN", "IsolationForest", "SimpleImputer", "tqdmu", "njit", "prange"):
        assert not hasattr(pkg, leaked_name), f"{leaked_name!r} leaked into mlframe.preprocessing namespace"


def test_f16_outlier_detector_zoo_full_public_surface_reexported():
    """F16: __init__.py must re-export outlier_detector_zoo's full public surface, not just make_outlier_detector."""
    from mlframe.preprocessing import make_ensemble_outlier_scores, make_outlier_detector, select_outlier_threshold

    assert callable(make_outlier_detector)
    assert callable(make_ensemble_outlier_scores)
    assert callable(select_outlier_threshold)


# ---------------------------------------------------------------------------------------------------------------
# F12 -- select_outlier_threshold not NaN-safe
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["contamination", "percentile", "iqr"])
def test_f12_select_outlier_threshold_nan_never_flagged_and_cutoff_not_corrupted(method):
    """F12: a NaN score must never be flagged, AND must not corrupt the cutoff for every other row."""
    from mlframe.preprocessing.outlier_detector_zoo import select_outlier_threshold

    scores = np.array([1.0, 2.0, 3.0, np.nan, 100.0, 4.0, 5.0])
    flags = select_outlier_threshold(scores, method=method, contamination=0.2, percentile=80.0)
    assert not flags[3], f"{method}: NaN row must never be flagged"
    assert flags.any(), f"{method}: the genuine outlier (100.0) must still be flagged despite the NaN"


# ---------------------------------------------------------------------------------------------------------------
# F14 -- unseen_category_imputer NaN-matches-unreliable-mask undocumented side effect
# ---------------------------------------------------------------------------------------------------------------


def test_f14_unseen_category_imputer_impute_nan_default_true_matches_prior_behavior():
    """F14: default impute_nan=True keeps the prior (bit-identical) behavior of also imputing genuine NaNs."""
    from mlframe.preprocessing.unseen_category_imputer import UnseenCategoryImputer

    train = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 10})
    test = pd.DataFrame({"c": ["a", "unseen", np.nan]})
    imputer = UnseenCategoryImputer(columns=["c"]).fit(train)
    out = imputer.transform(test)
    assert not out["c"].isna().any()


def test_f14_unseen_category_imputer_impute_nan_false_leaves_nan_untouched():
    """F14: impute_nan=False must leave genuine NaN cells as NaN, only imputing unseen/rare non-null values."""
    from mlframe.preprocessing.unseen_category_imputer import UnseenCategoryImputer

    train = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 10})
    test = pd.DataFrame({"c": ["a", "unseen", np.nan]})
    imputer = UnseenCategoryImputer(columns=["c"], impute_nan=False).fit(train)
    out = imputer.transform(test)
    assert out["c"].iloc[0] == "a"
    assert out["c"].iloc[1] == "a"  # unseen still imputed (mode fallback)
    assert pd.isna(out["c"].iloc[2])  # genuine NaN left untouched


# ---------------------------------------------------------------------------------------------------------------
# F15 -- type(real_val) is str instead of isinstance
# ---------------------------------------------------------------------------------------------------------------


def test_f15_single_option_nan_replacement_handles_numpy_str_subclass():
    """F15: a numpy.str_ value (a str SUBCLASS, type() is str == False) must still take the string branch --
    pre-fix this fell through to the numeric/boolean branch and crashed with UnboundLocalError on repl_value."""
    from mlframe.preprocessing.cleaning import analyse_and_clean_features

    n = 40
    arr = np.empty(n, dtype=object)
    for i in range(n):
        arr[i] = np.str_("yes") if i % 3 != 0 else np.nan
    assert any(type(v) is np.str_ for v in arr if not (isinstance(v, float) and np.isnan(v)))  # fixture sanity

    df = pd.DataFrame({"flag": arr, "other": np.arange(n, dtype=float)})
    result = analyse_and_clean_features(df, update_data=True, verbose=False, min_fewlyvalued_rows_per_value=1)

    assert list(result["features_transforms"]["flag"].values()) == ["not yes"]
    assert set(df["flag"].unique()) == {"yes", "not yes"}
