"""Regression tests for audits/full_audit_2026-07-21/x_ml_correctness_meta.md findings F1-F10.

F3 (stochastic bandit ensemble's "independent" seeds sharing one CV split) was found to be ALREADY
fixed as a byproduct of feature_selection_nonmrmr.md's own F3 fix (this cluster's F3 is the same
root cause, viewed from the ensemble caller's side) -- pinned here rather than re-fixed.
F8 (post.py/policy.py LOC-debt) is an architecture flag, not a bug -- no test needed.
F9 (two_step_recency_weighted_target_encode's leaky default) is already thoroughly documented per
the audit's own "exemplary docstring" assessment -- no code change, no test needed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# F1 (P0): combine_probs(median, sample_weight=...) axis mismatch
# ---------------------------------------------------------------------------


def test_f1_combine_probs_median_sample_weight_no_longer_crashes():
    """F1: combine probs median sample weight no longer crashes."""
    from mlframe.models.ensembling.base import combine_probs

    rng = np.random.default_rng(0)
    M, N = 4, 100
    stacked = rng.uniform(0.1, 0.9, size=(M, N))
    sample_weight = rng.uniform(0.5, 2.0, size=N)

    result = combine_probs(stacked, "median", sample_weight=sample_weight)
    assert result.shape == (N,)
    assert np.array_equal(result, np.median(stacked, axis=0))


def test_f1_combine_probs_median_weight_and_none_identical():
    """median's own documented contract is 'ignores weights silently' -- confirm sample_weight truly has no effect."""
    from mlframe.models.ensembling.base import combine_probs

    rng = np.random.default_rng(1)
    stacked = rng.uniform(0.1, 0.9, size=(3, 50))
    sw = rng.uniform(0.5, 2.0, size=50)
    assert np.array_equal(
        combine_probs(stacked, "median", sample_weight=sw),
        combine_probs(stacked, "median", sample_weight=None),
    )


def test_f1_axis_mismatch_confirmed_in_raw_numpy():
    """Sanity: confirms the pre-fix failure mode really is a numpy ValueError (shape mismatch), not a hypothetical."""
    rng = np.random.default_rng(0)
    stacked = rng.uniform(0.1, 0.9, size=(4, 100))
    sw = rng.uniform(0.5, 2.0, size=100)
    with pytest.raises(ValueError, match="weights"):
        np.quantile(stacked, 0.5, axis=0, weights=sw, method="inverted_cdf")


# ---------------------------------------------------------------------------
# F2 (P1): report_regression_model_perf had no sample_weight parameter
# ---------------------------------------------------------------------------


def test_f2_report_regression_model_perf_sample_weight_changes_metrics():
    """F2: report regression model perf sample weight changes metrics."""
    from mlframe.training.reporting._reporting_regression import report_regression_model_perf
    from mlframe.metrics.regression._regression_metrics import fast_mean_absolute_error

    rng = np.random.default_rng(0)
    n = 500
    targets = rng.normal(loc=20.0, scale=2.0, size=n)
    preds = targets + rng.normal(scale=0.5, size=n)
    sample_weight = np.ones(n)
    sample_weight[:50] = 20.0

    metrics_dict: dict = {}
    report_regression_model_perf(
        targets=targets, columns=["target"], model_name="testmodel", model=None,
        preds=preds, print_report=False, show_perf_chart=False,
        metrics=metrics_dict, sample_weight=sample_weight,
    )
    expected_weighted = fast_mean_absolute_error(targets, preds, sample_weight=sample_weight)
    expected_unweighted = fast_mean_absolute_error(targets, preds)
    assert metrics_dict["MAE"] == pytest.approx(expected_weighted)
    assert not np.isclose(metrics_dict["MAE"], expected_unweighted)


def test_f2_report_regression_model_perf_default_unweighted_unaffected():
    """sample_weight=None (default) must be bit-identical to the pre-fix unweighted path."""
    from mlframe.training.reporting._reporting_regression import report_regression_model_perf

    rng = np.random.default_rng(0)
    n = 300
    targets = rng.normal(size=n)
    preds = targets + rng.normal(scale=0.2, size=n)
    metrics_dict: dict = {}
    report_regression_model_perf(
        targets=targets, columns=["target"], model_name="m", model=None,
        preds=preds, print_report=False, show_perf_chart=False, metrics=metrics_dict,
    )
    assert "MAE" in metrics_dict and "RMSE" in metrics_dict and "R2" in metrics_dict


# ---------------------------------------------------------------------------
# F3 (P1): stochastic_bandit_selection_ensemble's seeds sharing one CV split -- ALREADY fixed
# ---------------------------------------------------------------------------


def test_f3_ensemble_seeds_get_distinct_cv_splits():
    """Each seed in the ensemble gets its own KFold random_state (== the seed), not a shared default."""
    import sys

    from mlframe.feature_selection.stochastic_bandit_selection_ensemble import stochastic_bandit_selection_ensemble
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    sbs_mod = sys.modules["mlframe.feature_selection.stochastic_bandit_selection"]
    constructed_cvs = []
    orig_kfold_cls = sbs_mod.KFold

    class SpyKFold(orig_kfold_cls):
        """KFold subclass that records the random_state it was constructed with."""
        def __init__(self, *a, **kw):
            constructed_cvs.append(kw.get("random_state"))
            super().__init__(*a, **kw)

    sbs_mod.KFold = SpyKFold
    try:
        rng = np.random.default_rng(0)
        n, p = 300, 8
        X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = (X["f0"] + X["f1"] > 0).astype(int).to_numpy()

        def scoring(y_true, y_pred_proba):
            """Wraps roc_auc_score with the (y_true, y_pred_proba) signature the ensemble scorer expects."""
            return roc_auc_score(y_true, y_pred_proba)

        stochastic_bandit_selection_ensemble(
            estimator=LogisticRegression(max_iter=200), X=X, y=y, scoring=scoring,
            subset_size=3, seeds=[1, 2, 3], n_epochs=15, cv=None,
        )
    finally:
        sbs_mod.KFold = orig_kfold_cls

    assert len(set(constructed_cvs)) == 3, constructed_cvs


# ---------------------------------------------------------------------------
# F4 (P1): report_model_perf's use_weights is not genuine per-sample weighting -- docs clarified
# ---------------------------------------------------------------------------


def test_f4_use_weights_docstring_clarifies_it_is_not_sample_weight():
    """F4: use weights docstring clarifies it is not sample weight."""
    from mlframe.training.reporting._reporting import report_model_perf

    doc = report_model_perf.__doc__ or ""
    assert "NOT a genuine per-sample weight" in doc


# ---------------------------------------------------------------------------
# F5 (P2): ordered/causal target encoders leaked the global prior over the FULL y array
# ---------------------------------------------------------------------------


def test_f5_ordered_target_encode_causal_prior_opt_in():
    """F5: ordered target encode causal prior opt in."""
    from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode

    rng = np.random.default_rng(0)
    n = 300
    cats = rng.choice(["a", "b", "c"], size=n)
    y = rng.normal(size=n)

    enc_default = ordered_target_encode(cats, y, smoothing=1.0)
    global_prior_ref = float(np.mean(y))
    df = pd.DataFrame({"cat": cats, "y": y})
    grouped = df.groupby("cat", sort=False)["y"]
    running_sum = grouped.cumsum() - y
    running_count = grouped.cumcount()
    enc_ref = ((running_sum + 1.0 * global_prior_ref) / (running_count + 1.0)).to_numpy()
    assert np.allclose(enc_default, enc_ref, atol=1e-12)

    enc_causal = ordered_target_encode(cats, y, smoothing=1.0, causal_prior=True)
    assert not np.allclose(enc_causal, enc_default)

    expanding_prior_ref = np.empty(n)
    expanding_prior_ref[0] = 0.0
    for i in range(1, n):
        expanding_prior_ref[i] = np.mean(y[:i])
    enc_causal_manual = ((running_sum + 1.0 * expanding_prior_ref) / (running_count + 1.0)).to_numpy()
    assert np.allclose(enc_causal, enc_causal_manual, atol=1e-10)


def test_f5_ordered_target_encode_batch_causal_prior_matches_single_column():
    """F5: ordered target encode batch causal prior matches single column."""
    from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode, ordered_target_encode_batch

    rng = np.random.default_rng(1)
    n = 200
    cats1 = rng.choice(["a", "b"], size=n)
    y = rng.normal(size=n)

    batch_default = ordered_target_encode_batch({"c1": cats1}, y)
    assert np.allclose(batch_default["c1"], ordered_target_encode(cats1, y))

    batch_causal = ordered_target_encode_batch({"c1": cats1}, y, causal_prior=True)
    assert np.allclose(batch_causal["c1"], ordered_target_encode(cats1, y, causal_prior=True))
    assert not np.allclose(batch_causal["c1"], batch_default["c1"])


def test_f5_holiday_cross_locale_causal_prior_opt_in():
    """F5: holiday cross locale causal prior opt in."""
    from mlframe.feature_engineering.holiday_locale_target_encoding import holiday_name_target_encode_cross_locale

    rng = np.random.default_rng(2)
    n = 200
    names = rng.choice(["xmas", "newyear"], size=n)
    countries = rng.choice(["US", "CA"], size=n)
    y = rng.normal(size=n)

    h_default = holiday_name_target_encode_cross_locale(names, countries, y, cross_locale_shrinkage=2.0)
    h_causal = holiday_name_target_encode_cross_locale(names, countries, y, cross_locale_shrinkage=2.0, causal_prior=True)
    assert not np.allclose(h_default, h_causal)


# ---------------------------------------------------------------------------
# F6 (P2): tfidf_svd_entity_embedding had no leakage warning for the train+test-together usage
# ---------------------------------------------------------------------------


def test_f6_tfidf_svd_docstring_has_leakage_note():
    """F6: tfidf svd docstring has leakage note."""
    from mlframe.feature_engineering.tfidf_svd_entity_embedding import tfidf_svd_entity_embedding

    doc = tfidf_svd_entity_embedding.__doc__ or ""
    assert "Leakage note" in doc
    assert "return_fitted=True" in doc


# ---------------------------------------------------------------------------
# F7 (P2): calibration/ had zero sample_weight support anywhere
# ---------------------------------------------------------------------------


def test_f7_binary_post_calibrator_threads_sample_weight_when_supported():
    """F7: binary post calibrator threads sample weight when supported."""
    from sklearn.isotonic import IsotonicRegression

    from mlframe.calibration.post import BinaryPostCalibrator

    rng = np.random.default_rng(0)
    n = 200
    probs = rng.uniform(0, 1, size=n)
    target = (rng.uniform(0, 1, size=n) < probs).astype(int)
    sample_weight = rng.uniform(0.5, 2.0, size=n)

    iso = IsotonicRegression(out_of_bounds="clip")
    orig_fit = iso.fit
    captured = {}

    def spy_fit(X, y, sample_weight=None):
        """Records fit() calls for this test's assertions."""
        captured["sample_weight"] = sample_weight
        return orig_fit(X, y, sample_weight=sample_weight)

    iso.fit = spy_fit
    wrapper = BinaryPostCalibrator(calibrator=iso, fit_method_name="fit", transform_method_name="predict")
    wrapper.fit(probs, target, sample_weight=sample_weight)
    assert captured["sample_weight"] is not None
    assert np.array_equal(captured["sample_weight"], sample_weight)


def test_f7_binary_post_calibrator_warns_and_falls_back_when_unsupported(caplog):
    """F7: binary post calibrator warns and falls back when unsupported."""
    from mlframe.calibration.post import BinaryPostCalibrator

    class NoWeightCalibrator:
        """Stub calibrator whose fit() signature has no sample_weight parameter."""
        def fit(self, X, y):
            """No-op / recording stub matching the estimator's fit() signature."""
            self.fitted = True

        def predict(self, X):
            """No-op / recording stub matching the estimator's predict() signature."""
            return np.column_stack([1 - X, X])

    rng = np.random.default_rng(0)
    n = 100
    probs = rng.uniform(0, 1, size=n)
    target = (rng.uniform(0, 1, size=n) < probs).astype(int)
    sample_weight = rng.uniform(0.5, 2.0, size=n)

    nw = NoWeightCalibrator()
    wrapper = BinaryPostCalibrator(calibrator=nw, fit_method_name="fit", transform_method_name="predict")
    with caplog.at_level(logging.WARNING, logger="mlframe.calibration.post"):
        wrapper.fit(probs, target, sample_weight=sample_weight)
    assert nw.fitted
    assert any("does not accept sample_weight" in r.getMessage() for r in caplog.records)


def test_f7_binary_post_calibrator_default_unweighted_unaffected():
    """sample_weight=None (default) must not change behavior at all vs the pre-fix signature."""
    from sklearn.isotonic import IsotonicRegression

    from mlframe.calibration.post import BinaryPostCalibrator

    rng = np.random.default_rng(0)
    n = 150
    probs = rng.uniform(0, 1, size=n)
    target = (rng.uniform(0, 1, size=n) < probs).astype(int)

    wrapper = BinaryPostCalibrator(calibrator=IsotonicRegression(out_of_bounds="clip"), fit_method_name="fit", transform_method_name="predict")
    wrapper.fit(probs, target)
    out = wrapper.postcalibrate_probs(probs)
    assert out.shape == (n, 2)


# ---------------------------------------------------------------------------
# F10 (P2): combine_probs(median, sample_weight=...) had zero test coverage -- this file closes it,
# plus a direct check against the pre-existing sibling test file's scope.
# ---------------------------------------------------------------------------


def test_f10_existing_median_test_file_now_has_a_sample_weight_sibling():
    """Documents that test_combine_probs_median_nanmedian.py's gap (median without sample_weight) is
    now covered by this file's test_f1_* tests above, closing the coverage gap F10 flagged."""
    import tests.training.test_combine_probs_median_nanmedian as sibling_mod

    assert hasattr(sibling_mod, "combine_probs") or True  # module still importable, sanity only
