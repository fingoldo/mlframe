"""Unit + biz_value tests for the model-agnostic conformal finalize core (`_conformal_finalize.py`).

Unit: structure inference, split-conformal vs CV+ dispatch + guarantee labels, coverage report shape.
biz_value: on a heteroscedastic target the normalized (locally-adaptive) score restores CONDITIONAL
coverage that the constant-width absolute score destroys, while both stay ~nominal marginally.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training._conformal_finalize import (
    conformal_regression_report,
    conformal_supports_split_guarantee,
    coverage_report,
    cv_plus_intervals,
    infer_split_structure,
    split_conformal_intervals,
)


def test_infer_split_structure_mapping():
    assert infer_split_structure() == "iid"
    assert infer_split_structure(time_column="ts") == "temporal"
    assert infer_split_structure(cv_strategy="purged") == "temporal"
    assert infer_split_structure(use_groups=True) == "grouped"
    assert infer_split_structure(wholeday_splitting=True) == "grouped"
    assert infer_split_structure(time_column="ts", use_groups=True) == "temporal_grouped"
    assert infer_split_structure(bucket_stratify=True) == "stratified"


def test_conformal_supports_split_guarantee_only_iid_and_stratified():
    assert conformal_supports_split_guarantee("iid")
    assert conformal_supports_split_guarantee("stratified")
    assert not conformal_supports_split_guarantee("temporal")
    assert not conformal_supports_split_guarantee("grouped")
    assert not conformal_supports_split_guarantee("temporal_grouped")


def test_split_conformal_absolute_marginal_coverage_iid():
    rng = np.random.default_rng(0)
    n = 4000
    res_cal = rng.standard_normal(n)  # homoscedastic
    y_pred_test = np.zeros(n)
    y_true_test = rng.standard_normal(n)
    iv = split_conformal_intervals(y_pred_test, res_cal, [0.1], score="absolute")
    rep = coverage_report(y_true_test, iv)
    assert 0.86 <= rep[0.1]["achieved_coverage"] <= 0.95


def test_cv_plus_label_and_validity():
    rng = np.random.default_rng(1)
    oof = rng.standard_normal(3000)
    out = conformal_regression_report(
        y_pred_test=np.zeros(3000),
        y_true_test=rng.standard_normal(3000),
        oof_residuals=oof,
        alphas=(0.1,),
    )
    assert out["method"] == "cv_plus"
    assert out["guarantee"] == "marginal>=1-2alpha"
    # CV+ symmetric band is valid (>= nominal) even if conservative.
    assert out["per_alpha"][0.1]["achieved_coverage"] >= 0.86


def test_report_split_path_labels_and_structure_flag():
    rng = np.random.default_rng(2)
    n = 2000
    out = conformal_regression_report(
        y_pred_test=np.zeros(n),
        y_true_test=rng.standard_normal(n),
        residuals_cal=rng.standard_normal(n),
        y_pred_cal=np.zeros(n),
        alphas=(0.1, 0.2),
        score="normalized",
        structure="temporal",
    )
    assert out["method"] == "split_conformal"
    assert out["guarantee"] == "marginal>=1-alpha"
    assert out["split_conformal_valid_for_structure"] is False  # temporal -> needs online variant
    assert set(out["per_alpha"]) == {0.1, 0.2}


def test_report_requires_some_calibration_source():
    with pytest.raises(ValueError):
        conformal_regression_report(
            y_pred_test=np.zeros(10),
            y_true_test=np.zeros(10),
            alphas=(0.1,),
        )


def _heteroscedastic(n: int, seed: int):
    """Oracle predictions with prediction-dependent noise scale.

    x in [0,1], pred = x (oracle mean), residual sd grows linearly with pred: sd = 0.2 + 2*pred.
    A constant-width band must over-cover at low pred and under-cover at high pred; a normalized
    band scaled by sigma_hat(pred) keeps per-bin coverage near nominal.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    pred = x.copy()
    sd = 0.2 + 2.0 * pred
    resid = sd * rng.standard_normal(n)
    return pred, resid


def _conditional_coverage_gap(y_pred_test, lo, hi, y_true_test, alpha, n_bins=5):
    """Max |per-bin coverage - (1-alpha)| over equal-count bins of the test prediction."""
    order = np.argsort(y_pred_test)
    inside = (y_true_test >= lo) & (y_true_test <= hi)
    gaps = []
    for chunk in np.array_split(order, n_bins):
        if chunk.size == 0:
            continue
        gaps.append(abs(float(np.mean(inside[chunk])) - (1.0 - alpha)))
    return max(gaps)


def test_biz_val_conformal_normalized_beats_absolute_conditional_coverage():
    """Normalized score's conditional-coverage gap is materially smaller than constant-width's.

    Measured (n=6000, seed avg, alpha=0.1): absolute conditional gap ~0.25-0.35 (badly over/under
    by prediction region), normalized ~0.05-0.10. Floor the win at: normalized_gap <= absolute_gap
    - 0.10 AND normalized_gap < 0.15, on the median of 3 seeds; both stay ~nominal marginally.
    """
    alpha = 0.1
    abs_gaps, norm_gaps, abs_cov, norm_cov = [], [], [], []
    for seed in range(3):
        pred_cal, res_cal = _heteroscedastic(6000, seed)
        pred_test, res_test = _heteroscedastic(6000, seed + 100)
        y_true_test = pred_test + res_test  # oracle mean + heteroscedastic noise

        iv_abs = split_conformal_intervals(pred_test, res_cal, [alpha], score="absolute")
        iv_norm = split_conformal_intervals(
            pred_test,
            res_cal,
            [alpha],
            score="normalized",
            y_pred_cal=pred_cal,
        )
        lo_a, hi_a = iv_abs[alpha]
        lo_n, hi_n = iv_norm[alpha]
        abs_gaps.append(_conditional_coverage_gap(pred_test, lo_a, hi_a, y_true_test, alpha))
        norm_gaps.append(_conditional_coverage_gap(pred_test, lo_n, hi_n, y_true_test, alpha))
        abs_cov.append(float(np.mean((y_true_test >= lo_a) & (y_true_test <= hi_a))))
        norm_cov.append(float(np.mean((y_true_test >= lo_n) & (y_true_test <= hi_n))))

    abs_gap = float(np.median(abs_gaps))
    norm_gap = float(np.median(norm_gaps))
    assert norm_gap < 0.15, (norm_gap, norm_gaps)
    assert norm_gap <= abs_gap - 0.10, (norm_gap, abs_gap)
    # Both ~nominal marginally (conformal guarantees this; sanity floor).
    assert 0.85 <= float(np.median(abs_cov)) <= 0.95
    assert 0.85 <= float(np.median(norm_cov)) <= 0.95
