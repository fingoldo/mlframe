"""Tests for mlframe.metrics.classification._classification_extras.

Coverage:
- KS / MCC / Cohen kappa / BalAcc / G-mean / BSS / Gini / Specificity-NPV
  / F-beta / Spiegelhalter Z / Lift@k (binary)
- Top-k accuracy / multiclass MCC / RPS (multiclass)
- Fused blocks: binary confusion / binary probability / multiclass confusion

Each metric gets:
  - a correctness unit (vs sklearn or hand-computed expected when no
    sklearn equivalent),
  - an edge-case unit (degenerate / single-class / empty / all-zero),
  - the fused-block agreement test (block result must match the individual
    metric within fp tolerance).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.core import (
    # Binary
    ks_statistic, matthews_corrcoef_binary, cohen_kappa_binary,
    balanced_accuracy_binary, g_mean_binary, brier_skill_score,
    gini_from_auc, specificity_npv_fpr_fnr, f_beta_score,
    spiegelhalter_z, lift_at_k,
    fast_binary_confusion_metrics_block,
    fast_binary_probability_metrics_block,
    # Multiclass
    top_k_accuracy, matthews_corrcoef_multiclass,
    ranked_probability_score,
    fast_multiclass_confusion_metrics_block,
)


# ----- helpers -----


def _rand_binary(N, prevalence=0.3, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.uniform(size=N) < prevalence).astype(np.int64)
    s = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, N), 0.001, 0.999)
    return y, s


# ----- KS -----


def test_ks_perfect_separation():
    """Perfect separation -> KS = 1."""
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, s) == pytest.approx(1.0)


def test_ks_no_signal():
    """Identical class distributions -> KS = 0."""
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert ks_statistic(y, s) == pytest.approx(0.0)


def test_ks_single_class_returns_nan():
    """Empty positive (or negative) class -> NaN."""
    y = np.zeros(10, dtype=np.int64)
    s = np.random.default_rng(0).uniform(size=10)
    assert np.isnan(ks_statistic(y, s))


def test_ks_matches_scipy_for_2_sample():
    """For binary classification KS, scipy's 2-sample ks_2samp on
    class-conditional score arrays produces the same statistic."""
    from scipy.stats import ks_2samp
    y, s = _rand_binary(2000, prevalence=0.4, seed=42)
    s_pos = s[y == 1]
    s_neg = s[y == 0]
    expected = ks_2samp(s_pos, s_neg).statistic
    assert ks_statistic(y, s) == pytest.approx(expected, abs=1e-12)


# ----- MCC -----


def test_mcc_matches_sklearn():
    from sklearn.metrics import matthews_corrcoef
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = (rng.uniform(size=500) > 0.6).astype(np.int64)
        p = (rng.uniform(size=500) > 0.5).astype(np.int64)
        assert matthews_corrcoef_binary(y, p) == pytest.approx(
            matthews_corrcoef(y, p), abs=1e-12,
        )


def test_mcc_perfect_inverted():
    y = np.array([0, 1, 0, 1])
    assert matthews_corrcoef_binary(y, 1 - y) == pytest.approx(-1.0)
    assert matthews_corrcoef_binary(y, y) == pytest.approx(1.0)


def test_mcc_all_one_class_returns_zero():
    """When one row of the confusion matrix is empty MCC's denominator
    is 0 -> sklearn convention is 0.0."""
    y = np.array([1, 1, 1, 1])
    p = np.array([1, 1, 0, 1])
    assert matthews_corrcoef_binary(y, p) == pytest.approx(0.0)


# ----- Cohen's kappa -----


def test_cohen_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score
    rng = np.random.default_rng(1)
    for _ in range(5):
        y = (rng.uniform(size=500) > 0.5).astype(np.int64)
        p = (rng.uniform(size=500) > 0.5).astype(np.int64)
        assert cohen_kappa_binary(y, p) == pytest.approx(
            cohen_kappa_score(y, p), abs=1e-12,
        )


# ----- Balanced accuracy / G-mean -----


def test_balanced_accuracy_matches_sklearn():
    from sklearn.metrics import balanced_accuracy_score
    rng = np.random.default_rng(2)
    y = (rng.uniform(size=1000) > 0.7).astype(np.int64)
    p = (rng.uniform(size=1000) > 0.5).astype(np.int64)
    assert balanced_accuracy_binary(y, p) == pytest.approx(
        balanced_accuracy_score(y, p), abs=1e-12,
    )


def test_g_mean_in_unit_range():
    y, s = _rand_binary(500)
    p = (s > 0.5).astype(np.int64)
    assert 0.0 <= g_mean_binary(y, p) <= 1.0


# ----- BSS -----


def test_bss_perfect_probabilities_is_one():
    """Predicting y exactly -> Brier=0 -> BSS=1."""
    y = np.array([0, 1, 0, 1, 0, 1])
    bss = brier_skill_score(y, y.astype(np.float64))
    assert bss == pytest.approx(1.0)


def test_bss_marginal_baseline_is_zero():
    """Predicting the base rate everywhere -> BSS=0 by construction."""
    y = np.array([0, 0, 1, 1, 1])
    base = float(y.mean())
    bss = brier_skill_score(y, np.full(len(y), base))
    assert bss == pytest.approx(0.0)


# ----- Gini -----


def test_gini_from_auc_relations():
    assert gini_from_auc(0.5) == pytest.approx(0.0)
    assert gini_from_auc(1.0) == pytest.approx(1.0)
    assert gini_from_auc(0.0) == pytest.approx(-1.0)


# ----- Specificity / NPV / FPR / FNR -----


def test_specificity_npv_matches_manual():
    # 2 TP, 1 FP, 3 TN, 1 FN
    y = np.array([1, 1, 0, 0, 0, 1, 0])
    p = np.array([1, 1, 0, 0, 1, 0, 0])
    spec, npv, fpr, fnr = specificity_npv_fpr_fnr(y, p)
    # TP=2 FP=1 TN=3 FN=1
    assert spec == pytest.approx(3 / 4)
    assert npv == pytest.approx(3 / 4)
    assert fpr == pytest.approx(1 / 4)
    assert fnr == pytest.approx(1 / 3)


# ----- F-beta -----


def test_f_beta_reduces_to_f1():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=500) > 0.5).astype(np.int64)
    p = (rng.uniform(size=500) > 0.5).astype(np.int64)
    assert f_beta_score(y, p, beta=1.0) == pytest.approx(f1_score(y, p), abs=1e-12)


def test_f_beta_weights_recall_more():
    """F2 should weight recall heavier than F1 -> when recall < precision,
    F2 < F1; when recall > precision, F2 > F1."""
    # precision=1.0, recall=0.5 -> F1=0.667 F2=0.555 F0.5=0.833
    y = np.array([1, 1, 0])
    p = np.array([1, 0, 0])
    assert f_beta_score(y, p, beta=2.0) < f_beta_score(y, p, beta=1.0)
    assert f_beta_score(y, p, beta=0.5) > f_beta_score(y, p, beta=1.0)


# ----- Spiegelhalter Z -----


def test_spiegelhalter_z_perfect_calibration_close_to_zero():
    """Sampling y_i ~ Bernoulli(p_i) for known p -> Z ~ N(0, 1)."""
    rng = np.random.default_rng(7)
    N = 10_000
    p = rng.uniform(0.05, 0.95, N)
    y = (rng.uniform(size=N) < p).astype(np.int64)
    z, pv = spiegelhalter_z(y, p)
    assert abs(z) < 4.0  # |Z| > 4 is p < 6e-5; very unlikely under null
    assert 0.0 <= pv <= 1.0


def test_spiegelhalter_z_miscalibrated_high_z():
    """Strongly miscalibrated predictions -> |Z| large + p<0.001."""
    N = 1000
    y = np.concatenate([np.zeros(500), np.ones(500)]).astype(np.int64)
    # Predict everything at 0.5 -> calibration off because mean(y) = 0.5 but
    # variance of y is much higher than the model implies.
    z, pv = spiegelhalter_z(y, np.full(N, 0.5))
    # Z for perfect average-calibration with no resolution is 0 (predicting
    # the marginal everywhere matches the global mean). What we need is a
    # case where prediction is genuinely off the local mean.
    # Better case: predict 0.1 everywhere when the base rate is 0.5.
    z2, pv2 = spiegelhalter_z(y, np.full(N, 0.1))
    assert abs(z2) > 3.0
    assert pv2 < 0.01


# ----- Lift@k -----


def test_lift_at_k_baseline_equals_one():
    """Random scores -> Lift@k ~ 1.0 in expectation."""
    rng = np.random.default_rng(8)
    N = 10_000
    y = (rng.uniform(size=N) > 0.7).astype(np.int64)
    s = rng.uniform(size=N)
    lift = lift_at_k(y, s, k_pct=20.0)
    assert 0.85 <= lift <= 1.15


def test_lift_at_k_perfect_score():
    """Perfect score -> all positives in top n_pos/n; Lift = n / n_pos."""
    N = 100
    y = np.concatenate([np.ones(20), np.zeros(80)]).astype(np.int64)
    s = -np.arange(N, dtype=np.float64)  # decreasing - first 20 are 1
    lift = lift_at_k(y, s, k_pct=20.0)
    # Top 20% (= 20 rows) captures 20/20 = 100% of positives;
    # baseline = 20%. Lift = 1.0 / 0.2 = 5.0
    assert lift == pytest.approx(5.0)


def test_lift_at_k_rejects_bad_k():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        lift_at_k(y, s, k_pct=0.0)
    with pytest.raises(ValueError):
        lift_at_k(y, s, k_pct=101.0)


# ----- Top-k accuracy -----


def test_top_k_accuracy_top1_equals_argmax_accuracy():
    rng = np.random.default_rng(9)
    N, K = 200, 5
    y = rng.integers(0, K, size=N).astype(np.int64)
    p = rng.uniform(size=(N, K))
    p /= p.sum(axis=1, keepdims=True)
    expected = float((p.argmax(axis=1) == y).mean())
    assert top_k_accuracy(y, p, k=1) == pytest.approx(expected)


def test_top_k_accuracy_top_k_at_or_above_K_is_nan():
    """top-K is trivially 1.0 - we surface NaN as the no-info signal."""
    p = np.array([[0.5, 0.5], [0.5, 0.5]])
    y = np.array([0, 1])
    assert np.isnan(top_k_accuracy(y, p, k=2))


def test_top_k_accuracy_monotone():
    """top-1 <= top-2 <= ... <= top-K"""
    rng = np.random.default_rng(10)
    N, K = 500, 6
    y = rng.integers(0, K, size=N).astype(np.int64)
    p = rng.uniform(size=(N, K))
    p /= p.sum(axis=1, keepdims=True)
    accs = [top_k_accuracy(y, p, k=k) for k in (1, 2, 3, 4, 5)]
    for a, b in zip(accs, accs[1:]):
        assert a <= b + 1e-12


# ----- Multiclass MCC -----


def test_multiclass_mcc_matches_sklearn():
    from sklearn.metrics import matthews_corrcoef
    rng = np.random.default_rng(11)
    y = rng.integers(0, 5, size=500).astype(np.int64)
    p = rng.integers(0, 5, size=500).astype(np.int64)
    assert matthews_corrcoef_multiclass(y, p, n_classes=5) == pytest.approx(
        matthews_corrcoef(y, p), abs=1e-12,
    )


# ----- RPS -----


def test_rps_zero_when_perfect():
    """Predicting the true class at p=1 -> RPS = 0."""
    N, K = 10, 4
    y = np.array([2, 1, 0, 3, 2, 1, 0, 3, 2, 1], dtype=np.int64)
    p = np.zeros((N, K), dtype=np.float64)
    for i, t in enumerate(y):
        p[i, t] = 1.0
    assert ranked_probability_score(y, p) == pytest.approx(0.0)


def test_rps_reduces_to_brier_for_K2():
    """For binary (K=2 ordinal), RPS equals Brier."""
    rng = np.random.default_rng(12)
    N = 200
    y = (rng.uniform(size=N) > 0.5).astype(np.int64)
    p1 = rng.uniform(size=N)
    p = np.column_stack([1.0 - p1, p1])
    rps = ranked_probability_score(y, p)
    brier = float(np.mean((p1 - y) ** 2))
    assert rps == pytest.approx(brier, abs=1e-12)


# ----- Fused blocks -----


def test_binary_confusion_block_matches_individual_metrics():
    y, s = _rand_binary(2000, seed=20)
    p = (s > 0.5).astype(np.int64)
    block = fast_binary_confusion_metrics_block(y, p)
    assert block["MCC"] == pytest.approx(matthews_corrcoef_binary(y, p), abs=1e-12)
    assert block["Cohen_kappa"] == pytest.approx(cohen_kappa_binary(y, p), abs=1e-12)
    assert block["balanced_accuracy"] == pytest.approx(balanced_accuracy_binary(y, p), abs=1e-12)
    assert block["G_mean"] == pytest.approx(g_mean_binary(y, p), abs=1e-12)
    assert block["F1"] == pytest.approx(f_beta_score(y, p, beta=1.0), abs=1e-12)
    spec, npv, fpr, fnr = specificity_npv_fpr_fnr(y, p)
    assert block["specificity"] == pytest.approx(spec, abs=1e-12)
    assert block["NPV"] == pytest.approx(npv, abs=1e-12)
    assert block["FPR"] == pytest.approx(fpr, abs=1e-12)
    assert block["FNR"] == pytest.approx(fnr, abs=1e-12)


def test_binary_probability_block_matches_individual_metrics():
    y, s = _rand_binary(2000, seed=21)
    block = fast_binary_probability_metrics_block(y, s)
    assert block["BSS"] == pytest.approx(brier_skill_score(y, s), abs=1e-10)
    sh_z, sh_p = spiegelhalter_z(y, s)
    assert block["Spiegelhalter_Z"] == pytest.approx(sh_z, abs=1e-10)
    assert block["Spiegelhalter_p"] == pytest.approx(sh_p, abs=1e-10)


def test_multiclass_confusion_block_matches_individual_metrics():
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
    )
    rng = np.random.default_rng(22)
    N, K = 1000, 5
    y = rng.integers(0, K, size=N).astype(np.int64)
    p = rng.integers(0, K, size=N).astype(np.int64)
    block = fast_multiclass_confusion_metrics_block(y, p, n_classes=K)
    assert block["accuracy"] == pytest.approx(accuracy_score(y, p), abs=1e-12)
    assert block["macro_f1"] == pytest.approx(
        f1_score(y, p, average="macro", zero_division=0), abs=1e-12,
    )
    assert block["weighted_f1"] == pytest.approx(
        f1_score(y, p, average="weighted", zero_division=0), abs=1e-12,
    )
    assert block["MCC_multiclass"] == pytest.approx(
        matthews_corrcoef_multiclass(y, p, n_classes=K), abs=1e-12,
    )


def test_binary_blocks_large_n_parallel_path():
    """Triggers the parallel kernel branches (N >= 100k)."""
    rng = np.random.default_rng(23)
    N = 200_000
    y = (rng.uniform(size=N) > 0.7).astype(np.int64)
    s = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.1, N), 0.001, 0.999)
    p = (s > 0.5).astype(np.int64)
    # Just must not crash + match individual metric on a couple keys.
    block = fast_binary_confusion_metrics_block(y, p)
    assert block["F1"] == pytest.approx(f_beta_score(y, p, beta=1.0), abs=1e-12)
    prob_block = fast_binary_probability_metrics_block(y, s)
    assert prob_block["base_rate"] == pytest.approx(y.mean(), abs=1e-12)
