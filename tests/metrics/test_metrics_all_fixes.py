"""Regression tests for audits/full_audit_2026-07-21/metrics_all.md's findings (F1-F11).

sklearn's actual r2_score/explained_variance_score convention for a degenerate (zero-variance) y_true was
verified directly (not assumed) via a standalone script before any fix landed: 1.0 for a perfect fit, 0.0 for
an imperfect one -- never -inf/NaN. Every R2-family fix below is checked against that verified convention.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------------------------------------------
# F1 -- fast_regression_metrics_block returned R2=-inf instead of sklearn's 0.0/1.0
# ---------------------------------------------------------------------------------------------------------------


def test_f1_fast_regression_metrics_block_constant_y_true_perfect_fit():
    """F1: constant y_true, perfect fit -> R2=1.0 (verified against real sklearn.metrics.r2_score)."""
    from mlframe.metrics.regression._regression_metrics import fast_regression_metrics_block

    y = np.full(100, 5.0)
    block = fast_regression_metrics_block(y, y.copy())
    assert block["R2"] == 1.0


def test_f1_fast_regression_metrics_block_constant_y_true_imperfect_fit():
    """F1: constant y_true, nonzero residual -> R2=0.0, not -inf."""
    from mlframe.metrics.regression._regression_metrics import fast_regression_metrics_block

    y = np.full(100, 5.0)
    p = y.copy()
    p[0] = 6.0
    block = fast_regression_metrics_block(y, p)
    assert block["R2"] == 0.0
    assert np.isfinite(block["R2"])


# ---------------------------------------------------------------------------------------------------------------
# F2 -- fast_regression_metrics_block_extended duplicated the -inf defect, also poisoned NSE/EV
# ---------------------------------------------------------------------------------------------------------------


def test_f2_fast_regression_metrics_block_extended_constant_y_true():
    """F2: R2/NSE/ExplainedVariance must all follow the sklearn convention, not -inf/NaN."""
    from mlframe.metrics.regression._regression_extras import fast_regression_metrics_block_extended

    y = np.full(100, 5.0)
    perfect = fast_regression_metrics_block_extended(y, y.copy())
    assert perfect["R2"] == 1.0
    assert perfect["NSE"] == 1.0
    assert perfect["ExplainedVariance"] == 1.0

    p = y.copy()
    p[0] = 6.0
    imperfect = fast_regression_metrics_block_extended(y, p)
    assert imperfect["R2"] == 0.0
    assert imperfect["NSE"] == 0.0
    assert imperfect["ExplainedVariance"] == 0.0
    assert all(np.isfinite(v) for v in (imperfect["R2"], imperfect["NSE"], imperfect["ExplainedVariance"]))


# ---------------------------------------------------------------------------------------------------------------
# F3 -- three R2-family kernels disagreed with each other and with sklearn on ss_tot==0
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("weighted", [False, True])
def test_f3_fast_r2_score_constant_y_true_matches_sklearn(weighted):
    """F3: fast_r2_score (the public, multioutput-aware entry point) must match sklearn's real convention."""
    from mlframe.metrics.core import fast_r2_score

    y = np.full(60, 3.0)
    kwargs = {"sample_weight": np.ones(60)} if weighted else {}
    assert fast_r2_score(y, y.copy(), **kwargs) == 1.0
    p = y.copy()
    p[0] = 4.0
    assert fast_r2_score(y, p, **kwargs) == 0.0


def test_f3_fast_nash_sutcliffe_constant_y_true_matches_sklearn_convention():
    """F3: fast_nash_sutcliffe (previously returned NaN here) must now agree with the other R2 kernels."""
    from mlframe.metrics.regression._regression_extras import fast_nash_sutcliffe

    y = np.full(50, 3.0)
    assert fast_nash_sutcliffe(y, y.copy()) == 1.0
    p = y.copy()
    p[0] = 4.0
    assert fast_nash_sutcliffe(y, p) == 0.0


def test_f3_all_r2_family_kernels_agree_on_degenerate_input():
    """F3: fast_r2_score, fast_regression_metrics_block, fast_regression_metrics_block_extended, and
    fast_nash_sutcliffe must now all return the SAME value for the same degenerate input (previously three
    different conventions: 0.0-always / 0.0-or--inf / NaN)."""
    from mlframe.metrics.core import fast_r2_score
    from mlframe.metrics.regression._regression_extras import fast_nash_sutcliffe, fast_regression_metrics_block_extended
    from mlframe.metrics.regression._regression_metrics import fast_regression_metrics_block

    y = np.full(40, 7.0)
    p = y.copy()
    p[3] = 9.0
    results = {
        "fast_r2_score": fast_r2_score(y, p),
        "block_R2": fast_regression_metrics_block(y, p)["R2"],
        "block_extended_R2": fast_regression_metrics_block_extended(y, p)["R2"],
        "nash_sutcliffe": fast_nash_sutcliffe(y, p),
    }
    assert len(set(results.values())) == 1, f"R2-family kernels disagree: {results}"
    assert next(iter(results.values())) == 0.0


# ---------------------------------------------------------------------------------------------------------------
# F4 -- fast_multiclass_confusion_metrics_block's macro averages deflated by phantom absent classes
# ---------------------------------------------------------------------------------------------------------------


def test_f4_multiclass_confusion_block_macro_metrics_exclude_absent_classes():
    """F4: n_classes declared larger than the classes actually present must not deflate macro P/R/F1."""
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

    from mlframe.metrics.classification._classification_extras_blocks import fast_multiclass_confusion_metrics_block

    y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0, 1, 2])
    block = fast_multiclass_confusion_metrics_block(y_true, y_pred, n_classes=10)  # 10 declared, only 3 present

    present_labels = [0, 1, 2]
    assert block["macro_precision"] == pytest.approx(precision_score(y_true, y_pred, labels=present_labels, average="macro"), abs=1e-12)
    assert block["macro_recall"] == pytest.approx(recall_score(y_true, y_pred, labels=present_labels, average="macro"), abs=1e-12)
    assert block["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, labels=present_labels, average="macro"), abs=1e-12)
    assert block["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred), abs=1e-12)


# ---------------------------------------------------------------------------------------------------------------
# F5 / F6 -- missing length validation on fast_roc_auc_unstable / fast_roc_auc's sample_weight
# ---------------------------------------------------------------------------------------------------------------


def test_f5_fast_roc_auc_unstable_raises_on_length_mismatch():
    """F5: a mismatched y_true/y_score length must raise ValueError, not read out of bounds."""
    from mlframe.metrics.core import fast_roc_auc_unstable

    with pytest.raises(ValueError):
        fast_roc_auc_unstable(np.array([0, 1, 1]), np.array([0.1, 0.2]))


def test_f6_fast_roc_auc_raises_on_sample_weight_length_mismatch():
    """F6: a mismatched sample_weight length must raise ValueError, not read out of bounds."""
    from mlframe.metrics.core import fast_roc_auc

    with pytest.raises(ValueError):
        fast_roc_auc(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.3, 0.2]), sample_weight=np.array([1.0, 1.0]))


# ---------------------------------------------------------------------------------------------------------------
# F7 -- weighted_kappa never validated labels fit inside an explicit n_classes
# ---------------------------------------------------------------------------------------------------------------


def test_f7_weighted_kappa_raises_on_out_of_range_labels_with_explicit_n_classes():
    """F7: an out-of-range label under a too-small explicit n_classes must raise, not write out of bounds."""
    from mlframe.metrics.classification._weighted_kappa import weighted_kappa

    with pytest.raises(ValueError, match="n_classes"):
        weighted_kappa(np.array([0, 1, 5]), np.array([0, 1, 2]), n_classes=3)


def test_f7_weighted_kappa_still_works_with_correct_n_classes():
    """F7: a correctly-sized explicit n_classes must still work (no false-positive rejection)."""
    from mlframe.metrics.classification._weighted_kappa import weighted_kappa

    result = weighted_kappa(np.array([0, 1, 2, 1]), np.array([0, 1, 2, 0]), n_classes=3)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------------------------------------------
# F8 -- _batch_per_class_ice_kernel's min/max scan seeded from the wrong sentinel
# ---------------------------------------------------------------------------------------------------------------


def test_f8_ice_kernel_minmax_seeding_bug_reproduction_and_fix():
    """F8: the old [1.0, 0.0] sentinel seeding never updated min_val for an all-out-of-[0,1] class (every
    value > 1.0 never satisfies `v < 1.0`), silently computing a wrong span; seeding from the first sample
    (this fix) correctly detects the true span, including the degenerate zero-span (constant) case."""

    def scan_minmax(seed_from_data: bool, y_p: np.ndarray):
        """Reproduces the kernel's min/max seeding step in pure Python for comparison."""
        if seed_from_data:
            min_val = max_val = y_p[0]
        else:
            min_val, max_val = 1.0, 0.0
        for v in y_p:
            if v > max_val:
                max_val = v
            if v < min_val:
                min_val = v
        return min_val, max_val

    y_p = np.array([5.0, 5.0, 5.0, 5.0])  # constant, entirely outside [0, 1]
    old_min, old_max = scan_minmax(False, y_p)
    new_min, new_max = scan_minmax(True, y_p)
    assert old_max - old_min != 0.0, "bug-reproduction sanity: old seeding must NOT detect the true zero span"
    assert new_max - new_min == 0.0, "fix must correctly detect the true (zero) span for constant input"


def test_f8_ice_kernel_handles_out_of_range_predictions_without_crashing():
    """F8: the real njit kernel must produce a finite result for out-of-[0,1] predictions (integration smoke test)."""
    from mlframe.metrics.classification._classification_report import _batch_per_class_ice_kernel

    N = 200
    y_true = np.zeros((N, 1), dtype=np.int8)
    y_true[:100, 0] = 1
    y_pred = np.full((N, 1), 5.0, dtype=np.float64)
    desc_idx = np.argsort(-y_pred, axis=0)

    ice = _batch_per_class_ice_kernel(
        y_true, y_pred, desc_idx, nbins=10, use_weights=True,
        mae_weight=3.0, std_weight=2.0, brier_loss_weight=0.8,
        roc_auc_weight=1.5, pr_auc_weight=0.1, min_roc_auc=0.54, roc_auc_penalty=0.0,
    )
    assert np.isfinite(ice).all()


# ---------------------------------------------------------------------------------------------------------------
# F9 -- _coerce_multilabel_array silently mis-cast NaN / out-of-range values
# ---------------------------------------------------------------------------------------------------------------


def test_f9_multilabel_hamming_loss_raises_on_nan():
    """F9: a NaN in a multilabel indicator column must raise, not silently cast to an in-range code."""
    from mlframe.metrics._multilabel_metrics import hamming_loss

    with pytest.raises(ValueError):
        hamming_loss(np.array([1.0, 0.0, np.nan]), np.array([1.0, 0.0, 1.0]))


def test_f9_multilabel_hamming_loss_raises_on_out_of_range_value():
    """F9: an out-of-{0,1} value must raise, not silently truncate into a wrong 0/1 code."""
    from mlframe.metrics._multilabel_metrics import hamming_loss

    with pytest.raises(ValueError):
        hamming_loss(np.array([1.0, 0.0, 2.0]), np.array([1.0, 0.0, 1.0]))


def test_f9_multilabel_hamming_loss_still_works_on_valid_input():
    """F9: valid 0/1 input must still compute correctly (no false-positive rejection)."""
    from mlframe.metrics._multilabel_metrics import hamming_loss

    assert hamming_loss(np.array([1, 0, 1]), np.array([1, 0, 0])) == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------------------------------------------
# F10 -- rmsle_loss clipped y_pred but never validated/clipped y_true
# ---------------------------------------------------------------------------------------------------------------


def test_f10_rmsle_loss_negative_y_true_is_clipped_not_nan():
    """F10: a negative y_true must be clipped (matching the y_pred convention), not silently produce NaN."""
    from mlframe.metrics.scoring import rmsle_loss

    val = rmsle_loss(np.array([-1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert np.isfinite(val)


def test_f10_rmsle_loss_warns_on_negative_y_true():
    """F10: clipping a negative y_true must emit a visible warning (mirrors fast_rmsle's own convention)."""
    from mlframe.metrics.scoring import rmsle_loss

    with pytest.warns(RuntimeWarning, match="negative"):
        rmsle_loss(np.array([-1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------------------------------------------
# F11 -- fast_concordance_index crashed with AttributeError on a plain Python list
# ---------------------------------------------------------------------------------------------------------------


def test_f11_fast_concordance_index_accepts_plain_python_lists():
    """F11: a plain Python list (a legal input to every sibling function) must work, not AttributeError."""
    from mlframe.metrics.regression._regression_corr import fast_concordance_index

    result = fast_concordance_index([1, 2, 3, 4, 5], [1, 3, 2, 5, 4])
    assert np.isfinite(result)


def test_f11_fast_concordance_index_raises_clean_error_on_length_mismatch():
    """F11: a mismatched-length list pair must raise a clean ValueError, not a confusing internal error."""
    from mlframe.metrics.regression._regression_corr import fast_concordance_index

    with pytest.raises(ValueError):
        fast_concordance_index([1, 2, 3], [1, 2])
