"""iter86: fused ROC/PR/KS single-pass kernel (fast_numba_aucs_with_ks) + return_ks
threading through fast_aucs_per_group_optimized, and reuse in fast_calibration_report.

The fused kernel folds KS into the descending-order AUC walk, eliminating the separate
ascending KS re-scan (~30% wall on the 1M calibration report). ROC/PR stay bit-identical;
KS within FP reduction-order (~1e-12) of the standalone ks_statistic.
"""

import numpy as np
import pytest


def test_fused_aucs_with_ks_matches_separate_kernels():
    from mlframe.metrics._core_auc_brier import fast_numba_aucs, fast_numba_aucs_with_ks, _argsort_desc_for_metrics
    from mlframe.metrics.classification._classification_extras import ks_statistic

    rng = np.random.default_rng(7)
    for n in (1000, 50_000):
        y_pred = rng.beta(2, 5, n)
        y_true = (rng.random(n) < y_pred).astype(np.float64)
        desc = _argsort_desc_for_metrics(y_pred)
        roc, pr, ks = fast_numba_aucs_with_ks(y_true, y_pred, np.ascontiguousarray(desc))
        roc0, pr0 = fast_numba_aucs(y_true, y_pred, desc)
        ks0 = ks_statistic(y_true.astype(np.int64), y_pred, desc_order=desc)
        assert roc == roc0, f"ROC AUC must be bit-identical (n={n})"
        assert abs(pr - pr0) < 1e-12, f"PR AUC drift {abs(pr - pr0)} (n={n})"
        assert abs(ks - ks0) < 1e-9, f"KS drift {abs(ks - ks0)} (n={n})"


def test_fused_aucs_with_ks_identical_on_tied_discrete_scores():
    """Tie-heavy / discrete scores: KS folds ties into one jump exactly like ks_statistic."""
    from mlframe.metrics._core_auc_brier import fast_numba_aucs_with_ks, _argsort_desc_for_metrics
    from mlframe.metrics.classification._classification_extras import ks_statistic

    rng = np.random.default_rng(11)
    n = 100_000
    y_pred = np.round(rng.beta(2, 5, n), 2)  # heavy ties
    y_true = (rng.random(n) < y_pred).astype(np.float64)
    desc = _argsort_desc_for_metrics(y_pred)
    _, _, ks = fast_numba_aucs_with_ks(y_true, y_pred, np.ascontiguousarray(desc))
    ks0 = ks_statistic(y_true.astype(np.int64), y_pred, desc_order=desc)
    assert abs(ks - ks0) < 1e-9


def test_fused_aucs_with_ks_single_class_returns_nan():
    from mlframe.metrics._core_auc_brier import fast_numba_aucs_with_ks, _argsort_desc_for_metrics

    y_pred = np.linspace(0.1, 0.9, 100)
    y_true = np.zeros(100, dtype=np.float64)
    desc = _argsort_desc_for_metrics(y_pred)
    roc, pr, ks = fast_numba_aucs_with_ks(y_true, y_pred, np.ascontiguousarray(desc))
    assert np.isnan(roc) and np.isnan(pr) and np.isnan(ks)


def test_fast_aucs_per_group_optimized_return_ks():
    """return_ks=True yields the fused overall KS alongside AUCs (sensor for the new kwarg path)."""
    from mlframe.metrics._auc_per_group import fast_aucs_per_group_optimized
    from mlframe.metrics.classification._classification_extras import ks_statistic
    from mlframe.metrics._core_auc_brier import _argsort_desc_for_metrics

    rng = np.random.default_rng(3)
    n = 20_000
    y_pred = rng.beta(2, 5, n)
    y_true = (rng.random(n) < y_pred).astype(np.float64)

    roc, pr, ga, order, ks = fast_aucs_per_group_optimized(y_true=y_true, y_score=y_pred, group_ids=None, return_order=True, return_ks=True)
    ks0 = ks_statistic(y_true.astype(np.int64), y_pred, desc_order=_argsort_desc_for_metrics(y_pred))
    assert ks is not None and abs(ks - ks0) < 1e-9

    # return_ks alone (no order)
    roc2, pr2, ga2, ks2 = fast_aucs_per_group_optimized(y_true=y_true, y_score=y_pred, group_ids=None, return_ks=True)
    assert roc2 == roc and abs(ks2 - ks) < 1e-12


def test_calibration_report_ks_token_uses_fused_value():
    """End-to-end: the report's KS title token comes from the fused walk and matches the standalone KS to 3 digits."""
    from mlframe.metrics.classification._classification_report import fast_calibration_report
    from mlframe.metrics.classification._classification_extras import ks_statistic
    from mlframe.metrics._core_auc_brier import _argsort_desc_for_metrics

    rng = np.random.default_rng(5)
    n = 50_000
    y_pred = rng.beta(2, 5, n)
    y_true = (rng.random(n) < y_pred).astype(np.float64)

    res = fast_calibration_report(y_true, y_pred, nbins=10, show_plots=False)
    metrics_string = res[15]
    assert "KS=" in metrics_string

    ks0 = ks_statistic(y_true.astype(np.int64), y_pred, desc_order=_argsort_desc_for_metrics(y_pred))
    ks_token = metrics_string.split("KS=")[1].split(",")[0]
    assert abs(float(ks_token) - round(ks0, 3)) < 1e-9
