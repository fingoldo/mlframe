"""The sample-weighted ICE: unit weights reproduce the unweighted metric, real weights change it the way they should."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from mlframe.metrics.classification._ice_kernel import _batch_per_class_ice_kernel_serial
from mlframe.metrics.classification._ice_kernel_weighted import _weighted_class_ice, batch_per_class_ice_weighted, bin_index
from mlframe.metrics.core import compute_probabilistic_multiclass_error

_ARGS = (10, True, 3.0, 2.0, 0.8, 1.5, 0.1, 0.54, 0.0, 0.0)


@pytest.mark.parametrize("n", [50, 1000, 20000])
def test_unit_weights_equal_the_unweighted_kernel(n):
    rng = np.random.default_rng(n)
    p = np.round(rng.beta(2, 5, size=(n, 3)), 2)  # rounded: heavy ties exercise the tie-run walk
    y = (rng.random((n, 3)) < p).astype(np.int8)
    d = np.ascontiguousarray(np.argsort(-p, axis=0).astype(np.int64))
    w = np.ones(n)
    b = np.column_stack([bin_index(y[:, k], p[:, k], w, 10, "uniform") for k in range(3)])
    np.testing.assert_array_equal(_batch_per_class_ice_kernel_serial(y, p, d, *_ARGS), batch_per_class_ice_weighted(y, p, d, b, w, *_ARGS))


def test_weighted_auc_terms_match_sklearn():
    rng = np.random.default_rng(1)
    n = 3000
    p = rng.random(n)
    y = (rng.random(n) < p).astype(np.int8)
    w = rng.exponential(size=n)
    d = np.argsort(-p).astype(np.int64)
    b = bin_index(y, p, w, 10, "uniform")
    only_roc = _weighted_class_ice(y, p, w, d, b, 10, True, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0)
    only_pr = _weighted_class_ice(y, p, w, d, b, 10, True, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0)
    assert abs(only_roc) + 0.5 == pytest.approx(roc_auc_score(y, p, sample_weight=w), abs=1e-12)
    assert -only_pr == pytest.approx(average_precision_score(y, p, sample_weight=w), abs=1e-12)


def test_integer_weights_equal_row_duplication():
    """Weight 3 on a row must score like three copies of it (the definition sample weights have to satisfy)."""
    rng = np.random.default_rng(2)
    n = 400
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    w = rng.integers(1, 4, size=n)
    weighted = compute_probabilistic_multiclass_error(y_true=y, y_score=p, sample_weight=w.astype(float))
    duplicated = compute_probabilistic_multiclass_error(y_true=np.repeat(y, w), y_score=np.repeat(p, w))
    assert weighted == pytest.approx(duplicated, rel=1e-9)


def test_equal_weights_leave_the_metric_unchanged_and_bad_weights_raise():
    rng = np.random.default_rng(3)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    base = compute_probabilistic_multiclass_error(y_true=y, y_score=p)
    assert compute_probabilistic_multiclass_error(y_true=y, y_score=p, sample_weight=np.full(500, 7.0)) == base
    bad = np.ones(500)
    bad[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        compute_probabilistic_multiclass_error(y_true=y, y_score=p, sample_weight=bad)


def test_weights_follow_their_rows_through_the_subgroup_and_time_split_wrappers():
    from mlframe.training._picklable_metrics import IntegralCalibrationError, RobustTimeSplitMetric

    ice = IntegralCalibrationError(method="multicrit", mae_weight=3, std_weight=2, brier_loss_weight=0.8, roc_auc_weight=1.5, pr_auc_weight=0.1,
                                   min_roc_auc=0.54, roc_auc_penalty=0.0, use_weighted_calibration=True, weight_by_class_npositives=False, nbins=10)
    rng = np.random.default_rng(4)
    n = 1200
    p = rng.random(n)
    y = (rng.random(n) < p).astype(int)
    w = rng.integers(1, 4, size=n).astype(float)
    wrapped = RobustTimeSplitMetric(ice, num_splits=3, std_coeff=0.5, greater_is_better=False, min_samples_per_split=100)
    splits = [slice(0, 400), slice(400, 800), slice(800, n)]
    vals = [ice(y[s], p[s], sample_weight=w[s]) for s in splits]
    assert wrapped(y, p, sample_weight=w) == pytest.approx(float(np.mean(vals)) + float(np.std(vals)) * 0.5, rel=1e-12)
