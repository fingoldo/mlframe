"""Cluster 17 regressions: tests for kernels that had no test naming them (audit 2026-11-20, findings 17-F19 and 17-F20).

The parallel RMSLE branch, the quantile-binning kernel, the general-power Tweedie kernel, the multiclass MCC, the BSS-from-Brier shortcut and the
log-cosh loss are each pinned to an independent reference.
"""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, matthews_corrcoef, mean_tweedie_deviance

from mlframe.metrics.calibration._calibration_plot import _quantile_binning_kernel
from mlframe.metrics.classification._classification_extras import brier_skill_score_from_brier, matthews_corrcoef_multiclass
from mlframe.metrics.regression._regression_benchmark import fast_logcosh_loss
from mlframe.metrics.regression._regression_deviance import fast_tweedie_deviance
from mlframe.metrics.regression._regression_extras import _rmsle_kernel_par, _rmsle_kernel_seq


def test_parallel_rmsle_matches_the_sequential_kernel_including_negative_counts():
    """On 50k rows with some negative values the parallel kernel returns the sequential value and the same negative-row count."""
    rng = np.random.default_rng(0)
    yt = rng.uniform(-0.1, 5, 50_000)
    yp = rng.uniform(-0.1, 5, 50_000)
    seq = _rmsle_kernel_seq(yt, yp)
    par = _rmsle_kernel_par(yt, yp, 4)
    assert par[1] == seq[1] > 0
    assert par[0] == pytest.approx(seq[0], rel=1e-12)


def test_quantile_binning_kernel_assigns_rows_by_edges_and_sends_nan_to_the_first_pocket():
    """Pockets by the edges [0, .5, 1]: counts and observed frequencies match a hand count; a NaN score lands in pocket 0."""
    y_pred = np.array([0.1, 0.2, 0.7, 0.9, np.nan])
    y_true = np.array([0, 1, 1, 1, 0], dtype=np.int64)
    edges = np.array([0.0, 0.5, 1.0])
    _freqs_pred, freqs_true, hits = _quantile_binning_kernel(y_true, y_pred, edges)
    assert hits.tolist() == [3, 2]
    assert freqs_true.tolist() == pytest.approx([1 / 3, 1.0])


@pytest.mark.parametrize("power", [1.5, 3.0])
def test_general_tweedie_kernel_matches_sklearn_for_power_between_1_and_2_and_above_2(power):
    """fast_tweedie_deviance at a non-special power equals sklearn's mean_tweedie_deviance."""
    rng = np.random.default_rng(1)
    yt = rng.uniform(0.1, 5, 2000)
    yp = rng.uniform(0.1, 5, 2000)
    assert fast_tweedie_deviance(yt, yp, power=power) == pytest.approx(mean_tweedie_deviance(yt, yp, power=power), rel=1e-9)


def test_multiclass_mcc_matches_sklearn():
    """Gorodkin multiclass MCC equals sklearn's matthews_corrcoef on a random 4-class problem."""
    rng = np.random.default_rng(2)
    yt = rng.integers(0, 4, 500)
    yp = np.where(rng.uniform(size=500) < 0.6, yt, rng.integers(0, 4, 500))
    assert matthews_corrcoef_multiclass(yt, yp) == pytest.approx(matthews_corrcoef(yt, yp), abs=1e-12)


def test_brier_skill_score_from_brier_matches_its_definition():
    """BSS = 1 - Brier / (p_bar (1 - p_bar)) for the model Brier that sklearn computes."""
    rng = np.random.default_rng(3)
    y = (rng.uniform(size=1000) < 0.3).astype(int)
    p = np.clip(0.3 + 0.2 * (y - 0.3) + rng.normal(0, 0.1, 1000), 0.01, 0.99)
    brier = brier_score_loss(y, p)
    base = y.mean() * (1 - y.mean())
    assert brier_skill_score_from_brier(brier, y) == pytest.approx(1 - brier / base, abs=1e-12)


def test_logcosh_loss_equals_the_direct_formula_and_survives_large_residuals():
    """Mean log(cosh(d)) matches numpy for small d and stays finite for d = 1e4, where cosh overflows."""
    d = np.array([-1.5, -0.2, 0.0, 0.3, 2.0])
    assert fast_logcosh_loss(np.zeros(5), d) == pytest.approx(float(np.mean(np.log(np.cosh(d)))), abs=1e-12)
    assert np.isfinite(fast_logcosh_loss(np.zeros(2), np.array([1e4, -1e4])))


@pytest.mark.parametrize("weighted", [False, True])
def test_variance_weighted_r2_still_matches_sklearn_after_the_single_copy_pass(weighted):
    """fast_r2_score(variance_weighted) on 3 outputs equals sklearn's r2_score, with and without sample weights (17-F18)."""
    from sklearn.metrics import r2_score

    from mlframe.metrics.regression._regression_metrics import fast_r2_score

    rng = np.random.default_rng(4)
    yt = rng.normal(size=(400, 3)) * np.array([1.0, 5.0, 0.2])
    yp = yt + rng.normal(size=(400, 3)) * np.array([0.5, 2.0, 0.1])
    w = rng.uniform(0.2, 2.0, 400) if weighted else None
    assert fast_r2_score(yt, yp, sample_weight=w, multioutput="variance_weighted") == pytest.approx(
        r2_score(yt, yp, sample_weight=w, multioutput="variance_weighted"), rel=1e-9
    )


def test_weighted_regression_metrics_run_and_refuse_a_short_weight_vector():
    """A weighted MAE / R2 call compiles and runs, and a weight vector shorter than y raises ValueError instead of reading past its buffer."""
    from mlframe.metrics.regression._regression_metrics import fast_mean_absolute_error, fast_r2_score

    y = np.arange(10, dtype=np.float64)
    p = y + 1.0
    assert fast_mean_absolute_error(y, p, sample_weight=np.ones(10)) == pytest.approx(1.0)
    assert np.isfinite(fast_r2_score(y, p, sample_weight=np.ones(10)))
    with pytest.raises(ValueError, match="sample_weight has 3 entries"):
        fast_mean_absolute_error(y, p, sample_weight=np.ones(3))


def test_production_r2_for_a_constant_target_follows_sklearn_not_minus_infinity():
    """17-F27: a constant y_true gives 0.0 for an imperfect fit and 1.0 for a perfect one, like sklearn; the -inf lives only in a bench copy."""
    from sklearn.metrics import r2_score

    from mlframe.metrics.regression._regression_metrics import fast_r2_score

    y = np.ones(6)
    assert fast_r2_score(y, y + 1.0) == r2_score(y, y + 1.0) == 0.0
    assert fast_r2_score(y, y) == r2_score(y, y) == 1.0
