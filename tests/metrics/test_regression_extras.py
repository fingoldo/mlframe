"""Tests for mlframe.metrics._regression_extras.

Coverage:
- RMSLE / MAPE-mean / SMAPE / MdAPE / wMAPE / MASE
- MBE / CV(RMSE) / NSE / Explained Variance / Huber
- Pearson / Spearman / Kendall / Concordance
- Fused extended block: numerical agreement with individual metrics
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.metrics.core import (
    fast_rmsle, fast_mape_mean, fast_smape, fast_mdape, fast_wmape, fast_mase,
    fast_mean_bias_error, fast_cv_rmse, fast_nash_sutcliffe,
    fast_explained_variance, fast_huber_loss,
    fast_pearson_corr, fast_spearman_corr, fast_kendall_tau,
    fast_concordance_index,
    fast_regression_metrics_block_extended,
)


# ----- RMSLE -----


def test_rmsle_zero_on_perfect_prediction():
    y = np.array([1.0, 2.0, 5.0, 10.0])
    assert fast_rmsle(y, y) == pytest.approx(0.0)


def test_rmsle_matches_manual_formula():
    y = np.array([1.0, 2.0, 5.0])
    p = np.array([0.5, 2.5, 4.0])
    expected = float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))
    assert fast_rmsle(y, p) == pytest.approx(expected, abs=1e-12)


def test_rmsle_warns_on_negative_inputs():
    y = np.array([1.0, -2.0, 5.0])
    p = np.array([1.0, 2.0, 5.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fast_rmsle(y, p)
        assert any("RMSLE" in str(rec.message) for rec in w)


def test_rmsle_all_negative_returns_nan():
    y = np.array([-1.0, -2.0])
    p = np.array([-1.0, -2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(fast_rmsle(y, p))


# ----- MAPE-mean -----


def test_mape_mean_matches_sklearn():
    from sklearn.metrics import mean_absolute_percentage_error
    rng = np.random.default_rng(0)
    y = rng.uniform(1, 100, 200)
    p = y + rng.normal(0, 5, 200)
    assert fast_mape_mean(y, p) == pytest.approx(
        mean_absolute_percentage_error(y, p), abs=1e-12,
    )


# ----- SMAPE / MdAPE / wMAPE -----


def test_smape_bounded():
    """sMAPE is in [0, 2] by construction."""
    rng = np.random.default_rng(1)
    y = rng.uniform(-5, 5, 500)
    p = rng.uniform(-5, 5, 500)
    val = fast_smape(y, p)
    assert 0.0 <= val <= 2.0


def test_mdape_robust_to_outlier():
    """One outlier should NOT inflate MdAPE the way it does MAPE."""
    y = np.full(1000, 10.0)
    p = y.copy()
    # 10% blow-up outlier
    p[0] = 1e6
    mape = fast_mape_mean(y, p)
    mdape = fast_mdape(y, p)
    assert mape > 90  # huge
    assert mdape == pytest.approx(0.0, abs=1e-12)  # median residual is 0


def test_wmape_handles_zero_y():
    """wMAPE shouldn't NaN on individual y=0 (only when ALL y are 0)."""
    y = np.array([0.0, 1.0, 2.0, 0.0])
    p = np.array([0.5, 1.0, 2.5, 0.1])
    val = fast_wmape(y, p)
    expected = (0.5 + 0.0 + 0.5 + 0.1) / (0.0 + 1.0 + 2.0 + 0.0)
    assert val == pytest.approx(expected, abs=1e-12)


def test_wmape_mixed_dtypes_bit_equivalent():
    """iter596: fast_wmape dropped the unconditional ``dtype=np.float64``
    cast. Bit-equivalence must hold across the dtype pairs that appear in
    regression reporting: (int targets, float64 preds), (float64, float32),
    and the pure float64 baseline."""
    rng = np.random.default_rng(20260530)
    n = 25_000
    y_int = rng.integers(1, 100, size=n, dtype=np.int64)
    y_f64 = y_int.astype(np.float64) + rng.random(n) * 0.5
    p_f64 = y_f64 + rng.normal(scale=0.1, size=n)
    p_f32 = p_f64.astype(np.float32)
    reference = fast_wmape(y_f64, p_f64)
    for y_t, y_p, atol in [
        (y_int, p_f64, 1e-6),
        (y_f64, p_f64, 1e-12),
        (y_f64, p_f32, 1e-5),
    ]:
        v = fast_wmape(y_t, y_p)
        ref = (np.abs(np.asarray(y_p, np.float64) - np.asarray(y_t, np.float64)).sum()
               / np.abs(np.asarray(y_t, np.float64)).sum())
        assert abs(v - ref) < atol, (
            f"dtypes ({y_t.dtype}, {y_p.dtype}): fast_wmape={v} vs ref={ref}"
        )
    # Sanity: float64+float64 path unchanged
    v_ctrl = fast_wmape(y_f64, p_f64)
    assert v_ctrl == pytest.approx(reference, abs=1e-12)


# ----- MASE -----


def test_mase_zero_on_perfect_prediction():
    rng = np.random.default_rng(3)
    y_train = rng.standard_normal(100) * 5 + 50
    y = y_train.copy()
    assert fast_mase(y, y, y_train, seasonality=1) == pytest.approx(0.0)


def test_mase_constant_train_returns_nan():
    """No naive-baseline signal -> NaN."""
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 3.0])
    y_train = np.array([5.0, 5.0, 5.0, 5.0])
    assert np.isnan(fast_mase(y, p, y_train, seasonality=1))


# ----- MBE / CV(RMSE) -----


def test_mbe_signed():
    y = np.array([1.0, 2.0, 3.0])
    # over-prediction by 1
    p = y + 1.0
    assert fast_mean_bias_error(y, p) == pytest.approx(1.0)
    # under-prediction by 1
    p = y - 1.0
    assert fast_mean_bias_error(y, p) == pytest.approx(-1.0)


def test_cv_rmse_unit_free():
    y = np.array([10.0, 20.0, 30.0])
    p = np.array([11.0, 21.0, 31.0])
    expected = float(np.sqrt(np.mean((y - p) ** 2)) / abs(y.mean()))
    assert fast_cv_rmse(y, p) == pytest.approx(expected, abs=1e-12)


# ----- NSE / Explained Variance -----


def test_nse_equals_r2_score():
    """NSE = R^2 (sklearn convention) on the same data."""
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(4)
    y = rng.standard_normal(500)
    p = y + rng.standard_normal(500) * 0.5
    assert fast_nash_sutcliffe(y, p) == pytest.approx(r2_score(y, p), abs=1e-12)


def test_explained_variance_matches_sklearn():
    from sklearn.metrics import explained_variance_score
    rng = np.random.default_rng(5)
    y = rng.standard_normal(500)
    p = y + rng.standard_normal(500) * 0.5 + 2.0
    assert fast_explained_variance(y, p) == pytest.approx(
        explained_variance_score(y, p), abs=1e-12,
    )


# ----- Huber -----


def test_huber_loss_quadratic_for_small_residuals():
    y = np.array([0.0, 0.0])
    p = np.array([0.5, 0.5])
    # |r|=0.5 < delta=1 -> 0.5 * 0.5^2 = 0.125 each
    assert fast_huber_loss(y, p, delta=1.0) == pytest.approx(0.125, abs=1e-12)


def test_huber_loss_linear_for_large_residuals():
    y = np.array([0.0])
    p = np.array([5.0])
    # |r|=5 > delta=1 -> delta*(|r| - 0.5*delta) = 1*(5-0.5) = 4.5
    assert fast_huber_loss(y, p, delta=1.0) == pytest.approx(4.5, abs=1e-12)


# ----- Pearson / Spearman / Kendall / C-index -----


def test_pearson_corr_matches_numpy():
    rng = np.random.default_rng(6)
    y = rng.standard_normal(300)
    p = 0.5 * y + rng.standard_normal(300) * 0.3
    assert fast_pearson_corr(y, p) == pytest.approx(
        float(np.corrcoef(y, p)[0, 1]), abs=1e-12,
    )


def test_spearman_corr_matches_scipy():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(7)
    y = rng.standard_normal(300)
    p = y + rng.standard_normal(300) * 0.3
    expected = float(spearmanr(y, p).correlation)
    assert fast_spearman_corr(y, p) == pytest.approx(expected, abs=1e-10)


def test_kendall_tau_matches_scipy_small_N():
    from scipy.stats import kendalltau
    rng = np.random.default_rng(8)
    N = 200
    y = rng.standard_normal(N)
    p = y + rng.standard_normal(N) * 0.3
    expected = float(kendalltau(y, p, variant="b").correlation)
    assert fast_kendall_tau(y, p) == pytest.approx(expected, abs=1e-10)


def test_kendall_tau_mid_range_uses_scipy_and_matches_numba_kernel():
    """For 500 < N <= 5000 the dispatch routes through scipy's O(N log N) merge-
    sort (the historical N<=5000 threshold mis-attributed a crossover that re-
    benching on modern scipy / numba put at ~N=400). The scalar return must
    match the in-process O(N^2) numba kernel to fp tolerance because both
    implement the same tie-corrected tau-b formula -- pin that equivalence so
    the threshold cannot silently drift back."""
    from scipy.stats import kendalltau
    from mlframe.metrics._regression_extras import _kendall_tau_b_kernel
    rng = np.random.default_rng(20260528)
    for N in (600, 1500, 3000):
        y = rng.standard_normal(N)
        p = y + rng.standard_normal(N) * 0.3
        scipy_val = float(kendalltau(y, p, variant="b").correlation)
        numba_val = float(_kendall_tau_b_kernel(y.astype(np.float64), p.astype(np.float64)))
        dispatched = fast_kendall_tau(y, p)
        # The dispatched scalar must match scipy (the routed path) exactly,
        # and the underlying numba kernel must agree with scipy to fp tolerance.
        assert dispatched == pytest.approx(scipy_val, abs=1e-12)
        assert numba_val == pytest.approx(scipy_val, abs=1e-10), (
            f"numba O(N^2) and scipy O(N log N) disagree at N={N}: "
            f"numba={numba_val} scipy={scipy_val}"
        )


def test_kendall_tau_falls_back_to_scipy_large_N():
    """N > 5000 should hand off to scipy without crashing."""
    rng = np.random.default_rng(9)
    N = 6000
    y = rng.standard_normal(N)
    p = y + rng.standard_normal(N) * 0.3
    val = fast_kendall_tau(y, p)
    assert np.isfinite(val) and -1.0 <= val <= 1.0


def test_concordance_index_range():
    rng = np.random.default_rng(10)
    y = rng.standard_normal(500)
    p_good = y + rng.standard_normal(500) * 0.1  # very correlated
    p_bad = rng.standard_normal(500)             # uncorrelated
    c_good = fast_concordance_index(y, p_good)
    c_bad = fast_concordance_index(y, p_bad)
    assert c_good > 0.9
    assert 0.4 <= c_bad <= 0.6


# ----- Fused extended block -----


def test_extended_block_matches_individual_metrics():
    """Each block-emitted scalar must equal its individual-call counterpart
    to within fp ordering jitter (< 1e-12)."""
    rng = np.random.default_rng(11)
    N = 5000
    y = rng.standard_normal(N) * 10 + 100
    p = y + rng.standard_normal(N) * 2
    block = fast_regression_metrics_block_extended(y, p)
    assert block["MAE"] == pytest.approx(float(np.mean(np.abs(y - p))), abs=1e-10)
    assert block["RMSE"] == pytest.approx(float(np.sqrt(np.mean((y - p) ** 2))), abs=1e-10)
    assert block["MaxError"] == pytest.approx(float(np.max(np.abs(y - p))), abs=1e-12)
    assert block["MBE"] == pytest.approx(fast_mean_bias_error(y, p), abs=1e-10)
    assert block["MAPE_mean"] == pytest.approx(fast_mape_mean(y, p), abs=1e-10)
    assert block["SMAPE"] == pytest.approx(fast_smape(y, p), abs=1e-10)
    assert block["wMAPE"] == pytest.approx(fast_wmape(y, p), abs=1e-10)
    assert block["CV_RMSE"] == pytest.approx(fast_cv_rmse(y, p), abs=1e-10)
    assert block["NSE"] == pytest.approx(fast_nash_sutcliffe(y, p), abs=1e-10)
    assert block["Pearson"] == pytest.approx(fast_pearson_corr(y, p), abs=1e-10)
    assert block["ExplainedVariance"] == pytest.approx(fast_explained_variance(y, p), abs=1e-10)


def test_extended_block_handles_large_n_parallel_path():
    """N >= 100k triggers the parallel kernel branches."""
    rng = np.random.default_rng(12)
    N = 200_000
    y = rng.standard_normal(N)
    p = y + rng.standard_normal(N) * 0.5
    block = fast_regression_metrics_block_extended(y, p)
    # Parallel path must agree with individual call within 1e-10.
    assert block["MAE"] == pytest.approx(float(np.mean(np.abs(y - p))), abs=1e-10)
    assert block["NSE"] == pytest.approx(fast_nash_sutcliffe(y, p), abs=1e-10)


def test_extended_block_empty_input():
    block = fast_regression_metrics_block_extended(np.array([]), np.array([]))
    for k in ("MAE", "RMSE", "MaxError", "R2", "MBE", "MAPE_mean", "SMAPE"):
        assert np.isnan(block[k])


def test_extended_block_constant_y():
    """Constant y -> ss_tot=0 -> R2/EV undefined (we return -inf / nan)."""
    y = np.full(100, 5.0)
    p = np.full(100, 5.5)
    block = fast_regression_metrics_block_extended(y, p)
    # MAE/RMSE finite
    assert block["MAE"] == pytest.approx(0.5, abs=1e-12)
    # R2 follows sklearn convention: -inf when ss_tot=0 and SSE>0
    assert block["R2"] == float("-inf")
