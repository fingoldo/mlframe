"""biz_value + unit tests for the Bessel-corrected (ddof=1) subgroup-dispersion std in the fairness metrics.

The fairness ``metric_std`` and the ``robust_mlperf_metric`` penalty estimate the population dispersion of model performance across a small number K of subgroups.
numpy's default ddof=0 (population std) systematically UNDER-reports that dispersion at small K, making a model look fairer than it is. The default is now the
Bessel-corrected sample std (ddof=1), matching pandas / R / the rest of mlframe's FE code; ddof=0 is kept as an opt-in for legacy parity.

biz_value win (bench ``metrics/_benchmarks/bench_fairness_std_bessel.py``): on K subgroups drawn from N(mu, sigma_true^2) with KNOWN sigma_true, the SIGNED std bias
(the systematic under-estimation) is cut ~2.2-2.7x in EVERY scenario (K2 -0.0217->-0.0100, K4 -0.0204->-0.0080, K6 -0.0105->-0.0039), the variance estimand becomes
essentially unbiased, and ddof=1 wins the |estimate - sigma_true| metric in 33/40 (scenario x seed) cells (majority at every scenario). Floors below set ~15% under
the measured signed-bias reduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.metrics._fairness_metrics import compute_fairness_metrics, robust_mlperf_metric


def _abs_err_metric(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def test_biz_val_fairness_std_bessel_reduces_signed_underestimation():
    """ddof=1 cuts the SYSTEMATIC under-estimation of subgroup-performance dispersion by >=1.8x at small K (measured ~2.2-2.7x)."""
    sigma_true = 0.08
    K = 4
    n_mc = 6000
    seeds = range(6)
    bias0_all = []
    bias1_all = []
    for seed in seeds:
        rng = np.random.default_rng(seed * 7919 + K)
        b0 = b1 = 0.0
        for _ in range(n_mc):
            perfs = rng.normal(0.8, sigma_true, size=K)
            b0 += float(np.std(perfs, ddof=0)) - sigma_true
            b1 += float(np.std(perfs, ddof=1)) - sigma_true
        bias0_all.append(b0 / n_mc)
        bias1_all.append(b1 / n_mc)
    mean_b0 = abs(np.mean(bias0_all))
    mean_b1 = abs(np.mean(bias1_all))
    # Both are negative (under-estimate); ddof=1 must be closer to zero.
    assert np.mean(bias0_all) < 0 and np.mean(bias1_all) < 0
    assert mean_b1 < mean_b0
    assert mean_b0 / mean_b1 >= 1.8, f"signed-bias reduction {mean_b0 / mean_b1:.2f}x below floor 1.8x"


def test_biz_val_fairness_variance_estimand_unbiased_with_ddof1():
    """The variance estimand is provably unbiased under ddof=1: |mean var-bias| must be far smaller than ddof=0's."""
    sigma_true = 0.1
    var_true = sigma_true**2
    K = 5
    n_mc = 8000
    rng = np.random.default_rng(12345)
    vb0 = vb1 = 0.0
    for _ in range(n_mc):
        perfs = rng.normal(0.7, sigma_true, size=K)
        vb0 += float(np.var(perfs, ddof=0)) - var_true
        vb1 += float(np.var(perfs, ddof=1)) - var_true
    vb0 /= n_mc
    vb1 /= n_mc
    assert abs(vb1) < abs(vb0) / 5.0, f"ddof=1 var bias {vb1:.2e} not <<  ddof=0 {vb0:.2e}"


def _make_subgroups(rng, K, sigma):
    """Two subgroups setup feeding compute_fairness_metrics: K bins with per-bin mean perf spread sigma."""
    n_per = 400
    n = K * n_per
    bin_ids = np.repeat(np.arange(K), n_per)
    # Per-bin error level => per-bin MAE differs across groups by ~sigma.
    bin_err = rng.normal(0.0, sigma, size=K)
    y_true = rng.normal(0.0, 1.0, size=n)
    # Inject a per-bin systematic prediction offset so MAE per bin ~ |offset|.
    y_pred = y_true + np.abs(bin_err)[bin_ids]
    bins = pd.Series(bin_ids)
    subgroups = {"grp": {"bins": bins}}
    subset_index = bins.index
    return y_true, y_pred, subgroups, subset_index


def test_unit_compute_fairness_metrics_default_is_ddof1_and_larger_than_ddof0():
    """Default metric_std equals the ddof=1 path and is >= the ddof=0 path on the same data (Bessel inflates)."""
    rng = np.random.default_rng(7)
    y_true, y_pred, subgroups, subset_index = _make_subgroups(rng, K=4, sigma=0.3)
    metrics = {"mae": _abs_err_metric}
    higher = {"mae": False}

    res_default = compute_fairness_metrics(metrics, higher, subgroups, subset_index, y_true, y_pred)
    res_d1 = compute_fairness_metrics(metrics, higher, subgroups, subset_index, y_true, y_pred, ddof=1)
    res_d0 = compute_fairness_metrics(metrics, higher, subgroups, subset_index, y_true, y_pred, ddof=0)

    std_default = float(res_default["metric_std"].iloc[0])
    std_d1 = float(res_d1["metric_std"].iloc[0])
    std_d0 = float(res_d0["metric_std"].iloc[0])

    assert std_default == pytest.approx(std_d1), "default must route to ddof=1"
    assert std_d1 > std_d0 > 0.0, "Bessel std must exceed population std on K>1 finite data"
    # Exact relationship: std_d1 = std_d0 * sqrt(K/(K-1)) with K=4 -> sqrt(4/3).
    assert std_d1 == pytest.approx(std_d0 * np.sqrt(4.0 / 3.0), rel=1e-9)


def test_unit_robust_mlperf_metric_default_penalty_uses_ddof1():
    """robust_mlperf_metric default dispersion penalty uses the Bessel std (larger penalty than ddof=0)."""
    rng = np.random.default_rng(3)
    n_per = 300
    K = 3
    n = K * n_per
    y_true = rng.normal(0.0, 1.0, size=n)
    bin_off = np.array([0.0, 0.4, 0.8])
    bin_ids = np.repeat(np.arange(K), n_per)
    y_pred = y_true + bin_off[bin_ids]
    bins = {g: np.where(bin_ids == g)[0] for g in range(K)}
    subgroups = {n: {"grp": {"bins": bins, "weight": 1.0}}}

    v_default = robust_mlperf_metric(y_true, y_pred, _abs_err_metric, higher_is_better=False, subgroups=subgroups, min_group_size=10)
    v_d1 = robust_mlperf_metric(y_true, y_pred, _abs_err_metric, higher_is_better=False, subgroups=subgroups, min_group_size=10, ddof=1)
    v_d0 = robust_mlperf_metric(y_true, y_pred, _abs_err_metric, higher_is_better=False, subgroups=subgroups, min_group_size=10, ddof=0)

    assert v_default == pytest.approx(v_d1)
    # lower-is-better metric => spread is ADDED; larger ddof=1 spread => larger (worse) robust value.
    assert v_d1 > v_d0


def test_unit_fairness_std_single_finite_bin_is_nan_not_crash():
    """A group with a single finite metric yields NaN std under ddof=1 (n_finite <= ddof), never a crash or spurious 0."""
    rng = np.random.default_rng(1)
    n = 400
    y_true = rng.normal(0.0, 1.0, size=n)
    y_pred = y_true.copy()
    bins = pd.Series(np.zeros(n, dtype=int))  # single bin
    subgroups = {"grp": {"bins": bins}}
    res = compute_fairness_metrics({"mae": _abs_err_metric}, {"mae": False}, subgroups, bins.index, y_true, y_pred)
    assert np.isnan(float(res["metric_std"].iloc[0]))
