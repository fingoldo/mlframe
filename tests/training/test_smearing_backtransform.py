"""Unit + biz_value tests for Duan smearing back-transform debias (A2)."""

from __future__ import annotations

import numpy as np

from mlframe.training._regression_calibration import duan_log_smearing_factor, smearing_predict


def test_smearing_factor_is_one_on_tiny_or_zero_residuals():
    assert duan_log_smearing_factor(np.array([0.0, 0.0, 0.0])) == 1.0  # <5 -> 1.0
    assert abs(duan_log_smearing_factor(np.zeros(100)) - 1.0) < 1e-12  # exp(0)=1


def test_smearing_factor_exceeds_one_for_spread_residuals():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(5000) * 0.5  # mean 0 -> mean(exp(r)) = exp(0.5*sigma^2) > 1
    assert duan_log_smearing_factor(r) > 1.0


def test_biz_val_smearing_reduces_logtarget_backtransform_bias():
    """On a log-normal target, naive exp(pred) under-predicts the mean; Duan smearing corrects it.

    Model is the oracle on the log scale (pred=log-signal); naive exp(pred) is biased low by ~exp(sigma^2/2),
    smearing multiplies by mean(exp(resid)) to recover the conditional mean. Floor: smearing halves the |bias|.
    """
    rng = np.random.default_rng(1)
    n = 8000
    signal = rng.standard_normal(n)  # log-scale mean
    sigma = 0.7
    log_y = signal + sigma * rng.standard_normal(n)
    y = np.exp(log_y)

    pred_log = signal  # oracle log-scale prediction
    resid_cal = log_y - pred_log  # held-out log-scale residuals (~N(0, sigma))

    naive = np.exp(pred_log)
    smeared = naive * duan_log_smearing_factor(resid_cal)

    bias_naive = float(np.mean(naive - y))
    bias_smeared = float(np.mean(smeared - y))
    assert abs(bias_naive) > 0  # naive is biased
    assert abs(bias_smeared) <= 0.5 * abs(bias_naive), (bias_naive, bias_smeared)


def test_general_smearing_matches_log_factor():
    rng = np.random.default_rng(2)
    pred = rng.standard_normal(200)
    resid = rng.standard_normal(3000) * 0.4
    via_general = smearing_predict(pred, resid, np.exp)
    via_factor = np.exp(pred) * duan_log_smearing_factor(resid)
    # Both estimate E[exp(pred+resid)]; agree to a few percent (subsample noise).
    assert np.allclose(via_general, via_factor, rtol=0.05)
