"""Unit + biz_value tests for Duan smearing back-transform debias (A2)."""

from __future__ import annotations

import numpy as np

import mlframe.training._regression_calibration as _regcal
from mlframe.training._regression_calibration import duan_log_smearing_factor, smearing_predict


def test_smearing_factor_is_one_on_tiny_or_zero_residuals():
    """Smearing factor is one on tiny or zero residuals."""
    assert duan_log_smearing_factor(np.array([0.0, 0.0, 0.0])) == 1.0  # <5 -> 1.0
    assert abs(duan_log_smearing_factor(np.zeros(100)) - 1.0) < 1e-12  # exp(0)=1


def test_smearing_factor_exceeds_one_for_spread_residuals():
    """Smearing factor exceeds one for spread residuals."""
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
    """General smearing matches log factor."""
    rng = np.random.default_rng(2)
    pred = rng.standard_normal(200)
    resid = rng.standard_normal(3000) * 0.4
    via_general = smearing_predict(pred, resid, np.exp)
    via_factor = np.exp(pred) * duan_log_smearing_factor(resid)
    # Both estimate E[exp(pred+resid)]; agree to a few percent (subsample noise).
    assert np.allclose(via_general, via_factor, rtol=0.05)


def test_smearing_predict_chunks_n_test_without_unbounded_broadcast(monkeypatch):
    """TRAINING_LOOSE_B-3 regression: n_test alone (not just n_cal) must be bounded per broadcast chunk.

    Pre-fix, ``smearing_predict`` built one ``(n_test, n_cal)`` array regardless of n_test, risking OOM
    on large predict batches. Forces a tiny per-chunk cell budget (so a realistic n_test/n_cal actually
    splits into multiple chunks) and asserts: no single ``inverse_fn`` call ever sees more than the
    budgeted number of cells, AND the chunked result is numerically identical to a direct (unchunked)
    computation over the same (already-subsampled) residual set.
    """
    rng = np.random.default_rng(3)
    n_test, n_cal = 500, 37
    pred = rng.standard_normal(n_test)
    resid = rng.standard_normal(n_cal) * 0.3

    seen_max_cells = 0

    def _spying_exp(x: np.ndarray) -> np.ndarray:
        """Records the largest single-call array size seen, then delegates to np.exp."""
        nonlocal seen_max_cells
        seen_max_cells = max(seen_max_cells, x.size)
        return np.exp(x)

    monkeypatch.setattr(_regcal, "_SMEARING_MAX_GRID_CELLS", n_cal * 10)  # forces ~5 chunks over n_test
    chunked = smearing_predict(pred, resid, _spying_exp, max_cal=n_cal + 1000)  # max_cal never trims here
    assert seen_max_cells <= n_cal * 10, "a single inverse_fn call exceeded the configured cell budget"
    assert seen_max_cells < n_test * n_cal, "chunking never actually kicked in for this fixture"

    direct = np.mean(np.exp(pred[:, None] + resid[None, :]), axis=1)
    assert np.allclose(chunked, direct, rtol=0, atol=1e-12)
