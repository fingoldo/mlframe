"""biz_value + unit + regression tests for the debiased binned ECE estimator.

The headline ECE in ``fast_calibration_report`` defaults to ``compute_ece_debiased`` because the plug-in
binned ECE positively overstates miscalibration by the per-bin Bernoulli sampling noise. These tests pin
the measured win (bias-vs-truth more than halved on calibrated synthetics) so a regression that silently
reverts to the plug-in estimator or breaks the noise correction trips the suite.

Bench: ``mlframe/metrics/_benchmarks/bench_ece_debiased.py`` -- debiased wins ~80% of (scenario,seed)
cells across nbins 10/15/20.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_metrics import (
    compute_ece_and_brier_decomposition,
    compute_ece_debiased,
)


def _plugin_ece(y, p, nbins):
    return compute_ece_and_brier_decomposition(np.asarray(y, np.float64), np.asarray(p, np.float64), nbins)[0]


def _gen_calibrated(rng, n, kind):
    if kind == "uniform":
        s = np.clip(rng.uniform(0.0, 1.0, n), 1e-6, 1 - 1e-6)
    elif kind == "beta_rare":
        s = np.clip(rng.beta(0.6, 8.0, n), 1e-6, 1 - 1e-6)
    else:  # bimodal
        half = n // 2
        s = np.empty(n)
        s[:half] = np.clip(rng.beta(2.0, 12.0, half), 1e-6, 1 - 1e-6)
        s[half:] = np.clip(rng.beta(12.0, 2.0, n - half), 1e-6, 1 - 1e-6)
    y = (rng.random(n) < s).astype(np.float64)  # acc(p) == p -> perfectly calibrated, true ECE = 0
    return s, y


@pytest.mark.parametrize("kind", ["uniform", "beta_rare", "bimodal"])
def test_biz_val_ece_debiased_halves_bias_on_calibrated(kind):
    """On a perfectly calibrated model (true ECE=0), the debiased estimator's |bias| is at most 70% of the
    plug-in's, averaged over 7 seeds. Measured ratio ~0.43; floor 0.70 absorbs seed noise."""
    nbins = 15
    n = 2000
    pl = db = 0.0
    for sd in range(7):
        rng = np.random.default_rng(sd)
        s, y = _gen_calibrated(rng, n, kind)
        pl += _plugin_ece(y, s, nbins)  # truth is 0, so the estimate IS the bias
        db += compute_ece_debiased(y, s, nbins)
    pl /= 7
    db /= 7
    assert db < pl, f"debiased ECE ({db:.5f}) must be below plug-in ({pl:.5f}) on calibrated {kind}"
    assert db <= 0.70 * pl, f"debiased ECE should more than 30%-reduce the bias: {db:.5f} vs 0.70*{pl:.5f}"


def test_biz_val_ece_debiased_wins_majority_of_cells():
    """Across calibrated scenarios x 7 seeds x nbins {10,15,20}, debiased has lower |bias| in the majority
    of cells (true ECE=0). Measured ~28/35 per nbins; floor at >60% of cells."""
    n = 2000
    wins = total = 0
    for nbins in (10, 15, 20):
        for kind in ("uniform", "beta_rare", "bimodal"):
            for sd in range(7):
                rng = np.random.default_rng(sd)
                s, y = _gen_calibrated(rng, n, kind)
                pl = abs(_plugin_ece(y, s, nbins))
                db = abs(compute_ece_debiased(y, s, nbins))
                total += 1
                if db < pl:
                    wins += 1
    assert wins / total >= 0.60, f"debiased should win majority of calibrated cells; got {wins}/{total}"


def test_ece_debiased_never_exceeds_plugin_on_perfectly_calibrated():
    """The noise correction can only subtract, so on calibrated data the debiased ECE is <= plug-in per-cell."""
    for sd in range(5):
        rng = np.random.default_rng(sd)
        s, y = _gen_calibrated(rng, 3000, "uniform")
        assert compute_ece_debiased(y, s, 12) <= _plugin_ece(y, s, 12) + 1e-12


def test_ece_debiased_tracks_real_miscalibration():
    """On a grossly miscalibrated model the debiased ECE stays large (the true-gap term dominates the small
    noise correction), so the estimator does not wash out genuine miscalibration."""
    rng = np.random.default_rng(0)
    n = 5000
    s = np.clip(rng.uniform(0, 1, n), 1e-6, 1 - 1e-6)
    acc = np.clip(0.5 + 0.5 * (s - 0.5) * 0.2, 0, 1)  # heavily squashed toward 0.5
    y = (rng.random(n) < acc).astype(np.float64)
    db = compute_ece_debiased(y, s, 15)
    assert db > 0.10, f"debiased ECE must still flag the gross miscalibration; got {db:.4f}"


def test_ece_debiased_empty_input_degenerate():
    assert compute_ece_debiased(np.empty(0), np.empty(0), 10) == 1.0


def test_ece_debiased_single_bin_span_zero():
    """All identical predictions -> single populated bin; must not crash and stays finite."""
    p = np.full(50, 0.3)
    y = (np.arange(50) % 2).astype(np.float64)
    v = compute_ece_debiased(y, p, 10)
    assert np.isfinite(v) and v >= 0.0


def test_report_default_uses_debiased_ece():
    """fast_calibration_report defaults ece_debiased=True; the returned headline ECE matches the debiased
    kernel and differs from the plug-in on a calibrated input (regression sensor for the default flip)."""
    from mlframe.metrics.classification._classification_report import fast_calibration_report

    rng = np.random.default_rng(3)
    s, y = _gen_calibrated(rng, 4000, "uniform")
    nbins = 12
    out_default = fast_calibration_report(y, s, nbins=nbins, show_plots=False, _precomputed_aucs=(0.5, 0.5))
    out_plugin = fast_calibration_report(y, s, nbins=nbins, ece_debiased=False, show_plots=False, _precomputed_aucs=(0.5, 0.5))
    ece_default = out_default[4]
    ece_plugin = out_plugin[4]
    assert ece_default == pytest.approx(compute_ece_debiased(y, s, nbins))
    assert ece_plugin == pytest.approx(_plugin_ece(y, s, nbins))
    assert ece_default < ece_plugin, "default headline ECE should be the lower-bias debiased value"
