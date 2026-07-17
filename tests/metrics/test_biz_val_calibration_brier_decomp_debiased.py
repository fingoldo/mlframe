"""biz_value + unit + regression tests for the debiased Brier (Murphy) decomposition.

The headline Brier reliability/resolution in ``fast_calibration_report`` default to
``compute_brier_decomposition_debiased`` because the plug-in binned REL/RES carry the same positive per-bin
Bernoulli sampling-noise inflation that overstates the plug-in ECE: ``E[(conf_b - acc_b)^2] = true^2 + Var(acc_b)``.
These tests pin the measured win (REL bias-vs-truth more than halved on calibrated synthetics) and the EXACT Murphy
identity preservation, so a regression that reverts to the plug-in decomposition or breaks the correction trips the
suite.

Bench: ``mlframe/metrics/_benchmarks/bench_brier_decomp_debiased.py`` -- debiased wins 34-36/40 (scenario,seed) cells
across nbins 10/15/20.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_metrics import (
    compute_brier_decomposition_debiased,
    compute_ece_and_brier_decomposition,
)


def _plugin_decomp(y, p, nbins):
    """Helper: Plugin decomp."""
    return compute_ece_and_brier_decomposition(np.asarray(y, np.float64), np.asarray(p, np.float64), nbins)


def _gen_calibrated(rng, n, kind):
    """Helper: Gen calibrated."""
    if kind == "uniform":
        s = np.clip(rng.uniform(0.0, 1.0, n), 1e-6, 1 - 1e-6)
    elif kind == "beta_rare":
        s = np.clip(rng.beta(0.6, 8.0, n), 1e-6, 1 - 1e-6)
    else:  # bimodal
        half = n // 2
        s = np.empty(n)
        s[:half] = np.clip(rng.beta(2.0, 12.0, half), 1e-6, 1 - 1e-6)
        s[half:] = np.clip(rng.beta(12.0, 2.0, n - half), 1e-6, 1 - 1e-6)
    y = (rng.random(n) < s).astype(np.float64)  # acc(p) == p -> perfectly calibrated, true REL = 0
    return s, y


@pytest.mark.parametrize("kind", ["uniform", "beta_rare", "bimodal"])
def test_biz_val_brier_decomp_debiased_halves_rel_bias_on_calibrated(kind):
    """On a perfectly calibrated model (true REL=0), the debiased REL's |bias| is at most 75% of the plug-in's,
    averaged over 8 seeds. Measured ratio ~0.46-0.73; floor 0.75 absorbs seed noise."""
    nbins = 15
    n = 2000
    pl = db = 0.0
    for sd in range(8):
        rng = np.random.default_rng(sd)
        s, y = _gen_calibrated(rng, n, kind)
        pl += _plugin_decomp(y, s, nbins)[1]  # truth REL is 0, so the estimate IS the bias
        db += compute_brier_decomposition_debiased(y, s, nbins)[0]
    pl /= 8
    db /= 8
    assert db < pl, f"debiased REL ({db:.5f}) must be below plug-in ({pl:.5f}) on calibrated {kind}"
    assert db <= 0.75 * pl, f"debiased REL should reduce the bias >=25%: {db:.5f} vs 0.75*{pl:.5f}"


def test_biz_val_brier_decomp_debiased_wins_majority_of_cells():
    """Across calibrated scenarios x 8 seeds x nbins {10,15,20}, debiased REL has lower |bias| in the majority
    of cells (true REL=0). Measured ~34-36/40 per nbins; floor at >60% of cells."""
    n = 2000
    wins = total = 0
    for nbins in (10, 15, 20):
        for kind in ("uniform", "beta_rare", "bimodal"):
            for sd in range(8):
                rng = np.random.default_rng(sd)
                s, y = _gen_calibrated(rng, n, kind)
                pl = abs(_plugin_decomp(y, s, nbins)[1])
                db = abs(compute_brier_decomposition_debiased(y, s, nbins)[0])
                total += 1
                if db < pl:
                    wins += 1
    assert wins / total >= 0.60, f"debiased should win majority of calibrated cells; got {wins}/{total}"


def test_brier_decomp_debiased_self_murphy_identity_exact():
    """The debiased decomposition is internally self-consistent: REL_db - RES_db + UNC == BinnedBrier_db to fp
    precision (BinnedBrier_db is computed FROM the debiased terms). UNC is unchanged from the plug-in."""
    for sd in range(5):
        rng = np.random.default_rng(sd)
        s, y = _gen_calibrated(rng, 3000, "bimodal")
        nbins = 15
        unc_pl = _plugin_decomp(y, s, nbins)[3]
        rel_db, res_db, unc_db, binned_db = compute_brier_decomposition_debiased(y, s, nbins)
        assert unc_db == pytest.approx(unc_pl, abs=1e-15)
        assert (rel_db - res_db + unc_db) == pytest.approx(binned_db, abs=1e-12)


def test_brier_decomp_debiased_rel_minus_res_matches_plugin_no_clamp():
    """Var(acc_b) is subtracted from BOTH REL and RES, so when no REL bin clamps the difference REL_db - RES_db ==
    REL_plugin - RES_plugin and the debiased BinnedBrier == plug-in BinnedBrier. Built on one fully-populated bin
    with a large gap (clamp cannot fire). This pins the Broecker both-subtract construction."""
    p = np.full(4000, 0.9)
    y = np.zeros(4000)
    y[:400] = 1.0  # acc=0.10 within the single bin; gap^2=0.64 >> Var(acc)/(n-1) so clamp never fires
    rel_pl, res_pl, _unc_pl, binned_pl = _plugin_decomp(y, p, 10)[1:]
    rel_db, res_db, _unc_db, binned_db = compute_brier_decomposition_debiased(y, p, 10)
    assert (rel_db - res_db) == pytest.approx(rel_pl - res_pl, abs=1e-12)
    assert binned_db == pytest.approx(binned_pl, abs=1e-12)


def test_brier_rel_debiased_lowers_reliability_vs_plugin():
    """The whole point: on calibrated data the debiased REL is at or below the plug-in REL per draw (the noise floor
    is subtracted then clamped, never added), so the headline reliability stops overstating miscalibration."""
    for sd in range(5):
        rng = np.random.default_rng(sd)
        s, y = _gen_calibrated(rng, 4000, "uniform")
        nbins = 20
        rel_pl = _plugin_decomp(y, s, nbins)[1]
        rel_db = compute_brier_decomposition_debiased(y, s, nbins)[0]
        assert rel_db <= rel_pl + 1e-12


def test_brier_rel_debiased_never_exceeds_plugin_on_calibrated():
    """The REL noise correction can only subtract (then clamp), so on calibrated data debiased REL <= plug-in REL."""
    for sd in range(5):
        rng = np.random.default_rng(sd)
        s, y = _gen_calibrated(rng, 3000, "uniform")
        rel_pl = _plugin_decomp(y, s, 12)[1]
        rel_db = compute_brier_decomposition_debiased(y, s, 12)[0]
        assert rel_db <= rel_pl + 1e-12


def test_brier_rel_debiased_tracks_real_miscalibration():
    """On a grossly miscalibrated model the debiased REL stays large (true-gap term dominates the small noise
    correction), so the estimator does not wash out genuine miscalibration."""
    rng = np.random.default_rng(0)
    n = 5000
    s = np.clip(rng.uniform(0, 1, n), 1e-6, 1 - 1e-6)
    acc = np.clip(0.5 + 0.5 * (s - 0.5) * 0.2, 0, 1)  # heavily squashed toward 0.5
    y = (rng.random(n) < acc).astype(np.float64)
    rel_db = compute_brier_decomposition_debiased(y, s, 15)[0]
    assert rel_db > 0.02, f"debiased REL must still flag the gross miscalibration; got {rel_db:.4f}"


def test_brier_decomp_debiased_empty_input_degenerate():
    """Brier decomp debiased empty input degenerate."""
    rel, res, unc, binned = compute_brier_decomposition_debiased(np.empty(0), np.empty(0), 10)
    assert (rel, res, unc, binned) == (1.0, 0.0, 0.0, 1.0)


def test_brier_decomp_debiased_single_bin_span_zero():
    """All identical predictions -> single populated bin; must not crash, REL stays finite and >=0."""
    p = np.full(50, 0.3)
    y = (np.arange(50) % 2).astype(np.float64)
    rel, _res, _unc, binned = compute_brier_decomposition_debiased(y, p, 10)
    assert np.isfinite(rel) and rel >= 0.0
    assert np.isfinite(binned)


def test_report_default_uses_debiased_brier_decomp():
    """fast_calibration_report defaults brier_debiased=True; the returned REL matches the debiased kernel and is
    below the plug-in REL on a calibrated input (regression sensor for the default flip)."""
    from mlframe.metrics.classification._classification_report import fast_calibration_report

    rng = np.random.default_rng(3)
    s, y = _gen_calibrated(rng, 4000, "uniform")
    nbins = 12
    out_default = fast_calibration_report(y, s, nbins=nbins, show_plots=False, _precomputed_aucs=(0.5, 0.5))
    out_plugin = fast_calibration_report(y, s, nbins=nbins, brier_debiased=False, show_plots=False, _precomputed_aucs=(0.5, 0.5))
    rel_default = out_default[5]
    rel_plugin = out_plugin[5]
    expected_db = compute_brier_decomposition_debiased(y, s, nbins)[0]
    assert rel_default == pytest.approx(expected_db)
    assert rel_plugin == pytest.approx(_plugin_decomp(y, s, nbins)[1])
    assert rel_default < rel_plugin, "default headline REL should be the lower-bias debiased value"
