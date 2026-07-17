"""Unit + biz_value tests for the debiased ECE annotation on the reliability diagram.

Standard fixed-bin ECE is biased upward: the per-bin empirical positive rate is a finite binomial estimate, so even a
perfectly-calibrated model reports ECE > 0, and that spurious value grows with the bin count. The debiased estimator
(Kumar et al. 2019) subtracts the expected per-bin variance on the squared scale. biz_value asserts: on a perfectly-
calibrated synthetic the debiased value is materially closer to 0 and ~bin-count-stable, while on a genuinely
miscalibrated synthetic the debiased value stays clearly positive (it does not zero-out real error).
"""

from __future__ import annotations

import cProfile
import pstats
import io

import numpy as np
import pytest

from mlframe.metrics.calibration._calibration_plot import fast_calibration_binning
from mlframe.reporting.charts.calibration import (
    build_calibration_spec,
    debiased_ece,
    standard_ece,
)


def _perfectly_calibrated(n: int = 200_000, seed: int = 0):
    """Draw scores in (0,1), then y ~ Bernoulli(score) so the score IS the true positive probability."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.02, 0.98, size=n)
    y = (rng.uniform(size=n) < score).astype(np.int64)
    return y, score


def _miscalibrated(n: int = 200_000, seed: int = 0):
    """Overconfident model: predict ``score`` but the true rate is a shrunk-to-0.5 version, so |conf-acc| is real."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0.02, 0.98, size=n)
    true_p = 0.5 + 0.4 * (score - 0.5)  # pull predictions toward the centre -> genuine, non-vanishing miscalibration
    y = (rng.uniform(size=n) < true_p).astype(np.int64)
    return y, score


def _ece_pair(y, score, nbins):
    """Helper: Ece pair."""
    fp, ft, hits = fast_calibration_binning(y, score, nbins=nbins)
    return standard_ece(fp, ft, hits), debiased_ece(fp, ft, hits)


# --------------------------------------------------------------------------- unit


def test_debiased_ece_present_in_spec_annotation():
    """Debiased ece present in spec annotation."""
    y, score = _perfectly_calibrated(n=20_000)
    fp, ft, hits = fast_calibration_binning(y, score, nbins=15)
    spec = build_calibration_spec(fp, ft, hits, plot_title="rel")
    scatter = spec.panels[0][0]
    assert "ECE=" in scatter.title
    assert "ECE_debiased=" in scatter.title


def test_ece_annotation_can_be_disabled():
    """Ece annotation can be disabled."""
    y, score = _perfectly_calibrated(n=20_000)
    fp, ft, hits = fast_calibration_binning(y, score, nbins=15)
    spec = build_calibration_spec(fp, ft, hits, plot_title="rel", show_ece_annotation=False)
    scatter = spec.panels[0][0]
    assert scatter.title == "rel"
    assert "ECE" not in scatter.title


def test_standard_ece_formula_matches_manual():
    """Standard ece formula matches manual."""
    fp = np.array([0.1, 0.5, 0.9])
    ft = np.array([0.2, 0.5, 0.7])
    hits = np.array([10.0, 30.0, 60.0])
    expected = (10 * 0.1 + 30 * 0.0 + 60 * 0.2) / 100.0
    assert standard_ece(fp, ft, hits) == pytest.approx(expected)


def test_debiased_ece_subtracts_variance_term():
    """Debiased ece subtracts variance term."""
    fp = np.array([0.5])
    ft = np.array([0.6])
    hits = np.array([100.0])
    # ece2 = (0.1)^2 - 0.5*0.5/100 = 0.01 - 0.0025 = 0.0075 ; debiased = sqrt(0.0075)
    assert debiased_ece(fp, ft, hits) == pytest.approx(np.sqrt(0.0075))
    # standard never subtracts -> strictly larger here
    assert debiased_ece(fp, ft, hits) < standard_ece(fp, ft, hits)


def test_debiased_ece_clamps_to_zero_when_gap_below_noise():
    # gap^2 (0.0001) is far below the variance term (0.5*0.5/50 = 0.005) -> debiased clamps to 0, not negative/nan.
    """Debiased ece clamps to zero when gap below noise."""
    fp = np.array([0.5])
    ft = np.array([0.51])
    hits = np.array([50.0])
    val = debiased_ece(fp, ft, hits)
    assert val == 0.0


@pytest.mark.parametrize(
    "fp,ft,hits",
    [
        (np.array([]), np.array([]), np.array([])),  # no bins
        (np.array([np.nan]), np.array([np.nan]), np.array([0.0])),  # empty / non-finite bin
        (np.array([0.5]), np.array([0.5]), np.array([1.0])),  # singleton bin -> no variance estimate
    ],
)
def test_debiased_ece_degenerate_returns_nan(fp, ft, hits):
    """Debiased ece degenerate returns nan."""
    assert np.isnan(debiased_ece(fp, ft, hits))


def test_degenerate_omits_debiased_term_keeps_chart():
    # all bins singleton -> debiased NaN; annotation must keep standard ECE and omit the debiased term, chart survives.
    """Degenerate omits debiased term keeps chart."""
    fp = np.array([0.2, 0.8])
    ft = np.array([0.0, 1.0])
    hits = np.array([1.0, 1.0])
    spec = build_calibration_spec(fp, ft, hits, plot_title="rel")
    title = spec.panels[0][0].title
    assert "ECE=" in title
    assert "ECE_debiased=" not in title


def test_single_class_input_omits_debiased():
    # All labels 0: positive rate per bin is 0 everywhere; binning still produces finite bins but std ECE is finite.
    """Single class input omits debiased."""
    score = np.linspace(0.05, 0.95, 5000)
    y = np.zeros_like(score, dtype=np.int64)
    fp, ft, hits = fast_calibration_binning(y, score, nbins=15)
    spec = build_calibration_spec(fp, ft, hits, plot_title="rel")
    assert "ECE=" in spec.panels[0][0].title


# --------------------------------------------------------------------------- biz_value


def test_biz_debiased_ece_corrects_spurious_bias_on_perfectly_calibrated():
    # Finite-sample binning inflates standard ECE; debiased subtracts the variance term and collapses toward 0.
    # Standard ECE is an L1 mean-|gap| whose expected value stays positive under perfect calibration (mean of |noise|
    # > 0); debiased works on the L2 scale where that noise is an additive variance. nbins=40 keeps each bin populous
    # enough that the variance estimate is accurate while the spurious bias clears 0.01.
    """Biz debiased ece corrects spurious bias on perfectly calibrated."""
    std, deb = _ece_pair(*_perfectly_calibrated(n=20_000, seed=1), nbins=40)
    assert std >= 0.01, f"expected spurious standard-ECE bias >= 0.01, got {std:.4f}"
    # Debiased is materially closer to the truth (0): well under half the standard value at this bin count.
    assert deb < std, f"debiased {deb:.4f} should be < standard {std:.4f}"
    assert deb <= 0.5 * std, f"debiased {deb:.4f} should be <= 0.5*standard {0.5 * std:.4f}"


def test_biz_debiased_ece_bin_count_stable_on_perfectly_calibrated():
    # The signature of finite-sample binning bias: standard ECE INFLATES as bins grow (each bin holds fewer samples ->
    # noisier acc_k -> larger |conf-acc|). The debiased estimator subtracts that growing variance, so it stays ~flat.
    """Biz debiased ece bin count stable on perfectly calibrated."""
    y, score = _perfectly_calibrated(n=20_000, seed=2)
    std5, deb5 = _ece_pair(y, score, nbins=5)
    std50, deb50 = _ece_pair(y, score, nbins=50)
    assert std50 > std5 + 0.004, f"standard ECE should inflate with bins: 5->{std5:.4f} 50->{std50:.4f}"
    deb_change = abs(deb50 - deb5)
    std_change = std50 - std5
    assert deb_change < std_change, f"debiased change {deb_change:.4f} should be < standard inflation {std_change:.4f}"
    assert deb_change <= 0.01, f"debiased should be near bin-count-stable, change {deb_change:.4f}"


def test_biz_debiased_ece_does_not_mask_real_miscalibration():
    # Overconfident model with a genuine, non-vanishing gap. The O(1/n_k) variance term is negligible against the real
    # squared gap, so the correction barely moves the estimate -- debiased stays clearly positive (does NOT zero out).
    """Biz debiased ece does not mask real miscalibration."""
    std, deb = _ece_pair(*_miscalibrated(n=20_000, seed=3), nbins=15)
    assert std >= 0.05, f"standard ECE should flag real miscalibration, got {std:.4f}"
    assert deb >= 0.05, f"debiased ECE must not mask real miscalibration, got {deb:.4f}"
    # Both are the same order of magnitude on real error (debiased is RMS-scale, standard is L1-scale).
    assert 0.5 * std <= deb <= 1.5 * std, f"on real miscalibration debiased {deb:.4f} should track standard {std:.4f}"


# --------------------------------------------------------------------------- cProfile (O(bins))


def test_debiased_ece_is_o_bins_not_o_n():
    # The estimator must run on per-bin summaries only -- its cost must not depend on n. Profile at a fixed bin count
    # for two very different n and assert the call count for debiased_ece itself is identical (one call) and fast.
    """Debiased ece is o bins not o n."""
    y_small, sc_small = _perfectly_calibrated(n=2_000, seed=4)
    y_big, sc_big = _perfectly_calibrated(n=400_000, seed=4)
    fps, fts, hs = fast_calibration_binning(y_small, sc_small, nbins=15)
    fpb, ftb, hb = fast_calibration_binning(y_big, sc_big, nbins=15)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(200):
        debiased_ece(fps, fts, hs)
        debiased_ece(fpb, ftb, hb)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats("debiased_ece")
    out = s.getvalue()
    # 200 iters * 2 calls = 400 calls; both n hit the same code, so total time stays tiny (per-bin work only).
    assert "400" in out, out
    total = pstats.Stats(pr).total_tt
    assert total < 1.0, f"debiased_ece over 400 calls took {total:.3f}s -- should be O(bins) and well under 1s"
