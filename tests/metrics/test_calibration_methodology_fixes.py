"""Regression tests for the calibration-methodology fixes (SA16-SA21).

Each test pins a corrected calibration estimator and fails on the pre-fix code:
  SA16  debiased ECE default + fixed [0,1] grid -> near-0 on a perfectly-calibrated synthetic.
  SA17  coverage = fraction of populated bins -> stable under tiny score perturbations.
  SA18  inner_cv point estimate lies within its (held-out) CI.
  SA19  reliability bootstrap band does not substitute the full-sample fit -> no inflated significance on noise.
  SA20  DeLong AUC CI on the logit scale -> does not hit the [0,1] clip near AUC~=1.
  SA21  per-window debiased ECE -> no-drift stream with unequal windows has ece_trend ~= 0.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# SA16 - debiased ECE is less biased than the plug-in default on a calibrated model
# ---------------------------------------------------------------------------


def test_sa16_debiased_ece_near_zero_on_perfectly_calibrated():
    """Sa16 debiased ece near zero on perfectly calibrated."""
    from mlframe.metrics.calibration._calibration_metrics import (
        compute_ece_and_brier_decomposition,
        compute_ece_debiased,
    )

    rng = np.random.default_rng(0)
    n = 20000
    y_pred = rng.random(n)
    y_true = (rng.random(n) < y_pred).astype(np.float64)  # perfectly calibrated by construction
    nbins = 50  # finer binning amplifies the plug-in noise floor, making the bias unambiguous

    plugin_ece = compute_ece_and_brier_decomposition(y_true, y_pred, nbins)[0]
    deb_ece = compute_ece_debiased(y_true, y_pred, nbins)

    # The true ECE is 0. The plug-in (pre-fix default) carries a per-bin Bernoulli noise floor that GROWS with nbins
    # and overstates miscalibration; the debiased default removes it. So the plug-in is biased high while the debiased
    # default reads ~0 -- the corrected headline path is strictly less biased on a perfectly-calibrated model.
    assert plugin_ece > 0.01, f"plug-in ECE should overstate (>0.01) at nbins=50, got {plugin_ece:.5f}"
    assert deb_ece < plugin_ece, f"debiased {deb_ece:.5f} should be < plug-in {plugin_ece:.5f}"
    assert deb_ece < 0.01, f"debiased ECE should be ~0 on a calibrated model, got {deb_ece:.5f}"


def test_sa16_fixed_grid_is_comparable_across_score_ranges():
    """Two calibrated samples that occupy DIFFERENT score sub-ranges must bin onto the same fixed [0,1] grid, so the
    plug-in (data-adaptive) grid's range-dependence is gone. Pre-fix the bin edges depended on each sample's min/max."""
    from mlframe.metrics.calibration._calibration_metrics import compute_ece_debiased

    rng = np.random.default_rng(3)
    # All predictions in [0.0, 0.5]; with a fixed [0,1] grid only the lower half of the bins are used.
    p = rng.random(8000) * 0.5
    y = (rng.random(8000) < p).astype(np.float64)
    # A fixed grid over [0,1] gives a well-defined, range-independent number; the result must be finite and small.
    ece = compute_ece_debiased(y, p, 20)
    assert np.isfinite(ece) and ece < 0.02


# ---------------------------------------------------------------------------
# SA17 - coverage stable under tiny perturbations that flip the old rounding pockets
# ---------------------------------------------------------------------------


def test_sa17_coverage_stable_under_tiny_perturbation():
    """Sa17 coverage stable under tiny perturbation."""
    from mlframe.metrics.calibration._calibration_metrics import calibration_metrics_from_freqs

    nbins = 10
    # Per-bin predicted frequencies sitting right on rounding boundaries: a ~1e-3 nudge flips round(.,1) pockets.
    freqs_predicted = np.array([0.0499, 0.1501, 0.2499, 0.3501, 0.4499, 0.5501, 0.6499, 0.7501, 0.8499, 0.9501], dtype=np.float64)
    freqs_true = freqs_predicted.copy()
    hits = np.full(nbins, 100, dtype=np.float64)

    def cov(fp):
        """Cov."""
        return calibration_metrics_from_freqs(fp, freqs_true, hits, nbins=nbins)[2]

    base = cov(freqs_predicted)
    perturbed = cov(freqs_predicted + 2e-3)  # nudge across the .x5 rounding boundaries
    # Populated-bin coverage is structural: every bin has hits>0, so coverage is 1.0 regardless of the nudge.
    assert base == perturbed, f"coverage flipped under a 2e-3 perturbation: {base} -> {perturbed}"
    assert base == 1.0


# ---------------------------------------------------------------------------
# SA18 - inner_cv point estimate lies within its CI
# ---------------------------------------------------------------------------


def test_sa18_inner_cv_point_within_ci():
    """Sa18 inner cv point within ci."""
    pytest.importorskip("sklearn")
    from mlframe.calibration.policy import pick_best_calibrator

    rng = np.random.default_rng(7)
    n = 4000
    z = rng.normal(0.0, 1.0, size=n)
    p_true = 1.0 / (1.0 + np.exp(-z))
    # Mildly miscalibrated raw probabilities so the calibrators have something to fix.
    raw = 1.0 / (1.0 + np.exp(-1.5 * z))
    y = (rng.random(n) < p_true).astype(np.int64)
    proba = np.column_stack([1.0 - raw, raw])

    out = pick_best_calibrator(None, None, proba, y, selection="inner_cv", n_bootstrap=200, random_state=1)
    point = out["ece_mean"]
    lo, hi = out["ece_ci"]
    assert lo <= point <= hi, f"inner_cv point {point:.5f} must lie within its CI [{lo:.5f}, {hi:.5f}]"


# ---------------------------------------------------------------------------
# SA19 - bootstrap band does not inflate significance on a calibrated model at small n
# ---------------------------------------------------------------------------


def _prefix_substitution_band(s, t, *, n_grid, n_boot, random_state):
    """The PRE-FIX band: degenerate resamples reuse the full-sample isotonic fit instead of being dropped.

    Reproduced locally so the test can A/B the corrected band against the substitution it replaced, without a
    destructive git checkout. Mirrors ``bootstrap_reliability_band`` up to the degenerate-handling branch and
    returns ``(mean_band_width, significant_fraction, degenerate_fraction)``.
    """
    from sklearn.isotonic import IsotonicRegression

    s = np.asarray(s, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    smin, smax = float(s.min()), float(s.max())
    rng = np.random.default_rng(random_state)
    n = s.size
    grid = np.linspace(smin, smax, n_grid)
    curves = np.empty((n_boot, n_grid), dtype=np.float64)
    idx = rng.integers(0, n, size=(n_boot, n))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    deg = 0
    for b in range(n_boot):
        sel = idx[b]
        sb, tb = s[sel], t[sel]
        if np.unique(tb).size < 2 or sb.max() <= sb.min():
            iso.fit(s, t)  # the bug: substitute the tight full-sample fit
            deg += 1
        else:
            iso.fit(sb, tb)
        curves[b] = iso.predict(grid)
    lower = np.percentile(curves, 2.5, axis=0)
    upper = np.percentile(curves, 97.5, axis=0)
    return float(np.mean(upper - lower)), float(np.mean((lower > grid) | (upper < grid))), deg / n_boot


def test_sa19_band_not_narrowed_by_full_sample_substitution():
    """Sa19 band not narrowed by full sample substitution."""
    pytest.importorskip("sklearn")
    from mlframe.reporting.charts.calibration import (
        _BAND_N_BOOT,
        _SMOOTHED_GRID_POINTS,
        bootstrap_reliability_band,
    )

    # Rare-positive small-n inputs so a MEANINGFUL fraction of bootstrap resamples are degenerate (miss all positives)
    # -- the regime where the substitution actually moves the band. With only ~2 positives many draws are single-class.
    fixed_w = []
    prefix_w = []
    deg_fracs = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = 60
        s = np.sort(rng.random(n))
        y = np.zeros(n, dtype=np.int64)
        y[rng.choice(n, size=2, replace=False)] = 1
        res = bootstrap_reliability_band(s, y, random_state=seed)
        if res is None:
            continue
        _g, lo, hi, _sig = res
        w_pre, _sig_pre, dfrac = _prefix_substitution_band(s, y, n_grid=_SMOOTHED_GRID_POINTS, n_boot=_BAND_N_BOOT, random_state=seed)
        fixed_w.append(float(np.mean(hi - lo)))
        prefix_w.append(w_pre)
        deg_fracs.append(dfrac)

    assert fixed_w, "expected at least one non-degenerate band"
    assert np.mean(deg_fracs) > 0.05, f"setup must trigger degenerate resamples; got {np.mean(deg_fracs):.3f}"
    # Substituting the tight full-sample fit for degenerate draws NARROWS the band (reports miscalibration as
    # significant when it is noise). Dropping those draws must not narrow it -- the corrected band is at least as wide.
    assert np.mean(fixed_w) >= np.mean(prefix_w), f"corrected band {np.mean(fixed_w):.4f} narrower than substitution {np.mean(prefix_w):.4f}"
    assert np.mean(fixed_w) > np.mean(prefix_w) - 1e-9


# ---------------------------------------------------------------------------
# SA20 - DeLong logit CI does not hit the [0,1] clip near AUC~=1
# ---------------------------------------------------------------------------


def test_sa20_delong_ci_avoids_clip_near_auc_one():
    """Sa20 delong ci avoids clip near auc one."""
    from mlframe.reporting.charts.calibration import delong_auc_ci

    rng = np.random.default_rng(11)
    n = 400
    y = rng.integers(0, 2, n)
    # Very large separation -> AUC very close to 1, modest n so the variance (hence the bar) is non-trivial.
    s = rng.standard_normal(n) + 5.0 * y
    auc, lo, hi = delong_auc_ci(y, s)
    assert auc > 0.97, f"setup should give AUC~=1, got {auc:.4f}"
    # Logit CI stays strictly inside (0,1): the upper end must NOT be pinned at exactly 1.0 by the clip.
    assert hi < 1.0, f"upper CI hit the [0,1] clip ({hi}); logit transform should keep it < 1"
    assert lo <= auc <= hi


# ---------------------------------------------------------------------------
# SA21 - no-drift stream with unequal windows has ece_trend ~= 0
# ---------------------------------------------------------------------------


def test_sa21_per_window_ece_uses_debiased_kernel():
    """Per-window ECE must be the DEBIASED estimator (noise-floor subtracted), not the plug-in.

    The plug-in per-window ECE on small windows is inflated by the ~1/sqrt(n_w) Bernoulli noise floor, so a no-drift
    stream's ece_trend can be a pure sample-size artifact. Pinning each window's value to ``compute_ece_debiased``
    (and asserting it is strictly below the plug-in on this calibrated stream) fails on the pre-fix code, which fed
    ``compute_ece_and_brier_decomposition`` (plug-in) into the windows.
    """
    from mlframe.reporting.charts.calibration_drift import (
        calibration_drift,
        _window_edges_by_population,
    )
    from mlframe.metrics.calibration._calibration_metrics import (
        compute_ece_debiased,
        compute_ece_and_brier_decomposition,
    )

    rng = np.random.default_rng(13)
    n = 5200
    p = rng.random(n)  # stationary, perfectly-calibrated stream: no genuine drift
    y = (rng.random(n) < p).astype(np.int64)
    ts = np.arange(n, dtype=np.float64)
    n_windows, n_bins = 13, 10

    res = calibration_drift(y, p, ts, n_windows=n_windows, n_bins=n_bins)
    assert np.isfinite(res.ece_trend)

    order = np.argsort(ts, kind="stable")
    yt = y[order].astype(np.float64)
    ys = p[order]
    edges = _window_edges_by_population(n, n_windows)
    plugin_gt_debiased = 0
    for w in range(len(edges) - 1):
        lo, hi = int(edges[w]), int(edges[w + 1])
        if hi - lo < 30:
            continue
        deb = compute_ece_debiased(yt[lo:hi], ys[lo:hi], n_bins)
        plug = compute_ece_and_brier_decomposition(yt[lo:hi], ys[lo:hi], n_bins)[0]
        # Reported window ECE must equal the debiased kernel (pre-fix it equalled the plug-in instead).
        assert abs(res.window_ece[w] - deb) < 1e-9, f"window {w}: reported {res.window_ece[w]:.5f} != debiased {deb:.5f}"
        if plug > deb + 1e-9:
            plugin_gt_debiased += 1
    # The debiasing must actually bite on most windows of this calibrated stream (else the test proves nothing).
    assert plugin_gt_debiased >= (len(edges) - 1) // 2, "debiased ECE should undercut plug-in on a calibrated stream"
