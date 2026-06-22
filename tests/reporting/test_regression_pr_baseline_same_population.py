"""Regression: the PR no-skill baseline must use the SAME population as the AP it is compared against.

Pre-fix per-class AP was computed on the stratified subsample but the no-skill prevalence baseline used full-n
(``bin_full / n_valid``), so AP-vs-baseline compared inconsistent populations. Post-fix prevalence is computed on
the finite subsample (``bin_yf.sum() / bin_yf.size``), matching the AP's population.
"""
import numpy as np

from mlframe.reporting.charts.multiclass import _pr_curves_panel


def test_pr_baseline_prevalence_uses_subsample_population():
    rng = np.random.default_rng(0)
    n = 4000
    # Binary one-vs-rest with class 1 rare in full data.
    yt = (rng.random(n) < 0.05).astype(int)
    proba = np.column_stack([1.0 - rng.random(n), rng.random(n)])
    proba /= proba.sum(axis=1, keepdims=True)

    # Force a subsample whose class-1 prevalence differs sharply from full-n (here: a positive-heavy slice),
    # so a full-n baseline would visibly disagree with the subsample-AP population.
    pos_idx = np.flatnonzero(yt == 1)
    neg_idx = np.flatnonzero(yt == 0)[:50]
    sub = np.concatenate([pos_idx, neg_idx])

    panel = _pr_curves_panel(yt, proba, classes=["neg", "pos"], sub=sub)

    # Recompute the expected subsample prevalence for class 1 exactly as the fixed code does.
    yt_s = yt[sub]
    col = proba[sub][:, 1]
    finite = np.isfinite(col)
    bin_yf = (yt_s == 1).astype(np.int8)[finite]
    expected_prevalence = float(int(bin_yf.sum())) / max(1, bin_yf.size)

    full_n_prevalence = float((yt == 1).sum()) / max(1, int((yt >= 0).sum()))
    # The two populations must genuinely differ, else the test proves nothing.
    assert abs(expected_prevalence - full_n_prevalence) > 0.1

    # Baselines are dotted ":" constant series, one per drawn class in draw_idx order [0, 1]; take class 1.
    baseline_series = [
        np.asarray(s) for s, st in zip(panel.y, panel.line_styles) if st == ":"
    ]
    assert len(baseline_series) == 2, len(baseline_series)
    baseline_val = float(baseline_series[1][0])
    assert np.isclose(baseline_val, expected_prevalence), (baseline_val, expected_prevalence)
    assert not np.isclose(baseline_val, full_n_prevalence)
