"""Unit + biz_value tests for the Winkler interval score (``_winkler``).

The Winkler score is the proper scoring rule that rewards sharpness and penalizes misses (``2/alpha`` per unit
of miss) for a central ``(1 - alpha)`` interval; lower is better. The unit tests pin the score's monotone
decrease as the interval tightens around the truth, the ``1/alpha`` penalty scaling, the coverage helper, and
the degenerate cases; the biz_value test proves a well-calibrated tight interval scores strictly (and
quantitatively) better than both an over-wide interval and an under-covering one on the same data.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite._winkler import (
    interval_quality_summary,
    mean_coverage,
    winkler_interval_score,
    winkler_score_per_group,
    winkler_score_per_row,
)


# --------------------------------------------------------------------------- unit


def test_winkler_decreases_as_interval_tightens_around_truth():
    # y == 0 always covered; score is pure width, so tighter (but still-covering) intervals score lower.
    y = np.zeros(1000)
    scores = [winkler_interval_score(y, np.full(1000, -w), np.full(1000, w), 0.1) for w in (5.0, 2.0, 1.0, 0.5)]
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] == pytest.approx(1.0)  # width 1.0, full coverage, no penalty


def test_miss_penalty_scales_with_inverse_alpha():
    # Every row misses by the same amount; the score minus the (constant) width is (2/alpha)*mean_excess,
    # so halving alpha must double the penalty part.
    y = np.full(500, 10.0)
    lo, hi = np.full(500, -1.0), np.full(500, 1.0)  # width 2, y misses above by 9 each
    width = 2.0
    s_020 = winkler_interval_score(y, lo, hi, 0.20) - width
    s_010 = winkler_interval_score(y, lo, hi, 0.10) - width
    assert s_020 == pytest.approx(0.5 * s_010, rel=1e-9)
    assert s_010 == pytest.approx((2.0 / 0.10) * 9.0, rel=1e-9)


def test_coverage_helper_correct():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    lo = np.array([-1.0, -1.0, 1.0, -1.0])  # rows 0,1,3 cover 0; row 2 does not (lo=1>0)
    hi = np.array([1.0, 1.0, 2.0, 1.0])
    assert mean_coverage(y, lo, hi) == pytest.approx(0.75)


def test_weighted_matches_unweighted_under_uniform_weights():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    lo, hi = y - 1.0 - rng.random(200), y + 1.0 + rng.random(200)
    lo[::7] += 3.0  # inject some misses
    a = winkler_interval_score(y, lo, hi, 0.1)
    b = winkler_interval_score(y, lo, hi, 0.1, sample_weight=np.ones(200))
    assert a == pytest.approx(b, rel=1e-12)
    assert mean_coverage(y, lo, hi) == pytest.approx(mean_coverage(y, lo, hi, sample_weight=np.ones(200)))


def test_per_row_mean_matches_scalar_score():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    lo, hi = y - rng.random(300), y + rng.random(300)
    lo[::5] += 2.0
    per = winkler_score_per_row(y, lo, hi, 0.2)
    assert per.mean() == pytest.approx(winkler_interval_score(y, lo, hi, 0.2), rel=1e-12)


def test_degenerate_zero_width_interval():
    # lo == hi: width 0; covered iff y == point (score 0), else pure penalty 2/alpha * |y - point|.
    y = np.array([0.0, 3.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([0.0, 0.0])
    per = winkler_score_per_row(y, lo, hi, 0.1)
    assert per[0] == pytest.approx(0.0)
    assert per[1] == pytest.approx((2.0 / 0.1) * 3.0)


def test_degenerate_all_miss_dominated_by_penalty():
    y = np.full(100, 100.0)
    lo, hi = np.full(100, -1.0), np.full(100, 1.0)
    s = winkler_interval_score(y, lo, hi, 0.1)
    assert mean_coverage(y, lo, hi) == 0.0
    assert s == pytest.approx(2.0 + (2.0 / 0.1) * 99.0)


def test_per_group_correctness():
    y = np.zeros(4)
    lo = np.array([-1.0, -1.0, -2.0, -2.0])
    hi = np.array([1.0, 1.0, 2.0, 2.0])
    g = np.array(["A", "A", "B", "B"])
    res = winkler_score_per_group(y, lo, hi, 0.1, g)
    assert res["A"]["winkler"] == pytest.approx(2.0)
    assert res["B"]["winkler"] == pytest.approx(4.0)
    assert res["A"]["coverage"] == 1.0 and res["A"]["n"] == 2


def test_alpha_out_of_range_raises():
    y = np.zeros(3)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            winkler_interval_score(y, y - 1, y + 1, bad)


def test_summary_reports_coverage_and_miss_split():
    y = np.array([-5.0, 0.0, 0.0, 5.0])  # rows 0 and 3 miss (below / above)
    lo, hi = np.full(4, -1.0), np.full(4, 1.0)
    s = interval_quality_summary(y, lo, hi, 0.1)
    assert s["target_coverage"] == pytest.approx(0.9)
    assert s["coverage"] == pytest.approx(0.5)
    assert s["below_rate"] == pytest.approx(0.25)
    assert s["above_rate"] == pytest.approx(0.25)
    assert s["miss_rate"] == pytest.approx(0.5)
    assert s["mean_width"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- biz_value


def test_biz_val_winkler_prefers_calibrated_over_wide_and_undercovering():
    rng = np.random.default_rng(42)
    n = 4000
    y = rng.normal(0.0, 1.0, n)
    z90 = 1.6448536269514722  # N(0,1) 0.95 quantile -> a calibrated central 90% interval

    calib_lo, calib_hi = np.full(n, -z90), np.full(n, z90)  # ~90% coverage, sharp
    wide_lo, wide_hi = np.full(n, -4.0), np.full(n, 4.0)  # ~100% coverage, way too wide
    under_lo, under_hi = np.full(n, -0.5), np.full(n, 0.5)  # ~38% coverage, far too narrow

    alpha = 0.10
    s_calib = winkler_interval_score(y, calib_lo, calib_hi, alpha)
    s_wide = winkler_interval_score(y, wide_lo, wide_hi, alpha)
    s_under = winkler_interval_score(y, under_lo, under_hi, alpha)

    # Calibrated is the winner (lowest Winkler), quantitatively -- measured ~0.55x of both competitors.
    assert s_calib < s_wide * 0.75, f"calib {s_calib} not < 0.75*wide {s_wide}"
    assert s_calib < s_under * 0.75, f"calib {s_calib} not < 0.75*under {s_under}"

    # Coverage sanity: calibrated near nominal, wide over-covers, under grossly under-covers.
    assert 0.86 <= mean_coverage(y, calib_lo, calib_hi) <= 0.94
    assert mean_coverage(y, wide_lo, wide_hi) >= 0.999
    assert mean_coverage(y, under_lo, under_hi) <= 0.45

    # The Winkler score correctly rewards the calibrated width while the naive "widest = safest" (max
    # coverage) loses -- proving the score is not just coverage.
    assert s_wide > s_calib and mean_coverage(y, wide_lo, wide_hi) > mean_coverage(y, calib_lo, calib_hi)
