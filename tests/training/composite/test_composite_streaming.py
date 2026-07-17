"""Regression + biz_value tests for the streaming drift detector hardening
(audit composite_2026_06_10 FUTURE items S7, S8).

S7 -- pure level-shift drift (slope alpha unchanged, intercept beta jumps) must
trigger a refit. The pre-fix detector tested ONLY the alpha z-score, so a
level-shift left every prediction biased by the intercept delta yet never
refit. These tests FAIL on the alpha-only logic.

S8 -- a FIFO rolling buffer mixes pre- and post-drift rows; an OLS over the
WHOLE buffer is biased toward the dead (pre-drift) regime. The change-point-
aware refit fits ONLY the live (post-change) segment and recovers the current
coefficients measurably better than the blended whole-buffer fit. The biz_value
test pins that win; it FAILS on the whole-buffer-only logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import streaming_alpha_check_and_refit
from mlframe.training.composite.streaming import (
    _detect_change_point,
    _ols_alpha_beta_sse,
)


def _brute_best_split(y, base, min_segment_n=30):
    """O(n^2) reference: best split + combined SSE by independent per-segment OLS."""
    n = y.size
    best_k, best_sse = -1, float("inf")
    for k in range(min_segment_n, n - min_segment_n + 1):
        _, _, sl = _ols_alpha_beta_sse(y[:k], base[:k])
        _, _, sr = _ols_alpha_beta_sse(y[k:], base[k:])
        if sl + sr < best_sse:
            best_sse, best_k = sl + sr, k
    return best_k, best_sse


# ===========================================================================
# S7 -- level-shift / intercept drift must refit
# ===========================================================================


class TestLevelShiftDrift:
    def test_pure_level_shift_triggers_refit(self) -> None:
        """Slope unchanged (alpha=1.0), intercept jumps from 0 to 8 -- a pure
        level shift. The alpha-only pre-fix detector NEVER refits this (the
        slope z-score is ~0); the beta z-score must fire it.
        """
        rng = np.random.default_rng(10)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        true_alpha = 1.0
        # Buffer's true intercept is 8.0; deployed intercept is 0.0.
        y = true_alpha * base + 8.0 + rng.normal(scale=0.5, size=n)
        new_alpha, new_beta, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=true_alpha,
            current_beta=0.0,
            z_threshold=3.0,
        )
        assert info["refit"] is True, f"pure level-shift (beta 0 -> 8, alpha stable) must trigger a refit; alpha-only detector would miss it. info={info}"
        assert info["reason"] == "drift_detected_level"
        # The slope was stable -> its z-score stays small...
        assert info["z_alpha"] < 3.0
        # ...while the intercept z-score is what fired the drift.
        assert info["z_beta"] > 3.0
        # The corrected intercept tracks the new regime (~8.0).
        assert abs(new_beta - 8.0) < 0.5

    def test_alpha_only_zscore_would_not_fire_on_level_shift(self) -> None:
        """Direct pin of the pre-fix gap: on a pure level shift the SLOPE
        z-score alone is below threshold, so an alpha-only detector no-ops.
        Our detector must escalate via z_beta instead.
        """
        rng = np.random.default_rng(11)
        n = 600
        base = rng.normal(loc=5.0, scale=1.5, size=n)
        y = 0.7 * base + 12.0 + rng.normal(scale=0.4, size=n)
        _, _, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=0.7,
            current_beta=0.0,
            z_threshold=3.0,
        )
        # Pre-fix behaviour (alpha-only): z_alpha < threshold -> would be
        # "no_drift". Post-fix: z_beta carries the drift.
        assert info["z_alpha"] < 3.0
        assert info["refit"] is True
        assert info["z_score"] == pytest.approx(info["z_beta"], rel=1e-9)

    def test_slope_drift_still_labelled_drift_detected(self) -> None:
        """A genuine slope drift keeps the original 'drift_detected' reason
        (back-compat with the alpha-drift contract)."""
        rng = np.random.default_rng(12)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 2.0 * base + rng.normal(scale=0.5, size=n)
        _, _, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=0.5,
            current_beta=0.0,
            z_threshold=3.0,
        )
        assert info["refit"] is True
        assert info["reason"] == "drift_detected"
        assert info["z_alpha"] > 3.0

    def test_no_drift_when_both_coefs_match(self) -> None:
        """Neither slope nor intercept drift -> no refit (guards against the
        level-shift z_beta spuriously firing on a stationary buffer)."""
        rng = np.random.default_rng(13)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 0.85 * base + 3.14 + rng.normal(scale=0.5, size=n)
        new_alpha, new_beta, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=0.85,
            current_beta=3.14,
            z_threshold=3.0,
        )
        assert info["refit"] is False
        assert info["reason"] == "no_drift"
        assert new_alpha == pytest.approx(0.85, abs=1e-12)
        assert new_beta == pytest.approx(3.14, abs=1e-12)


# ===========================================================================
# S8 -- change-point-aware refit on a mixed FIFO buffer
# ===========================================================================


def _make_mixed_buffer(
    seed: int,
    n_pre: int,
    n_post: int,
    alpha_pre: float,
    beta_pre: float,
    alpha_post: float,
    beta_post: float,
    *,
    noise: float = 0.4,
):
    """Build a FIFO buffer: dead regime first (head), live regime last (tail)."""
    rng = np.random.default_rng(seed)
    base_pre = rng.normal(loc=10.0, scale=2.0, size=n_pre)
    base_post = rng.normal(loc=10.0, scale=2.0, size=n_post)
    y_pre = alpha_pre * base_pre + beta_pre + rng.normal(scale=noise, size=n_pre)
    y_post = alpha_post * base_post + beta_post + rng.normal(scale=noise, size=n_post)
    y = np.concatenate([y_pre, y_post])
    base = np.concatenate([base_pre, base_post])
    return y, base


class TestChangePointDetection:
    def test_detects_break_on_mixed_buffer(self) -> None:
        y, base = _make_mixed_buffer(
            20,
            n_pre=300,
            n_post=200,
            alpha_pre=0.5,
            beta_pre=0.0,
            alpha_post=2.5,
            beta_post=0.0,
        )
        cp = _detect_change_point(y, base)
        assert cp["found"] is True
        # The detected break sits near the true split (index 300).
        assert abs(cp["cp_index"] - 300) <= 30
        assert cp["n_post"] == len(y) - cp["cp_index"]

    def test_no_break_on_homogeneous_buffer(self) -> None:
        """A single-regime buffer must NOT manufacture a break (F gate)."""
        rng = np.random.default_rng(21)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 1.3 * base + 2.0 + rng.normal(scale=0.4, size=n)
        cp = _detect_change_point(y, base)
        assert cp["found"] is False
        assert cp["cp_index"] == -1

    def test_on_scan_matches_brute_force(self) -> None:
        """The O(n) prefix-sum split scan must select the SAME break index and
        report the SAME combined SSE as the O(n^2) per-segment-refit reference
        (numerical equivalence of the perf optimization)."""
        rng = np.random.default_rng(23)
        for s in range(5):
            n = 400 + 40 * s
            base = rng.normal(10.0, 2.0, n)
            y = np.where(
                np.arange(n) < n // 2,
                0.5 * base + 1.0,
                2.5 * base - 3.0,
            ) + rng.normal(0.0, 0.4, n)
            cp = _detect_change_point(y, base)
            bk, bs = _brute_best_split(y, base)
            assert cp["cp_index"] == bk, (s, cp["cp_index"], bk)
            assert cp["sse_split"] == pytest.approx(bs, rel=1e-7, abs=1e-6)

    def test_ols_helper_matches_polyfit(self) -> None:
        """The closed-form OLS helper agrees with numpy polyfit (correctness)."""
        rng = np.random.default_rng(22)
        base = rng.normal(size=400)
        y = 1.7 * base - 0.6 + rng.normal(scale=0.3, size=400)
        alpha, beta, sse = _ols_alpha_beta_sse(y, base)
        p_alpha, p_beta = np.polyfit(base, y, 1)
        assert alpha == pytest.approx(p_alpha, rel=1e-9)
        assert beta == pytest.approx(p_beta, rel=1e-9)
        resid = y - (alpha * base + beta)
        assert sse == pytest.approx(float(np.dot(resid, resid)), rel=1e-9)


class TestChangePointAwareRefit:
    def test_biz_value_change_point_refit_beats_whole_buffer(self) -> None:
        """biz_value (S8): on a FIFO buffer that drifted mid-window, the
        change-point-aware refit recovers the LIVE-regime alpha markedly better
        than the whole-buffer (blended) fit. Pins the measurable win; FAILS on
        the whole-buffer-only logic (detect_change_point=False).

        Setup: 320 dead rows (alpha=0.5) then 200 live rows (alpha=2.5). The
        true live alpha is 2.5. A whole-buffer OLS lands somewhere in between
        (biased toward the larger dead segment).
        """
        n_seeds = 8
        true_post_alpha = 2.5
        errs_cp = []
        errs_full = []
        for s in range(n_seeds):
            y, base = _make_mixed_buffer(
                100 + s,
                n_pre=320,
                n_post=200,
                alpha_pre=0.5,
                beta_pre=0.0,
                alpha_post=true_post_alpha,
                beta_post=0.0,
            )
            a_cp, _, info_cp = streaming_alpha_check_and_refit(
                y,
                base,
                current_alpha=0.5,
                current_beta=0.0,
                z_threshold=3.0,
                detect_change_point=True,
            )
            a_full, _, info_full = streaming_alpha_check_and_refit(
                y,
                base,
                current_alpha=0.5,
                current_beta=0.0,
                z_threshold=3.0,
                detect_change_point=False,
            )
            assert info_cp["refit"] is True
            assert info_cp["change_point"] >= 0, "cp path must have used a break"
            errs_cp.append(abs(a_cp - true_post_alpha))
            errs_full.append(abs(a_full - true_post_alpha))
        mean_cp = float(np.mean(errs_cp))
        mean_full = float(np.mean(errs_full))
        # The change-point refit recovers the live alpha to within tight error;
        # the blended whole-buffer fit is badly biased toward the dead regime.
        assert mean_cp < 0.10, f"cp refit should track live alpha; mean err {mean_cp:.4f}"
        # Measured win is large (blend lands near ~1.5, |err|~1.0 vs cp ~0.03);
        # floor at 5x with margin so a regression that disables the cp path trips.
        assert mean_full > 5.0 * mean_cp, f"change-point refit must beat whole-buffer fit by >5x on the live-alpha error; cp={mean_cp:.4f} full={mean_full:.4f}"

    def test_change_point_disabled_is_legacy_full_buffer(self) -> None:
        """With detect_change_point=False the refit uses the whole buffer and
        reports change_point=-1 (legacy path preserved as an opt-out)."""
        y, base = _make_mixed_buffer(
            200,
            n_pre=300,
            n_post=200,
            alpha_pre=0.5,
            beta_pre=0.0,
            alpha_post=2.5,
            beta_post=0.0,
        )
        a_full, _, info = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=0.5,
            current_beta=0.0,
            detect_change_point=False,
        )
        assert info["change_point"] == -1
        # Whole-buffer fit on a 300/200 dead/live blend lands between regimes.
        assert 0.5 < a_full < 2.5

    def test_homogeneous_buffer_uses_full_window(self) -> None:
        """No break -> change_point=-1 and n_fit == finite buffer size; the fit
        is numerically identical to the legacy whole-buffer path."""
        rng = np.random.default_rng(201)
        n = 500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 2.0 * base + rng.normal(scale=0.4, size=n)
        a_on, b_on, info_on = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=0.5,
            current_beta=0.0,
            detect_change_point=True,
        )
        a_off, b_off, info_off = streaming_alpha_check_and_refit(
            y,
            base,
            current_alpha=0.5,
            current_beta=0.0,
            detect_change_point=False,
        )
        assert info_on["change_point"] == -1
        assert info_on["n_fit"] == n
        # Same fit window (whole buffer) -> bit-identical coefficients.
        assert a_on == pytest.approx(a_off, abs=1e-12)
        assert b_on == pytest.approx(b_off, abs=1e-12)
