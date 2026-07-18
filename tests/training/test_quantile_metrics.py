"""Tests for ``mlframe.metrics.quantile``: pinball / coverage /
mean_interval_width / winkler / pit / quantile_summary.

Verifies bit-equivalence with sklearn's ``mean_pinball_loss`` where
applicable; coverage / Winkler / PIT computed on hand-curated arrays
with closed-form expected values.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.quantile import (
    coverage,
    mean_interval_width,
    pinball_loss,
    pinball_loss_per_alpha,
    pit_values,
    quantile_summary,
    winkler_score,
)


@pytest.fixture
def std_normal_data():
    """Std normal data."""
    rng = np.random.default_rng(0)
    n = 500
    y = rng.standard_normal(n)
    # Constant predictions at the theoretical std-normal quantiles.
    preds = np.column_stack(
        [
            np.full(n, -1.2815515655446004),  # q_0.1
            np.zeros(n),  # q_0.5
            np.full(n, 1.2815515655446004),  # q_0.9
        ]
    )
    alphas = (0.1, 0.5, 0.9)
    return y, preds, alphas


class TestPinballLoss:
    """Groups tests covering pinball loss."""
    def test_matches_sklearn(self, std_normal_data):
        """Matches sklearn."""
        from sklearn.metrics import mean_pinball_loss

        y, preds, alphas = std_normal_data
        for j, a in enumerate(alphas):
            mine = pinball_loss(y, preds[:, j], a)
            skl = mean_pinball_loss(y, preds[:, j], alpha=a)
            assert abs(mine - skl) < 1e-12

    def test_per_alpha_dict(self, std_normal_data):
        """Per alpha dict."""
        from sklearn.metrics import mean_pinball_loss

        y, preds, alphas = std_normal_data
        per_a = pinball_loss_per_alpha(y, preds, alphas)
        for j, a in enumerate(alphas):
            assert abs(per_a[a] - mean_pinball_loss(y, preds[:, j], alpha=a)) < 1e-12

    def test_per_alpha_fused_bit_identical_to_per_column(self):
        # The fused row-major kernel must be bit-identical to scoring each alpha
        # on its own column via pinball_loss (the pre-fusion per-column path).
        """Per alpha fused bit identical to per column."""
        rng = np.random.default_rng(7)
        for n, k in ((5000, 9), (50000, 19)):
            y = np.ascontiguousarray(rng.standard_normal(n))
            preds = np.ascontiguousarray(rng.standard_normal((n, k)))
            alphas = list(np.linspace(0.05, 0.95, k))
            per_a = pinball_loss_per_alpha(y, preds, alphas)
            for j, a in enumerate(alphas):
                assert per_a[float(a)] == pinball_loss(y, preds[:, j], a)

    def test_shape_mismatch_raises(self):
        """Shape mismatch raises."""
        with pytest.raises(ValueError, match="shape"):
            pinball_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]), 0.5)

    def test_per_alpha_shape_mismatch_raises(self):
        """Per alpha shape mismatch raises."""
        with pytest.raises(ValueError, match="shape\\[1\\]"):
            pinball_loss_per_alpha(
                np.array([1.0, 2.0]),
                np.array([[1.0], [2.0]]),
                [0.1, 0.5],
            )


class TestCoverage:
    """Groups tests covering coverage."""
    def test_perfect_coverage_unit(self):
        """Perfect coverage unit."""
        y = np.array([1.0, 2.0, 3.0])
        lo = np.array([0.5, 1.5, 2.5])
        hi = np.array([1.5, 2.5, 3.5])
        assert coverage(y, lo, hi) == pytest.approx(1.0)

    def test_zero_coverage(self):
        """Zero coverage."""
        y = np.array([10.0, 20.0])
        lo = np.array([0.0, 0.0])
        hi = np.array([1.0, 1.0])
        assert coverage(y, lo, hi) == pytest.approx(0.0)

    def test_partial_coverage(self):
        """Partial coverage."""
        y = np.array([1.0, 5.0, 10.0])
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([2.0, 4.0, 100.0])
        # row 0: 0 <= 1 <= 2 -> in
        # row 1: 0 <= 5 <= 4 -> out
        # row 2: 0 <= 10 <= 100 -> in
        assert coverage(y, lo, hi) == pytest.approx(2.0 / 3.0)

    def test_empirical_coverage_close_to_nominal(self, std_normal_data):
        """Empirical coverage close to nominal."""
        y, preds, _alphas = std_normal_data
        cov = coverage(y, preds[:, 0], preds[:, 2])
        # Nominal 80% on a fresh std-normal sample; allow 5pp slack.
        assert abs(cov - 0.8) < 0.05


class TestWidth:
    """Groups tests covering width."""
    def test_constant_width(self):
        """Constant width."""
        lo = np.array([0.0, 1.0, 2.0])
        hi = np.array([1.0, 2.0, 3.0])
        assert mean_interval_width(lo, hi) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        """Shape mismatch raises."""
        with pytest.raises(ValueError, match="shape"):
            mean_interval_width(np.array([0.0, 1.0]), np.array([2.0]))


class TestWinkler:
    """Groups tests covering winkler."""
    def test_no_penalty_equals_width(self):
        # All y inside the interval -> Winkler == width.
        """No penalty equals width."""
        y = np.array([0.5, 1.5, 2.5])
        lo = np.array([0.0, 1.0, 2.0])
        hi = np.array([1.0, 2.0, 3.0])
        assert winkler_score(y, lo, hi, alpha_miscov=0.2) == pytest.approx(1.0)

    def test_below_penalty(self):
        # Single row: y=-1, lo=0, hi=1, alpha_miscov=0.2
        # width=1; below penalty = (2/0.2) * (0 - (-1)) = 10 * 1 = 10
        """Below penalty."""
        y = np.array([-1.0])
        lo = np.array([0.0])
        hi = np.array([1.0])
        assert winkler_score(y, lo, hi, alpha_miscov=0.2) == pytest.approx(11.0)

    def test_above_penalty(self):
        """Above penalty."""
        y = np.array([5.0])
        lo = np.array([0.0])
        hi = np.array([1.0])
        # width=1; above penalty = (2/0.2) * (5-1) = 10 * 4 = 40
        assert winkler_score(y, lo, hi, alpha_miscov=0.2) == pytest.approx(41.0)

    def test_invalid_alpha_miscov_rejected(self):
        """Invalid alpha miscov rejected."""
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            winkler_score(np.array([0.5]), np.array([0.0]), np.array([1.0]), alpha_miscov=0.0)


class TestPIT:
    """Groups tests covering p i t."""
    def test_y_at_median_maps_to_0_5(self):
        """Y at median maps to 0 5."""
        y = np.array([0.0])
        preds = np.array([[-1.28, 0.0, 1.28]])
        pit = pit_values(y, preds, (0.1, 0.5, 0.9))
        assert pit[0] == pytest.approx(0.5)

    def test_left_tail_clips_to_alpha_min(self):
        """Left tail clips to alpha min."""
        y = np.array([-100.0])
        preds = np.array([[-1.28, 0.0, 1.28]])
        pit = pit_values(y, preds, (0.1, 0.5, 0.9))
        assert pit[0] == 0.1

    def test_right_tail_clips_to_alpha_max(self):
        """Right tail clips to alpha max."""
        y = np.array([100.0])
        preds = np.array([[-1.28, 0.0, 1.28]])
        pit = pit_values(y, preds, (0.1, 0.5, 0.9))
        assert pit[0] == 0.9

    def test_pit_uniform_for_well_calibrated_model(self):
        # Synthetic well-calibrated: y from std-normal, preds at theoretical
        # quantiles -> PIT should be approximately uniform over [0.1, 0.9].
        """Pit uniform for well calibrated model."""
        rng = np.random.default_rng(0)
        n = 5000
        y = rng.standard_normal(n)
        from scipy.stats import norm

        alphas = np.linspace(0.05, 0.95, 19)
        preds = np.tile(norm.ppf(alphas), (n, 1))
        pit = pit_values(y, preds, alphas)
        # KS test against uniform. Loose threshold (0.08): we want
        # to catch egregious miscalibration (skewed/U-shaped PIT), not
        # statistically-insignificant noise. The clip at the alpha-grid
        # boundaries also adds a sliver of mass at the endpoints that
        # the strict KS budget would catch.
        from scipy.stats import kstest

        ks_stat, _ = kstest(pit, "uniform", args=(alphas[0], alphas[-1] - alphas[0]))
        assert ks_stat < 0.08, f"PIT KS stat {ks_stat:.4f} > 0.08 -- model not well-calibrated"

    def test_pit_robust_to_micro_crossings(self):
        # Tiny crossing in input -> sort-internal handling makes pit_values
        # still produce sensible output (no crash).
        """Pit robust to micro crossings."""
        y = np.array([0.5])
        preds = np.array([[-0.1, 0.0, -0.05]])  # q_0.5 < q_0.9 violated
        pit = pit_values(y, preds, (0.1, 0.5, 0.9))
        # No crash; pit in [0.1, 0.9]
        assert 0.1 <= pit[0] <= 0.9

    def test_pit_njit_kernel_bit_identical_to_numpy_reference(self):
        """The whole-batch njit kernel must reproduce the explicit per-row
        argsort + np.interp reference EXACTLY (bit-for-bit) across distinct,
        tied, and non-monotone rows. A future kernel rewrite that changes the
        sort tie-order or the interp slope formula trips this sensor."""
        rng = np.random.default_rng(7)
        alphas = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

        def reference(y, P, a_arr):
            """Reference."""
            out = np.empty(y.shape[0], dtype=np.float64)
            for i in range(y.shape[0]):
                order = np.argsort(P[i])
                sq = P[i][order]
                sa = a_arr[order]
                if y[i] <= sq[0]:
                    out[i] = float(sa[0])
                elif y[i] >= sq[-1]:
                    out[i] = float(sa[-1])
                else:
                    out[i] = float(np.interp(y[i], sq, sa))
            return np.clip(out, 0.0, 1.0)

        for tied, nonmono in ((False, False), (True, False), (False, True)):
            base = rng.normal(size=4000)
            P = base[:, None] + (alphas - 0.5)[None, :] * 2.0
            if not tied:
                P = P + rng.normal(scale=0.3, size=(4000, len(alphas)))
            if nonmono:
                P[:, 2], P[:, 3] = P[:, 3].copy(), P[:, 2].copy()
            y = base + rng.normal(scale=0.5, size=4000)
            got = pit_values(y, P, alphas)
            ref = reference(np.asarray(y, dtype=np.float64), np.asarray(P, dtype=np.float64), alphas)
            assert np.array_equal(
                got, ref
            ), f"PIT njit kernel diverged from numpy reference (tied={tied}, nonmono={nonmono}); maxdiff={np.max(np.abs(got - ref)):.2e}"


class TestSummary:
    """Groups tests covering summary."""
    def test_summary_shape(self, std_normal_data):
        """Summary shape."""
        y, preds, alphas = std_normal_data
        s = quantile_summary(y, preds, alphas, coverage_pairs=[(0.1, 0.9)])
        assert "pinball_per_alpha" in s
        assert "coverage_0.1_0.9" in s
        assert "mean_width_0.1_0.9" in s
        assert "winkler_0.1_0.9" in s
        # pinball per alpha is a dict keyed by float-alpha
        assert set(s["pinball_per_alpha"]) == set(alphas)

    def test_unknown_coverage_pair_silently_skipped(self, std_normal_data):
        """Unknown coverage pair silently skipped."""
        y, preds, alphas = std_normal_data
        s = quantile_summary(y, preds, alphas, coverage_pairs=[(0.05, 0.95)])
        # Asked for a pair not in alphas -> not in summary.
        assert "coverage_0.05_0.95" not in s
        # Pinball per-alpha still present.
        assert "pinball_per_alpha" in s
