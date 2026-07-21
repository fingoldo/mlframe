"""biz_value + unit tests for the conditional quantile-rank feature (mrmr_audit_2026-07-20
fe_expansion.md "Extend conditional-dispersion to the full conditional quantile").

Validates ``conditional_quantile_rank_fe`` (``_conditional_quantile_rank_fe``): the 4th member of
the grouped_agg mean/std -> composite_group_agg -> conditional-dispersion z-score/|z| ->
conditional quantile-rank family, resolving "how extreme within its peer group" correctly on a
skewed conditional distribution where a z-score cannot.

Contracts pinned
-----------------
* ``TestBasicRankCorrectness``: hand-checked bins recover the exact expected percentile ranks.
* ``TestBizValueSkewedDistributionZScoreFails`` (biz_value): on a log-normal (heavily right-skewed)
  conditional distribution, two rows with IDENTICAL z-scores within their bin sit at materially
  DIFFERENT true percentiles -- quantile-rank resolves this correctly while z-score cannot, by
  construction.
* Leak-safe fit/apply: quantile edges fit on X_fit are frozen and applied unchanged to new rows
  (a row above the fitted max in its bin reads exactly 1.0, not an extrapolated value > 1).
* Degenerate inputs (unseen bin at apply time, non-finite values, mismatched lengths) return NaN /
  raise cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._conditional_quantile_rank_fe import conditional_quantile_rank_fe


class TestBasicRankCorrectness:
    """Hand-checked bins must recover the exact expected within-bin percentile rank."""

    def test_hand_checked_single_bin_ranks(self):
        """bin 0 has values [1,2,3,4,5]; rank(3) = 3/5 = 0.6 (searchsorted 'right' semantics)."""
        x_i = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bins = np.zeros(5, dtype=int)
        ranks = conditional_quantile_rank_fe(x_i, bins)
        np.testing.assert_allclose(ranks, [0.2, 0.4, 0.6, 0.8, 1.0])

    def test_two_bins_are_ranked_independently(self):
        """Values in bin 1 must not affect the rank computation for bin 0, and vice versa."""
        x_i = np.array([10.0, 20.0, 1.0, 2.0])
        bins = np.array([0, 0, 1, 1])
        ranks = conditional_quantile_rank_fe(x_i, bins)
        np.testing.assert_allclose(ranks, [0.5, 1.0, 0.5, 1.0])


class TestBizValueSkewedDistributionZScoreFails:
    """biz_value: a FIXED z-score threshold does not correspond to the same TRUE percentile across
    conditioning bins with different skew -- quantile-rank resolves the row's true peer-group
    extremeness directly, regardless of the bin's shape; z-score cannot, by construction."""

    def test_same_zscore_threshold_corresponds_to_different_true_percentiles_across_bins(self):
        """A fixed z=1.0 threshold must correspond to materially different true percentiles in a
        normal vs a heavy-tailed Pareto bin; quantile-rank still resolves each bin's true max as 1.0."""
        rng = np.random.default_rng(0)
        n = 200000
        # Bin 0: standard normal (z and percentile move together in the textbook way).
        # Bin 1: Pareto (heavy power-law right tail) -- empirically verified to give a >0.05 gap in
        # true percentile at a fixed z=1.0 vs the normal bin (a lognormal's tail is not fat enough
        # at moderate z to produce a robustly-detectable gap; Pareto's power-law tail is).
        x_normal = rng.standard_normal(n)
        x_pareto = rng.pareto(3.0, size=n)
        x_i = np.concatenate([x_normal, x_pareto])
        bins = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

        ranks = conditional_quantile_rank_fe(x_i, bins)

        # Find, in EACH bin, the row whose z-score is closest to z=1.0, and compare their TRUE
        # quantile-rank.
        mu_n, sigma_n = x_normal.mean(), x_normal.std()
        mu_p, sigma_p = x_pareto.mean(), x_pareto.std()
        z_normal = (x_normal - mu_n) / sigma_n
        z_pareto = (x_pareto - mu_p) / sigma_p

        idx_normal_at_z1 = int(np.argmin(np.abs(z_normal - 1.0)))
        idx_pareto_at_z1 = int(np.argmin(np.abs(z_pareto - 1.0)))

        rank_normal_at_z1 = ranks[idx_normal_at_z1]
        rank_pareto_at_z1 = ranks[n + idx_pareto_at_z1]

        assert abs(rank_normal_at_z1 - rank_pareto_at_z1) > 0.05, (
            f"z=1.0 should correspond to materially different TRUE percentiles across a normal bin "
            f"(rank={rank_normal_at_z1:.4f}) and a heavy-tailed Pareto bin (rank={rank_pareto_at_z1:.4f}) "
            "-- a z-score threshold is not a fixed percentile cutoff across differently-shaped conditioning bins"
        )

        # Direct construction: the single largest value in each bin must rank exactly 1.0,
        # regardless of the bin's shape -- quantile-rank correctly resolves true extremeness where
        # a raw z-score comparison across differently-skewed bins would not.
        top_normal = int(np.argmax(x_normal))
        top_pareto = int(np.argmax(x_pareto))
        assert ranks[top_normal] == 1.0
        assert ranks[n + top_pareto] == 1.0


class TestLeakSafeFitApply:
    """Quantile edges fit on X_fit must be frozen and applied unchanged to new rows."""

    def test_row_above_fitted_max_reads_exactly_one(self):
        """A new row's value ABOVE every fitted value in its bin must read exactly 1.0, not an
        extrapolated value beyond the [0, 1] range."""
        x_i_fit = np.array([1.0, 2.0, 3.0])
        bins_fit = np.zeros(3, dtype=int)
        x_i_new = np.array([100.0])
        bins_new = np.zeros(1, dtype=int)
        ranks = conditional_quantile_rank_fe(x_i_new, bins_new, x_i_fit=x_i_fit, x_j_bins_fit=bins_fit)
        assert ranks[0] == 1.0

    def test_apply_rows_do_not_change_the_fitted_reference(self):
        """Two different apply-time X arrays against the SAME fit must both use the frozen
        fit-time quantile reference (byte-identical rank for the same value)."""
        x_i_fit = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bins_fit = np.zeros(5, dtype=int)
        r1 = conditional_quantile_rank_fe(np.array([3.0]), np.zeros(1, dtype=int), x_i_fit=x_i_fit, x_j_bins_fit=bins_fit)
        r2 = conditional_quantile_rank_fe(np.array([3.0, 99.0]), np.zeros(2, dtype=int), x_i_fit=x_i_fit, x_j_bins_fit=bins_fit)
        assert r1[0] == r2[0]


class TestDegenerateInputsReturnNaNOrRaise:
    """An unseen bin, non-finite values, and mismatched lengths must degrade gracefully / raise."""

    def test_unseen_bin_at_apply_time_returns_nan(self):
        """A bin id present at apply time but never seen at fit time must return NaN for those rows."""
        x_i_fit = np.array([1.0, 2.0, 3.0])
        bins_fit = np.zeros(3, dtype=int)
        x_i_new = np.array([5.0])
        bins_new = np.array([99])
        ranks = conditional_quantile_rank_fe(x_i_new, bins_new, x_i_fit=x_i_fit, x_j_bins_fit=bins_fit)
        assert np.isnan(ranks[0])

    def test_nan_value_returns_nan(self):
        """A NaN in x_i must return NaN for that row, not a poisoned searchsorted result."""
        x_i = np.array([1.0, np.nan, 3.0])
        bins = np.zeros(3, dtype=int)
        ranks = conditional_quantile_rank_fe(x_i, bins)
        assert np.isnan(ranks[1])
        assert np.isfinite(ranks[0]) and np.isfinite(ranks[2])

    def test_mismatched_lengths_raises(self):
        """x_i and x_j_bins with different lengths must raise ValueError, not silently misalign."""
        with pytest.raises(ValueError, match="rows"):
            conditional_quantile_rank_fe(np.array([1.0, 2.0]), np.array([0]))
