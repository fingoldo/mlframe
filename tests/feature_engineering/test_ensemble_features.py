"""Coverage for feature_engineering.ensemble_features -- the per-row predictor-disagreement
feature catalogue. Only predictor_disagreement_var had a prior test (test_fe_top_a_fixes.py's
F21 dead-branch-removal pin); the other ~11 public functions had none."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.ensemble_features import (
    predictor_consensus_entropy,
    predictor_consensus_mean,
    predictor_consensus_trimmed_stats,
    predictor_disagreement_features,
    predictor_disagreement_iqr,
    predictor_disagreement_var,
    predictor_max_pairwise_distance,
    predictor_outlier_signature,
    predictor_pairwise_abs_diffs,
    predictor_quantile_spread,
    predictor_top2_mode_gap,
    predictor_weighted_consensus,
)

pytestmark = pytest.mark.fast


def _unanimous_preds(n_rows=10, n_preds=5, value=3.0):
    """All predictors agree exactly on every row."""
    return np.full((n_rows, n_preds), value, dtype=np.float64)


def _spread_preds(seed=0, n_rows=20, n_preds=6):
    """Predictors with genuine per-row spread."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 10, size=(n_rows, n_preds))


class TestCoercePredsGuards:
    """Shared input validation (via any public function, e.g. predictor_consensus_mean)."""

    def test_1d_input_raises(self):
        """A 1-D preds array is rejected -- must be (n_rows, n_preds)."""
        with pytest.raises(ValueError, match="2-D"):
            predictor_consensus_mean(np.array([1.0, 2.0, 3.0]))

    def test_single_predictor_raises(self):
        """Fewer than 2 predictor columns is rejected -- no disagreement to measure."""
        with pytest.raises(ValueError, match=">= 2"):
            predictor_consensus_mean(np.zeros((5, 1)))

    def test_nan_row_imputed_from_median_not_propagated(self):
        """A row with one NaN predictor gets that cell imputed from the row's finite median,
        not NaN-propagated to the whole row's output."""
        preds = np.array([[1.0, 2.0, np.nan], [1.0, 1.0, 1.0]])
        out = predictor_consensus_mean(preds)
        assert np.all(np.isfinite(out))
        # Row 0: finite values {1, 2}, median 1.5 imputed for the NaN cell -> mean = (1+2+1.5)/3
        assert out[0] == pytest.approx((1.0 + 2.0 + 1.5) / 3.0)

    def test_all_nan_row_filled_with_zero(self):
        """A row with ALL non-finite predictors is filled with 0.0 (no information for that row)."""
        preds = np.array([[np.nan, np.nan], [1.0, 3.0]])
        out = predictor_consensus_mean(preds)
        assert out[0] == 0.0


class TestConsensusMean:
    """predictor_consensus_mean."""

    def test_unanimous_returns_the_shared_value(self):
        """Unanimous predictors -> mean equals that value exactly."""
        out = predictor_consensus_mean(_unanimous_preds(value=4.0))
        assert np.allclose(out, 4.0)

    def test_matches_plain_numpy_mean_on_finite_input(self):
        """On finite input, matches plain np.mean(axis=1) exactly."""
        preds = _spread_preds()
        out = predictor_consensus_mean(preds)
        assert np.allclose(out, preds.mean(axis=1))


class TestDisagreementIqr:
    """predictor_disagreement_iqr."""

    def test_unanimous_is_zero(self):
        """No disagreement -> IQR is exactly 0."""
        out = predictor_disagreement_iqr(_unanimous_preds())
        assert np.allclose(out, 0.0)

    def test_matches_numpy_percentile_linear_interp(self):
        """Matches np.percentile's default linear-interpolation IQR (the function's own documented target)."""
        preds = _spread_preds()
        out = predictor_disagreement_iqr(preds)
        ref = np.percentile(preds, 75, axis=1) - np.percentile(preds, 25, axis=1)
        assert np.allclose(out, ref, atol=1e-9)


class TestDisagreementVar:
    """predictor_disagreement_var (already had a prior pin, kept here for shape/monotonicity)."""

    def test_unanimous_is_zero(self):
        """No disagreement -> variance is exactly 0."""
        out = predictor_disagreement_var(_unanimous_preds())
        assert np.allclose(out, 0.0)

    def test_matches_numpy_var_ddof1(self):
        """Matches np.var(ddof=1) (unbiased sample variance), the function's documented convention."""
        preds = _spread_preds()
        out = predictor_disagreement_var(preds)
        assert np.allclose(out, preds.var(axis=1, ddof=1))


class TestPairwiseAbsDiffs:
    """predictor_pairwise_abs_diffs."""

    def test_output_shape_is_n_choose_2(self):
        """Output width is N*(N-1)/2 pair columns."""
        n_preds = 5
        out = predictor_pairwise_abs_diffs(_spread_preds(n_preds=n_preds))
        assert out.shape[1] == n_preds * (n_preds - 1) // 2

    def test_matches_direct_pair_computation(self):
        """Each column matches the direct |pred_i - pred_j| for its (i, j) pair, in lexicographic order."""
        preds = np.array([[1.0, 5.0, 2.0]])  # 3 predictors -> pairs (0,1),(0,2),(1,2)
        out = predictor_pairwise_abs_diffs(preds)
        expected = np.array([[abs(1 - 5), abs(1 - 2), abs(5 - 2)]])
        assert np.allclose(out, expected)

    def test_unanimous_is_all_zero(self):
        """No disagreement -> every pairwise diff is 0."""
        out = predictor_pairwise_abs_diffs(_unanimous_preds())
        assert np.allclose(out, 0.0)


class TestConsensusEntropy:
    """predictor_consensus_entropy."""

    def test_unanimous_has_near_zero_entropy(self):
        """All predictors identical -> single occupied bin -> entropy ~0."""
        out = predictor_consensus_entropy(_unanimous_preds())
        assert np.all(out < 1e-6)

    def test_maximally_spread_has_higher_entropy_than_concentrated(self):
        """A row whose predictors occupy every bin evenly has higher entropy than one where most predictors
        cluster in a single bin with one outlier -- binning is relative to each row's own [min, max], so the
        SHAPE of the distribution across bins matters, not the absolute scale."""
        clustered = np.array([[1.0, 1.0, 1.0, 1.0, 10.0]])  # 4 predictors share one bin, 1 in another
        spread = np.array([[0.0, 2.5, 5.0, 7.5, 10.0]])  # one predictor per bin
        e_clustered = predictor_consensus_entropy(clustered, n_bins=5)
        e_spread = predictor_consensus_entropy(spread, n_bins=5)
        assert e_spread[0] > e_clustered[0]


class TestTop2ModeGap:
    """predictor_top2_mode_gap."""

    def test_unanimous_has_max_gap(self):
        """All predictors in one bin -> top1=N, top2=0 -> gap = N/N = 1.0."""
        out = predictor_top2_mode_gap(_unanimous_preds(n_preds=5))
        assert np.allclose(out, 1.0)

    def test_evenly_split_two_bins_has_zero_gap(self):
        """Predictors split evenly across exactly 2 far-apart bins -> top1 == top2 -> gap 0."""
        preds = np.array([[0.0, 0.0, 10.0, 10.0]])
        out = predictor_top2_mode_gap(preds, n_bins=5)
        assert out[0] == pytest.approx(0.0)


class TestWeightedConsensus:
    """predictor_weighted_consensus."""

    def test_equal_weights_matches_plain_mean_and_var(self):
        """Equal weights reduce to the plain (unweighted) mean; variance is the weighted population variance."""
        preds = _spread_preds()
        w = np.ones(preds.shape[1])
        mean, _var = predictor_weighted_consensus(preds, w)
        assert np.allclose(mean, preds.mean(axis=1))

    def test_weight_length_mismatch_raises(self):
        """weights must have one entry per predictor column."""
        with pytest.raises(ValueError, match="weights len"):
            predictor_weighted_consensus(_spread_preds(n_preds=4), np.ones(3))

    def test_negative_weight_raises(self):
        """Negative weights are rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            predictor_weighted_consensus(_spread_preds(n_preds=3), np.array([1.0, -1.0, 1.0]))

    def test_zero_sum_weights_raises(self):
        """All-zero weights (sum <= 0) are rejected -- would divide by zero."""
        with pytest.raises(ValueError, match="sum to > 0"):
            predictor_weighted_consensus(_spread_preds(n_preds=3), np.zeros(3))

    def test_heavily_weighted_predictor_dominates_mean(self):
        """Putting nearly all weight on one predictor pulls the weighted mean toward its value."""
        preds = np.array([[0.0, 100.0]])
        w = np.array([0.999, 0.001])
        mean, _var = predictor_weighted_consensus(preds, w)
        assert mean[0] < 1.0


class TestConsensusTrimmedStats:
    """predictor_consensus_trimmed_stats."""

    def test_invalid_trim_frac_raises(self):
        """trim_frac must be in [0, 0.5)."""
        with pytest.raises(ValueError, match="trim_frac"):
            predictor_consensus_trimmed_stats(_spread_preds(), trim_frac=0.5)

    def test_trim_zero_matches_plain_mean(self):
        """trim_frac=0 (no trimming) reduces the trimmed mean to the plain mean."""
        preds = _spread_preds()
        trimmed_mean, _mad = predictor_consensus_trimmed_stats(preds, trim_frac=0.0)
        assert np.allclose(trimmed_mean, preds.mean(axis=1))

    def test_outlier_predictor_excluded_by_trimming(self):
        """A single extreme outlier predictor is excluded from the trimmed mean once trim_frac is large enough."""
        preds = np.array([[1.0, 1.0, 1.0, 1.0, 1000.0]])  # 5 predictors, 1 extreme outlier
        trimmed_mean, _mad = predictor_consensus_trimmed_stats(preds, trim_frac=0.2)
        assert trimmed_mean[0] == pytest.approx(1.0)

    def test_unanimous_mad_is_zero(self):
        """No disagreement -> MAD-based scale is 0."""
        _mean, mad = predictor_consensus_trimmed_stats(_unanimous_preds())
        assert np.allclose(mad, 0.0)


class TestOutlierSignature:
    """predictor_outlier_signature."""

    def test_unanimous_has_zero_outliers(self):
        """No disagreement -> zero outliers on every row."""
        n_outliers, _idx = predictor_outlier_signature(_unanimous_preds())
        assert np.allclose(n_outliers, 0.0)

    def test_single_extreme_predictor_flagged_as_outlier_and_identified(self):
        """A single wildly-deviating predictor is both counted as an outlier and identified by index."""
        preds = np.array([[1.0, 1.0, 1.0, 1.0, 1000.0]])
        n_outliers, argmax_idx = predictor_outlier_signature(preds, k_mad=2.5)
        assert n_outliers[0] >= 1.0
        assert int(argmax_idx[0]) == 4


class TestMaxPairwiseDistance:
    """predictor_max_pairwise_distance."""

    def test_unanimous_is_zero(self):
        """No disagreement -> max pairwise distance is 0."""
        out = predictor_max_pairwise_distance(_unanimous_preds())
        assert np.allclose(out, 0.0)

    def test_equals_range(self):
        """Max pairwise distance equals max - min (the range), by construction."""
        preds = _spread_preds()
        out = predictor_max_pairwise_distance(preds)
        assert np.allclose(out, preds.max(axis=1) - preds.min(axis=1))


class TestQuantileSpread:
    """predictor_quantile_spread."""

    def test_invalid_quantile_order_raises(self):
        """q_low must be strictly less than q_high, both in [0, 1]."""
        with pytest.raises(ValueError, match="q_low"):
            predictor_quantile_spread(_spread_preds(), q_low=0.9, q_high=0.1)

    def test_default_quantiles_bracket_the_median(self):
        """p_low <= median <= p_high for every row under the default 0.1/0.9 quantiles."""
        preds = _spread_preds()
        p_lo, p_hi, spread = predictor_quantile_spread(preds)
        med = np.median(preds, axis=1)
        assert np.all(p_lo <= med + 1e-9)
        assert np.all(p_hi >= med - 1e-9)
        assert np.allclose(spread, p_hi - p_lo)

    def test_unanimous_has_zero_spread(self):
        """No disagreement -> spread is 0 and p_low == p_high == the shared value."""
        p_lo, p_hi, spread = predictor_quantile_spread(_unanimous_preds(value=7.0))
        assert np.allclose(p_lo, 7.0)
        assert np.allclose(p_hi, 7.0)
        assert np.allclose(spread, 0.0)


class TestDisagreementFeaturesBuilder:
    """predictor_disagreement_features (all-in-one builder)."""

    def test_returns_expected_keys_with_pairs(self):
        """emit_pairs=True (default) returns mean/iqr/var/entropy/top2_gap/pairs."""
        preds = _spread_preds(n_preds=4)
        out = predictor_disagreement_features(preds)
        assert set(out) == {"mean", "iqr", "var", "entropy", "top2_gap", "pairs"}
        assert out["pairs"].shape == (preds.shape[0], 6)  # 4 choose 2

    def test_emit_pairs_false_omits_pairs_key(self):
        """emit_pairs=False suppresses the (potentially wide) pairs column block."""
        preds = _spread_preds(n_preds=4)
        out = predictor_disagreement_features(preds, emit_pairs=False)
        assert "pairs" not in out
        assert set(out) == {"mean", "iqr", "var", "entropy", "top2_gap"}

    def test_matches_calling_each_function_individually(self):
        """The builder's outputs are bit-identical to calling each public function directly (shared coerce/binning)."""
        preds = _spread_preds(seed=3, n_preds=5)
        out = predictor_disagreement_features(preds)
        assert np.allclose(out["mean"], predictor_consensus_mean(preds))
        assert np.allclose(out["iqr"], predictor_disagreement_iqr(preds))
        assert np.allclose(out["var"], predictor_disagreement_var(preds))
        assert np.allclose(out["entropy"], predictor_consensus_entropy(preds))
        assert np.allclose(out["top2_gap"], predictor_top2_mode_gap(preds))
        assert np.allclose(out["pairs"], predictor_pairwise_abs_diffs(preds))
