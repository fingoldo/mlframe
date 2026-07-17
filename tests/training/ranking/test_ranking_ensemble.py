"""Unit tests for ``ensemble_ranker_scores`` (RRF / Borda / score_mean).

RRF is scale-invariant — verifies the function correctly handles models
with wildly different score ranges (CB ~[0,1], XGB ~[-10,+10], LGB
~[-0.5,+0.5]). Borda is also rank-only. score_mean requires explicit
``assume_comparable_scales=True`` else WARN + RRF fallback.
"""

from __future__ import annotations


import numpy as np
import pytest

from mlframe.training.ranking import ensemble_ranker_scores


@pytest.fixture
def three_aligned_models():
    """3 models, 5 queries × 4 docs = 20 rows. All emit different score
    scales but agree on the same ranking within each query."""
    rng = np.random.default_rng(42)
    n_q = 5
    n_per = 4
    n = n_q * n_per
    gids = np.repeat(np.arange(n_q), n_per)
    # Identical "true" ranking per query: doc 0 best, doc 3 worst.
    base_rank = np.tile(np.array([3, 2, 1, 0]), n_q).astype(float)
    # Each model emits a different MONOTONE transform of base_rank.
    cb_scores = base_rank * 0.1 + rng.normal(0, 0.001, n)  # [0, 0.3] range
    xgb_scores = base_rank * 5.0 + rng.normal(0, 0.05, n)  # [0, 15] range
    lgb_scores = base_rank * 0.01 + rng.normal(0, 0.0001, n)  # tiny range
    return [cb_scores, xgb_scores, lgb_scores], gids


class TestRRF:
    """Reciprocal Rank Fusion."""

    def test_aligned_models_keep_ranking(self, three_aligned_models):
        """All 3 models rank identically within each query -> ensemble
        preserves that ranking, regardless of score scale."""
        scores_per_model, gids = three_aligned_models
        out = ensemble_ranker_scores(scores_per_model, gids, method="rrf")
        # For each query, ensemble score should be monotone-decreasing
        # in the original "true" order (doc 0 best, doc 3 worst).
        for q in np.unique(gids):
            mask = gids == q
            ens_q = out[mask]
            # Rank within query: argmax should be at position 0.
            assert np.argmax(ens_q) == 0, f"query {q}: top-1 is not doc 0"

    def test_rrf_score_for_top_item_matches_formula(self):
        """Single query, all 3 models put item 0 at rank 1.
        RRF score for item 0 = 3 * 1/(60+1) = 3/61 ≈ 0.04918.
        """
        gids = np.zeros(3, dtype=int)
        # Each model: item 0 best, item 1 mid, item 2 worst.
        scores_per_model = [
            np.array([1.0, 0.5, 0.1]),
            np.array([0.99, 0.5, 0.01]),
            np.array([10.0, 0.0, -10.0]),
        ]
        out = ensemble_ranker_scores(scores_per_model, gids, method="rrf", rrf_k=60)
        expected_top = 3 * (1.0 / (60 + 1))
        assert out[0] == pytest.approx(expected_top)

    def test_rrf_tied_items_get_equal_ranks(self):
        """Canonical RRF gives genuinely TIED items EQUAL (averaged) ranks.

        The prior dense-positional scheme broke ties by array index, so two
        rows with identical member scores received DIFFERENT fused mass --
        position-dependent fusion noise. With average-rank tie handling, rows
        that tie across every member must get bit-identical fused scores.
        """
        gids = np.zeros(4, dtype=int)
        # Rows 1 and 2 tie (0.5) in BOTH members -> must get identical output.
        scores_per_model = [
            np.array([0.9, 0.5, 0.5, 0.1]),
            np.array([0.8, 0.5, 0.5, 0.2]),
        ]
        out = ensemble_ranker_scores(scores_per_model, gids, method="rrf", rrf_k=60)
        assert out[1] == pytest.approx(out[2])
        # Averaged rank for the tied pair is (2+3)/2 = 2.5 -> 2 * 1/(60+2.5).
        assert out[1] == pytest.approx(2 * (1.0 / (60 + 2.5)))

    def test_rrf_invariant_to_monotone_transform(self, three_aligned_models):
        """Apply softmax / log to one model -> RRF result identical."""
        scores_per_model, gids = three_aligned_models
        out_baseline = ensemble_ranker_scores(scores_per_model, gids, method="rrf")
        # Apply np.tanh to model 1 -- monotone, preserves rank.
        scores_transformed = list(scores_per_model)
        scores_transformed[1] = np.tanh(scores_transformed[1])
        out_transformed = ensemble_ranker_scores(scores_transformed, gids, method="rrf")
        # Per-row ensemble score may differ in absolute value (tanh changes
        # the rank slightly only if there are tie-breaks)... but the
        # PER-QUERY ORDERING must match. Argsort within each query.
        for q in np.unique(gids):
            mask = gids == q
            np.testing.assert_array_equal(
                np.argsort(-out_baseline[mask]),
                np.argsort(-out_transformed[mask]),
                err_msg=f"query {q}: rank order changed under monotone transform",
            )


class TestBorda:
    """Borda count -- per-query rank averaging, lower=better, output negated."""

    def test_aligned_models_preserve_top_doc(self, three_aligned_models):
        scores_per_model, gids = three_aligned_models
        out = ensemble_ranker_scores(scores_per_model, gids, method="borda")
        for q in np.unique(gids):
            mask = gids == q
            ens_q = out[mask]
            assert np.argmax(ens_q) == 0

    def test_borda_score_increases_with_better_rank(self):
        """3 models, item 0 always rank 1, item 1 always rank 2, etc.
        Borda raw sum = N_models * rank. Output is negated so item 0
        (lowest sum) becomes highest score.
        """
        gids = np.zeros(3, dtype=int)
        scores_per_model = [
            np.array([3.0, 2.0, 1.0]),
            np.array([10.0, 5.0, 0.0]),
            np.array([0.5, 0.3, 0.1]),
        ]
        out = ensemble_ranker_scores(scores_per_model, gids, method="borda")
        # Item 0: rank 1 from each -> sum 3 -> negated -3
        # Item 1: rank 2 -> sum 6 -> negated -6
        # Item 2: rank 3 -> sum 9 -> negated -9
        np.testing.assert_array_equal(out, np.array([-3.0, -6.0, -9.0]))


class TestScoreMean:
    """score_mean requires assume_comparable_scales=True; otherwise hard-fail.

    Audit C-P1-4 (2026): hard-fail replaces the previous silent fallback to
    RRF. Operators previously saw 'score_mean' in config + metadata while
    RRF math actually executed; method choice is a contract, not a
    suggestion. Calibrate scores externally and pass the flag, OR pick
    rrf/borda explicitly.
    """

    def test_score_mean_without_flag_raises(self, three_aligned_models):
        scores_per_model, gids = three_aligned_models
        with pytest.raises(ValueError, match="assume_comparable_scales"):
            ensemble_ranker_scores(scores_per_model, gids, method="score_mean")

    def test_score_mean_with_flag_returns_arithmetic_mean(self, three_aligned_models):
        scores_per_model, gids = three_aligned_models
        out = ensemble_ranker_scores(
            scores_per_model,
            gids,
            method="score_mean",
            assume_comparable_scales=True,
        )
        expected = np.mean(np.asarray(scores_per_model), axis=0)
        np.testing.assert_allclose(out, expected)


class TestValidation:
    """Input shape validation."""

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ensemble_ranker_scores([], np.array([0, 0]), method="rrf")

    def test_score_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            ensemble_ranker_scores(
                [np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])],
                np.array([0, 0, 0]),
                method="rrf",
            )

    def test_groups_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            ensemble_ranker_scores(
                [np.array([1.0, 2.0])],
                np.array([0, 0, 0]),  # too long
                method="rrf",
            )

    def test_unknown_method_raises(self, three_aligned_models):
        scores_per_model, gids = three_aligned_models
        with pytest.raises(ValueError, match="unknown ensemble method"):
            ensemble_ranker_scores(scores_per_model, gids, method="bayes_voting")
