"""biz_value test for ``votenrank.rank_splice.segment_rank_splice``.

The win (2nd_amex-default-prediction.md): a specialist model's predictions for a segment can have an
arbitrarily different score CALIBRATION/SCALE than the main model. Directly substituting the specialist's raw
scores into the segment corrupts the GLOBAL ranking (the segment's rows may all jump to the top or bottom of
the overall distribution regardless of their true relative quality, if the specialist's scale happens to sit
outside the main model's segment range). Rank-splicing keeps every segment row's value from the exact
multiset the main model already assigned there (segment's aggregate position in the global distribution is
untouched) while still using the specialist's better within-segment ordering knowledge -- giving a
materially better GLOBAL AUC than naive raw-score substitution when the specialist is miscalibrated but has
genuine ordering signal.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.votenrank.rank_splice import segment_rank_splice


def _make_miscalibrated_specialist_dataset(n_main: int, n_segment: int, seed: int):
    rng = np.random.default_rng(seed)
    y_main = rng.integers(0, 2, n_main)
    main_scores_main = rng.uniform(0.3, 0.7, n_main) + 0.2 * y_main  # well-calibrated main-model scores

    y_segment = rng.integers(0, 2, n_segment)
    # Main model is nearly random for this segment (it's the data-sparse segment it's weak on) but still
    # occupies the SAME calibrated scale as the rest of the population.
    main_scores_segment = rng.uniform(0.3, 0.7, n_segment) + 0.02 * y_segment

    # Specialist has genuine ordering signal for the segment but is WILDLY miscalibrated in scale (its raw
    # scores sit far outside [0, 1], e.g. an unnormalized decision-function output).
    specialist_scores = 1000.0 + 50.0 * y_segment + rng.normal(scale=5.0, size=n_segment)

    main_scores = np.concatenate([main_scores_main, main_scores_segment])
    y = np.concatenate([y_main, y_segment])
    segment_mask = np.concatenate([np.zeros(n_main, dtype=bool), np.ones(n_segment, dtype=bool)])
    return main_scores, specialist_scores, y, segment_mask


def test_biz_val_rank_splice_beats_raw_substitution_under_scale_mismatch():
    main_scores, specialist_scores, y, segment_mask = _make_miscalibrated_specialist_dataset(n_main=800, n_segment=200, seed=0)

    naive_scores = main_scores.copy()
    naive_scores[segment_mask] = specialist_scores  # naive raw substitution
    auc_naive = roc_auc_score(y, naive_scores)

    spliced_scores = segment_rank_splice(main_scores, specialist_scores, segment_mask)
    auc_spliced = roc_auc_score(y, spliced_scores)

    assert auc_spliced > auc_naive + 0.1, f"expected rank-splicing to materially beat naive raw-score substitution under scale mismatch, got spliced={auc_spliced:.4f} naive={auc_naive:.4f}"
    assert auc_spliced > 0.7, f"expected rank-splicing to recover strong global AUC despite the specialist's miscalibrated scale, got {auc_spliced:.4f}"


def test_segment_rank_splice_preserves_non_segment_rows_exactly():
    main_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    specialist_scores = np.array([9.0, 1.0])
    segment_mask = np.array([False, True, False, True, False])

    result = segment_rank_splice(main_scores, specialist_scores, segment_mask)
    np.testing.assert_allclose(result[~segment_mask], main_scores[~segment_mask])


def test_segment_rank_splice_preserves_segment_value_multiset():
    main_scores = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
    specialist_scores = np.array([0.1, 0.9, 0.5])
    segment_mask = np.array([True, False, True, False, True])

    result = segment_rank_splice(main_scores, specialist_scores, segment_mask)
    np.testing.assert_allclose(np.sort(result[segment_mask]), np.sort(main_scores[segment_mask]))
    # Segment rows are [0, 2, 4] with main values [1.0, 3.0, 2.0] (sorted: [1.0, 2.0, 3.0]) and specialist
    # scores [0.1, 0.9, 0.5] (within-segment rank order: row0=rank0, row4=rank1, row2=rank2) -- so row0 gets
    # the smallest sorted value (1.0), row4 the middle (2.0), row2 the largest (3.0).
    np.testing.assert_allclose(result[[0, 4, 2]], [1.0, 2.0, 3.0])


def _make_dual_noisy_segment_dataset(n_segment: int, seed: int):
    rng = np.random.default_rng(seed)
    y_segment = rng.integers(0, 2, n_segment)
    # Both the main model and the specialist have genuine but individually-noisy signal on this segment --
    # neither one alone is a strong ranker, but their rank ESTIMATES are independently noisy around the same
    # true order, so averaging the two rank positions should reduce variance (an ensembling effect) versus
    # trusting either one alone.
    main_scores_segment = y_segment.astype(np.float64) + rng.normal(scale=1.8, size=n_segment)
    specialist_scores = y_segment.astype(np.float64) + rng.normal(scale=1.8, size=n_segment)
    return main_scores_segment, specialist_scores, y_segment


def test_biz_val_rank_splice_soft_blend_beats_hard_cutover_under_noisy_specialist():
    main_scores, specialist_scores, y = _make_dual_noisy_segment_dataset(n_segment=400, seed=1)
    segment_mask = np.ones_like(y, dtype=bool)  # whole population is the segment for this isolated comparison

    hard_cutover = segment_rank_splice(main_scores, specialist_scores, segment_mask)  # blend_weight=0.0 default
    auc_hard = roc_auc_score(y, hard_cutover)

    soft_blend = segment_rank_splice(main_scores, specialist_scores, segment_mask, blend_weight=0.5)
    auc_blend = roc_auc_score(y, soft_blend)

    assert auc_blend > auc_hard + 0.02, f"expected soft-blend to reduce rank noise vs hard specialist-only cutover, got blend={auc_blend:.4f} hard={auc_hard:.4f}"
    assert auc_blend > 0.63, f"expected soft-blend to reach a materially better AUC than either noisy single source, got {auc_blend:.4f}"


def test_segment_rank_splice_blend_weight_zero_is_bit_identical_to_default():
    main_scores, specialist_scores, y, segment_mask = _make_miscalibrated_specialist_dataset(n_main=800, n_segment=200, seed=0)
    default_result = segment_rank_splice(main_scores, specialist_scores, segment_mask)
    explicit_zero_result = segment_rank_splice(main_scores, specialist_scores, segment_mask, blend_weight=0.0)
    np.testing.assert_array_equal(default_result, explicit_zero_result)


def test_segment_rank_splice_blend_weight_out_of_range_raises():
    import pytest

    main_scores = np.array([1.0, 2.0, 3.0])
    specialist_scores = np.array([1.0, 2.0, 3.0])
    segment_mask = np.array([True, True, True])
    with pytest.raises(ValueError):
        segment_rank_splice(main_scores, specialist_scores, segment_mask, blend_weight=1.5)
    with pytest.raises(ValueError):
        segment_rank_splice(main_scores, specialist_scores, segment_mask, blend_weight=-0.1)


def test_segment_rank_splice_mismatched_lengths_raises():
    import pytest

    main_scores = np.array([1.0, 2.0, 3.0])
    segment_mask = np.array([True, False, True])
    with pytest.raises(ValueError):
        segment_rank_splice(main_scores, np.array([1.0, 2.0, 3.0, 4.0]), segment_mask)  # neither full-length nor segment-length
