"""biz_value test for ``evaluation.check_pairwise_score_correlation``.

The win: reproduces the source's exact diagnostic use case (a 5th-place AmEx team trusting public-LB over
CV for blend-weight selection). A noisy, small-sample validation source (analogous to CV) shows LOW rank
correlation with the true (private) per-member quality, correctly flagged untrustworthy, while a low-noise
trusted validation source (analogous to public LB) shows HIGH correlation, correctly flagged trustworthy --
exactly the "correlation check... before finalizing submission choice" gate the source describes.
"""

from __future__ import annotations

import numpy as np

from mlframe.evaluation.blend_source_selection import check_pairwise_score_correlation


def test_biz_val_check_pairwise_score_correlation_flags_noisy_cv_untrustworthy():
    """Check pairwise score correlation flags noisy cv untrustworthy."""
    rng = np.random.default_rng(0)
    n_members = 15
    true_quality = rng.uniform(0.5, 0.6, n_members)

    cv_scores = true_quality + rng.normal(0, 0.25, n_members)  # noisy, small-sample -- analogous to CV
    trusted_scores = true_quality + rng.normal(0, 0.01, n_members)  # low-noise -- analogous to public LB
    private_scores = true_quality + rng.normal(0, 0.01, n_members)  # ground-truth check

    result_cv = check_pairwise_score_correlation(cv_scores, private_scores)
    result_trusted = check_pairwise_score_correlation(trusted_scores, private_scores)

    assert result_cv["trust_source_a"] is False, result_cv
    assert result_trusted["trust_source_a"] is True, result_trusted
    assert result_trusted["spearman_correlation"] > result_cv["spearman_correlation"] + 0.4


def test_check_pairwise_score_correlation_identical_sources_perfect_agreement():
    """Check pairwise score correlation identical sources perfect agreement."""
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.7])
    result = check_pairwise_score_correlation(scores, scores)
    assert np.isclose(result["spearman_correlation"], 1.0)
    assert result["rank_agreement"] == 1.0
    assert result["trust_source_a"] is True


def test_check_pairwise_score_correlation_shape_mismatch_raises():
    """Check pairwise score correlation shape mismatch raises."""
    import pytest

    with pytest.raises(ValueError):
        check_pairwise_score_correlation([0.1, 0.2], [0.1, 0.2, 0.3])


def test_check_pairwise_score_correlation_extra_none_is_bit_identical_to_prior_default():
    """Default (no ``oos_scores_extra``) behavior must be unchanged -- regression-tests the extension."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, 20)
    b = rng.uniform(0, 1, 20)

    baseline = check_pairwise_score_correlation(a, b)
    extended = check_pairwise_score_correlation(a, b, oos_scores_extra=None)

    assert extended.keys() == baseline.keys()
    assert extended["spearman_correlation"] == baseline["spearman_correlation"]
    assert extended["rank_agreement"] == baseline["rank_agreement"]
    assert extended["trust_source_a"] == baseline["trust_source_a"]


def test_check_pairwise_score_correlation_with_extra_preserves_pairwise_ab_result():
    """Passing extra sources must not change the A-vs-B pairwise result itself."""
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, 20)
    b = rng.uniform(0, 1, 20)
    c = rng.uniform(0, 1, 20)

    baseline = check_pairwise_score_correlation(a, b)
    extended = check_pairwise_score_correlation(a, b, oos_scores_extra=[c])

    assert extended["spearman_correlation"] == baseline["spearman_correlation"]
    assert extended["rank_agreement"] == baseline["rank_agreement"]
    assert extended["trust_source_a"] == baseline["trust_source_a"]


def test_biz_val_check_pairwise_score_correlation_multi_source_flags_outlier_fold():
    """The win: N validation folds where one fold is a poorly-correlated outlier is flagged by the
    multi-source summary in a single call, instead of requiring O(N^2) manual pairwise calls."""
    rng = np.random.default_rng(3)
    n_members = 20
    true_quality = rng.uniform(0.4, 0.7, n_members)

    good_folds = [true_quality + rng.normal(0, 0.01, n_members) for _ in range(4)]
    outlier_fold = rng.uniform(0.4, 0.7, n_members)  # unrelated to true_quality -- the outlier

    fold_a, fold_b, *extra = [*good_folds, outlier_fold]
    result = check_pairwise_score_correlation(fold_a, fold_b, oos_scores_extra=extra)

    n_sources = 2 + len(extra)
    assert result["correlation_matrix"].shape == (n_sources, n_sources)
    outlier_idx = n_sources - 1
    assert result["outlier_source_indices"] == [outlier_idx]
    assert all(result["trust_per_source"][i] for i in range(n_sources) if i != outlier_idx)
    assert result["min_pairwise_correlation"] < 0.5
