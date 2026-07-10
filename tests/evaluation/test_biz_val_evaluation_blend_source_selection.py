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
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.7])
    result = check_pairwise_score_correlation(scores, scores)
    assert np.isclose(result["spearman_correlation"], 1.0)
    assert result["rank_agreement"] == 1.0
    assert result["trust_source_a"] is True


def test_check_pairwise_score_correlation_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        check_pairwise_score_correlation([0.1, 0.2], [0.1, 0.2, 0.3])
