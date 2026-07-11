"""Pick which validation source to trust for blend-weight selection, via pairwise per-member score correlation.

Ensemble blend weights fit against CV scores can overfit CV noise -- a 5th-place AmEx-default-prediction team
found public-LB-weighted blending generalized better than CV-weighted blending for their setup, and used a
correlation check between per-member CV and LB scores as a sanity gate before finalizing weights. This is a
general pattern beyond Kaggle terminology: given per-ensemble-member scores from two candidate validation
sources (e.g. internal CV vs. a large untouched holdout), check how well they RANK-AGREE across members --
low agreement is itself the signal that one source (usually the noisier/smaller one) shouldn't be trusted for
weight selection. Composes with the existing `compare_cv_schemes` (which validation SCHEME best tracks a
ground-truth score) and `constrained_weight_blend` (optimizes weights against whichever score source is
chosen) rather than reimplementing either.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr


def check_pairwise_score_correlation(
    oos_scores_a: npt.ArrayLike, oos_scores_b: npt.ArrayLike, oos_scores_extra: Optional[Sequence[npt.ArrayLike]] = None
) -> dict:
    """Rank-correlation between two validation sources' per-ensemble-member scores.

    Parameters
    ----------
    oos_scores_a, oos_scores_b
        ``(n_members,)`` scores for the SAME set of ensemble members/candidates, one array per validation
        source (e.g. CV score and a trusted-holdout score for each candidate blend weight/model).
    oos_scores_extra
        Optional additional validation sources beyond A and B (e.g. more CV folds/splits), each
        ``(n_members,)`` and aligned to the same members. When given, the pairwise A-vs-B result below is
        UNCHANGED (bit-identical to the 2-source call) and the return dict gains a multi-source summary
        covering all ``2 + len(oos_scores_extra)`` sources at once, so callers don't have to manually call
        this function once per pair to find an outlier source.

    Returns
    -------
    dict
        ``spearman_correlation`` (rank correlation between the two sources across members),
        ``rank_agreement`` (fraction of member pairs whose relative order agrees between the two sources),
        ``trust_source_a`` (bool: ``False`` when correlation is weak -- below 0.5 -- meaning source A's
        per-member ranking doesn't reliably track source B, so weight selection should prefer source B (or
        neither) rather than blindly trusting A).

        When ``oos_scores_extra`` is given, also:
        ``correlation_matrix`` (``(n_sources, n_sources)`` Spearman correlation between every pair of
        sources, in the order [A, B, *extra]), ``min_pairwise_correlation`` (the weakest correlation across
        all pairs -- the overall trust bottleneck), ``trust_per_source`` (list of bool, one per source:
        ``False`` when that source's MEAN correlation with every other source is below 0.5, i.e. it doesn't
        rank-agree with the rest and is likely the outlier -- mean rather than min so one lone outlier
        doesn't also drag down every other source's flag), ``outlier_source_indices`` (indices where
        ``trust_per_source`` is ``False``).
    """
    a = np.asarray(oos_scores_a, dtype=np.float64)
    b = np.asarray(oos_scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("check_pairwise_score_correlation: oos_scores_a and oos_scores_b must have the same shape")
    if a.shape[0] < 2:
        raise ValueError("check_pairwise_score_correlation: need at least 2 members to compute a rank correlation")

    corr, _p = spearmanr(a, b)
    corr = float(corr) if np.isfinite(corr) else 0.0

    n = a.shape[0]
    rank_a, rank_b = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    # vectorized pairwise-comparison matrices instead of an O(n^2) Python double loop (44x faster at n=500,
    # bit-identical) -- broadcast each rank vector against itself to get every pair's ">" relation in one
    # pass, then compare the two boolean matrices only on the upper triangle (each unordered pair once).
    a_gt = rank_a[:, None] > rank_a[None, :]
    b_gt = rank_b[:, None] > rank_b[None, :]
    iu = np.triu_indices(n, k=1)
    total = iu[0].shape[0]
    rank_agreement = float(np.sum(a_gt[iu] == b_gt[iu])) / total if total > 0 else 1.0

    result = {
        "spearman_correlation": corr,
        "rank_agreement": rank_agreement,
        "trust_source_a": corr >= 0.5,
    }

    if oos_scores_extra is not None:
        sources = [a, b] + [np.asarray(s, dtype=np.float64) for s in oos_scores_extra]
        n_sources = len(sources)
        for s in sources:
            if s.shape != a.shape:
                raise ValueError("check_pairwise_score_correlation: every source in oos_scores_extra must have the same shape as oos_scores_a/b")

        stacked = np.vstack(sources)  # (n_sources, n_members)
        corr_matrix, _p_matrix = spearmanr(stacked, axis=1)
        corr_matrix = np.atleast_2d(np.asarray(corr_matrix, dtype=np.float64))
        corr_matrix = np.where(np.isfinite(corr_matrix), corr_matrix, 0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        iu2 = np.triu_indices(n_sources, k=1)
        min_pairwise_correlation = float(np.min(corr_matrix[iu2])) if iu2[0].shape[0] > 0 else 1.0

        off_diag_mask = ~np.eye(n_sources, dtype=bool)
        # mean (not min) pairwise correlation vs. every other source: a lone outlier drags down everyone's
        # min equally, but only the outlier itself has a low MEAN across all its pairs -- the rest keep a
        # high mean because most of their pairs are with each other, not with the outlier.
        trust_per_source = [bool(np.mean(corr_matrix[i][off_diag_mask[i]]) >= 0.5) for i in range(n_sources)]
        outlier_source_indices = [i for i, trusted in enumerate(trust_per_source) if not trusted]

        result["correlation_matrix"] = corr_matrix
        result["min_pairwise_correlation"] = min_pairwise_correlation
        result["trust_per_source"] = trust_per_source
        result["outlier_source_indices"] = outlier_source_indices

    return result


__all__ = ["check_pairwise_score_correlation"]
