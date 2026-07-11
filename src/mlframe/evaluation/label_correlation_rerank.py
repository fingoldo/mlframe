"""``detect_correlated_label_pairs``/``label_correlation_rerank``: correlation-aware multi-label top-K rerank.

Source: 2nd_santander-product-recommendation.md -- MAP@7 simulation study shows that when two labels are
near-perfectly correlated (always co-occur), each label's OWN raw predicted probability is a noisier estimate
of "should this be in the top-K" than the PAIR's joint evidence; the source describes analytically deriving a
combined-probability decision threshold and "closing gaps" between correlated labels' predicted ranks.

Rather than re-deriving the source's analytical MAP@K threshold formula (error-prone to generalize correctly
outside MAP@7's exact combinatorics, as the source's own critique flags), this module uses an empirically
safer mechanism with the same effect: detect near-perfectly co-occurring label pairs from TRAIN co-occurrence
statistics, then at inference average each detected pair's predicted scores together -- pulling a correlated
pair's ranks toward each other (their joint evidence), which is exactly what "closing the gap between
correlated labels' predicted ranks" means, without depending on a metric-specific closed-form derivation.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def detect_correlated_label_pairs(y_multilabel: np.ndarray, min_cooccurrence_rate: float = 0.9, min_support: int = 5) -> List[Tuple[int, int]]:
    """Flag label-index pairs that co-occur near-perfectly in a binary multi-label matrix.

    A pair ``(i, j)`` is flagged when, among rows where EITHER label is present, both
    ``P(label_j present | label_i present) >= min_cooccurrence_rate`` and the symmetric converse hold --
    i.e. the two labels are near-interchangeable indicators of the same underlying event.

    Parameters
    ----------
    y_multilabel
        ``(n_samples, n_labels)`` binary (0/1) label matrix.
    min_cooccurrence_rate
        Minimum conditional co-occurrence rate (both directions) to flag a pair.
    min_support
        Minimum number of rows where each label individually is present, to avoid flagging rare-label pairs
        on too little evidence.

    Returns
    -------
    list of (int, int)
        Flagged label-index pairs, ``i < j``.
    """
    y_arr = np.asarray(y_multilabel).astype(np.float64)
    n_labels = y_arr.shape[1]
    counts = y_arr.sum(axis=0)

    # a per-pair Python loop computing (y[:,i] & y[:,j]).sum() separately pays a full-array scan PER PAIR
    # (O(n_labels^2) numpy calls, measured as the dominant cost: 41.6s cProfile total at n_labels=200/50000
    # rows) -- a single matrix multiplication computes every pair's co-occurrence count at once (BLAS-backed,
    # one call), since (y.T @ y)[i, j] is exactly count(label_i AND label_j) for a 0/1 matrix.
    both_counts = y_arr.T @ y_arr

    pairs: List[Tuple[int, int]] = []
    for i in range(n_labels):
        if counts[i] < min_support:
            continue
        for j in range(i + 1, n_labels):
            if counts[j] < min_support:
                continue
            both = both_counts[i, j]
            rate_i_given_j = both / counts[j]
            rate_j_given_i = both / counts[i]
            if rate_i_given_j >= min_cooccurrence_rate and rate_j_given_i >= min_cooccurrence_rate:
                pairs.append((i, j))

    return pairs


def label_correlation_rerank(pred_scores: np.ndarray, correlated_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """Average each detected correlated pair's predicted scores together, per row.

    Parameters
    ----------
    pred_scores
        ``(n_samples, n_labels)`` predicted per-label scores/probabilities.
    correlated_pairs
        Label-index pairs from :func:`detect_correlated_label_pairs` (or supplied directly).

    Returns
    -------
    np.ndarray
        ``(n_samples, n_labels)`` reranked scores: labels in a correlated pair are both replaced by their
        row-wise mean (pulling their ranks together); labels not in any pair are unchanged. If a label
        appears in multiple pairs, the LAST pair processed wins (pairs are typically near-disjoint in
        practice; callers with overlapping pair chains should pre-merge them into groups).
    """
    scores = np.asarray(pred_scores, dtype=np.float64).copy()
    for i, j in correlated_pairs:
        pair_mean = (scores[:, i] + scores[:, j]) / 2.0
        scores[:, i] = pair_mean
        scores[:, j] = pair_mean
    return np.asarray(scores)


__all__ = ["detect_correlated_label_pairs", "label_correlation_rerank"]
