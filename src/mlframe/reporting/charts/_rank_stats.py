"""Shared rank statistics for the chart builders.

Two panels used to carry their own Spearman: one with AVERAGE-tied ranks, one with ORDINAL ranks. On a
low-cardinality feature (an integer count, a binned score, a one-hot) the ordinal version depends on the ROW ORDER
of tied values, so shuffling the frame changed the number, and the two panels could report different directions for
the same relationship. Average ranks are the definition (and what scipy's ``rankdata`` returns), so that is the one
implementation both sites use now.
"""

from __future__ import annotations

import numpy as np

# Above this length the whole-vector rank+Pearson runs in the njit kernel (single argsort + tie-average in machine
# code) instead of the pure-Python tie-collapse loop below; ~2.1x at N=200k, bit-identical (same average-rank
# convention, so the dispatch cannot change a reported correlation).
SPEARMAN_NJIT_MIN_N = 5_000


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks, 1-based (ties share the mean of their rank span) -- ``scipy.stats.rankdata``'s default method.

    Fully vectorised: sort once, then derive each tied run's average rank from its first and last ordinal position.
    The per-run Python loop this replaces was O(distinct values) at interpreter speed, which is the whole array on
    continuous data.
    """
    x = np.asarray(x)
    n = int(x.shape[0])
    if n == 0:
        return np.empty(0, dtype=np.float64)
    # Within-tie ordering is irrelevant (a tied run collapses to one average rank), so the faster quicksort is fine.
    order = np.argsort(x)
    sorted_x = x[order]
    dense = np.empty(n, dtype=np.intp)
    dense[0] = 0
    if n > 1:
        np.cumsum(sorted_x[1:] != sorted_x[:-1], out=dense[1:])
    counts = np.bincount(dense)
    last_ord = np.cumsum(counts)  # 1-based last ordinal per distinct value
    first_ord = last_ord - counts + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ((first_ord + last_ord) / 2.0)[dense]
    return ranks


def ovr_rank_auc(labels_pos: np.ndarray, proba: np.ndarray, n_classes: int) -> np.ndarray:
    """One-vs-rest AUC per class from tie-averaged rank sums (the Mann-Whitney form), NaN where undefined.

    ``labels_pos`` holds each row's class POSITION (negative = excluded). Shared by the multiclass and multilabel
    overlays, which carried near-identical copies of this loop: same formula, same tie handling, two places to fix.
    """
    out = np.full(n_classes, np.nan, dtype=np.float64)
    n = int(np.asarray(labels_pos).shape[0])
    if n == 0:
        return out
    valid = labels_pos >= 0
    for k in range(n_classes):
        col = proba[:, k]
        finite = np.isfinite(col) & valid
        scores = col[finite]
        pos = labels_pos[finite] == k
        n_pos = int(pos.sum())
        n_neg = int(scores.shape[0] - n_pos)
        if n_pos == 0 or n_neg == 0:
            continue  # a one-vs-rest split with only one class present has no defined AUC
        rank_sum_pos = float(rankdata(scores)[pos].sum())
        out[k] = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return out


def spearman(a: np.ndarray, b: np.ndarray, *, degenerate: float = float("nan")) -> float:
    """Spearman rank correlation via average-tied ranks + Pearson on the ranks; O(n log n), no scipy dependency.

    ``degenerate`` is returned when the correlation is undefined (fewer than 2 rows, or one side constant): the two
    call sites disagree on what to show there -- one wants NaN so the panel can say "no relationship measurable",
    the other wants 0.0 so a verdict string stays printable -- so the caller states it rather than the helper
    guessing.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or a.size != b.size:
        return degenerate
    if a.size >= SPEARMAN_NJIT_MIN_N:
        try:
            from mlframe.metrics.rank_correlation import spearmanr_batched_numba

            val = float(spearmanr_batched_numba(a.reshape(1, -1), b.reshape(1, -1))[0])
            return degenerate if not np.isfinite(val) else val
        except ImportError:
            pass
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom == 0.0:  # one side is constant, so no monotone relationship is defined
        return degenerate
    return float(np.sum(ra * rb) / denom)


__all__ = ["SPEARMAN_NJIT_MIN_N", "ovr_rank_auc", "rankdata", "spearman"]
