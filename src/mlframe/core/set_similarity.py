"""Set-similarity coefficients on two sets / boolean masks (PZAD err_multirankcluster).

The multiclass/ranking/clustering lecture (Дьяконов 2020, slides 34-35) surveys the family of
set-similarity coefficients used when the target is a SET (interval-as-set, recommended-item set,
cluster membership, overlapping-span prediction): Jaccard, Dice/Sørensen, Szymkiewicz-Simpson (overlap),
Braun-Blanquet, Ochiai (set cosine), Kulczynski, plus the asymmetric Tversky index. mlframe has a
multilabel Jaccard over label matrices, but no general two-set coefficient family.

All take either two 1-D boolean masks of equal length OR two Python sets/iterables of hashable items.
Each returns a similarity in ``[0, 1]`` (Tversky in ``[0, 1]`` for the standard alpha, beta >= 0).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "jaccard",
    "dice",
    "overlap",
    "braun_blanquet",
    "ochiai",
    "kulczynski",
    "tversky",
]


def _counts(a, b):
    """Return (|A∩B|, |A|, |B|) for a, b given as boolean masks (equal length) or as sets/iterables."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.dtype == bool or (a_arr.dtype != object and b_arr.dtype == bool):
        if a_arr.shape != b_arr.shape:
            raise ValueError("set_similarity: boolean masks must have the same shape.")
        am = a_arr.astype(bool)
        bm = b_arr.astype(bool)
        inter = float(np.count_nonzero(am & bm))
        return inter, float(np.count_nonzero(am)), float(np.count_nonzero(bm))
    sa, sb = set(a), set(b)
    return float(len(sa & sb)), float(len(sa)), float(len(sb))


def jaccard(a, b) -> float:
    """Jaccard index ``|A∩B| / |A∪B|``. Both empty -> 1.0 (identical)."""
    inter, na, nb = _counts(a, b)
    union = na + nb - inter
    return 1.0 if union == 0.0 else inter / union


def dice(a, b) -> float:
    """Dice / Sørensen coefficient ``2|A∩B| / (|A|+|B|)``. Both empty -> 1.0."""
    inter, na, nb = _counts(a, b)
    denom = na + nb
    return 1.0 if denom == 0.0 else 2.0 * inter / denom


def overlap(a, b) -> float:
    """Szymkiewicz-Simpson overlap coefficient ``|A∩B| / min(|A|,|B|)``; 1.0 when one set contains the other. Empty -> 1.0."""
    inter, na, nb = _counts(a, b)
    m = min(na, nb)
    return 1.0 if m == 0.0 else inter / m


def braun_blanquet(a, b) -> float:
    """Braun-Blanquet coefficient ``|A∩B| / max(|A|,|B|)``. Empty -> 1.0."""
    inter, na, nb = _counts(a, b)
    m = max(na, nb)
    return 1.0 if m == 0.0 else inter / m


def ochiai(a, b) -> float:
    """Ochiai coefficient (set cosine) ``|A∩B| / sqrt(|A|·|B|)``. Empty -> 1.0."""
    inter, na, nb = _counts(a, b)
    denom = na * nb
    return 1.0 if denom == 0.0 else inter / np.sqrt(denom)


def kulczynski(a, b) -> float:
    """Kulczynski coefficient ``(|A∩B|/2)·(1/|A| + 1/|B|)`` = mean of the two inclusion ratios. Empty -> 1.0."""
    inter, na, nb = _counts(a, b)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return 0.5 * inter * (1.0 / na + 1.0 / nb)


def tversky(a, b, *, alpha: float = 0.5, beta: float = 0.5) -> float:
    """Asymmetric Tversky index ``|A∩B| / (|A∩B| + alpha·|A\\B| + beta·|B\\A|)``.

    Generalizes Jaccard (alpha=beta=1) and Dice (alpha=beta=0.5); asymmetric weights let ``a`` (prediction) and
    ``b`` (reference) contribute false-positives / false-negatives differently. ``alpha, beta >= 0``.
    """
    if alpha < 0 or beta < 0:
        raise ValueError("tversky: alpha and beta must be >= 0.")
    inter, na, nb = _counts(a, b)
    a_only = na - inter
    b_only = nb - inter
    denom = inter + alpha * a_only + beta * b_only
    return 1.0 if denom == 0.0 else inter / denom
