"""``varying_size_top_k_subsets``: generate diverse-size feature subsets from a ranked importance list.

Source: 7th_elo-merchant-category-recommendation.md -- "at late stage I use target permutation... finally get
12 different feature sets with number from 200~700" used as 12 base LGBM models for stacking. Distinct from
``FeatureSubsetBaggingEnsemble`` (correlation-cluster-diverse subsets of a FIXED size, for variance
reduction) -- this generates VARYING-SIZE top-k prefixes of an existing importance ranking (e.g. from
permutation-null-importance or MRMR), explicitly for feeding a diverse set of base models into a stacking
ensemble rather than committing to one "best" feature set.
"""
from __future__ import annotations

from typing import List, Sequence


def varying_size_top_k_subsets(ranked_features: Sequence[str], sizes: Sequence[int]) -> List[List[str]]:
    """Return one top-k prefix of ``ranked_features`` per size in ``sizes``.

    Parameters
    ----------
    ranked_features
        Feature names ordered best-first (e.g. by permutation-null-importance or MRMR MI-gain).
    sizes
        Subset sizes to generate (e.g. ``[200, 300, 500, 700]``); each size is capped at
        ``len(ranked_features)``.

    Returns
    -------
    list of list of str
        One feature-name list per requested size, in the same order as ``sizes``.
    """
    n = len(ranked_features)
    return [list(ranked_features[: min(size, n)]) for size in sizes]


__all__ = ["varying_size_top_k_subsets"]
