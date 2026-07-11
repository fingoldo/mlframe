"""``varying_size_top_k_subsets``: generate diverse-size feature subsets from a ranked importance list.

Source: 7th_elo-merchant-category-recommendation.md -- "at late stage I use target permutation... finally get
12 different feature sets with number from 200~700" used as 12 base LGBM models for stacking. Distinct from
``FeatureSubsetBaggingEnsemble`` (correlation-cluster-diverse subsets of a FIXED size, for variance
reduction) -- this generates VARYING-SIZE top-k prefixes of an existing importance ranking (e.g. from
permutation-null-importance or MRMR), explicitly for feeding a diverse set of base models into a stacking
ensemble rather than committing to one "best" feature set.

GAP (2026-07-11): the literal top-k prefix has no notion of CROSS-SUBSET diversity. When several
top-ranked features form a highly-correlated cluster (e.g. sensor duplicates, or several engineered
variants of the same latent signal), EVERY size's prefix leads with the same cluster members -- small-``k``
subsets can end up entirely composed of one redundant cluster while missing other, equally-informative,
signal directions altogether. That defeats the whole point of varying-size subsets for stacking diversity:
the base models end up highly correlated with each other (same dominant cluster, same blind spots) instead
of exploring genuinely different feature combinations. ``diversify=True`` below rotates which cluster
member leads in each subset (cluster membership found via a simple pairwise-|correlation| pass over
``data``, the same anchor-greedy idea DCD uses internally, kept self-contained here since DCD needs a fitted
MRMR context this free function does not have) so each subset still respects the importance ranking at the
CLUSTER level but samples a different representative -- broader coverage of the true signal at every size,
and less-correlated base-model predictions for the downstream stacker. Default (``diversify=False``) is
byte-identical to the prior behavior.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

FloatMatrix = Union[np.ndarray, "object"]  # ndarray or a pandas DataFrame with columns named like ranked_features


def _cluster_anchors(ranked_features: Sequence[str], data: FloatMatrix, corr_threshold: float) -> dict:
    """Greedy anchor-based clustering of ``ranked_features`` by pairwise |Pearson corr| on ``data``.

    Walks features in rank order (best-first); a feature joins the first existing cluster whose anchor
    (the cluster's best-ranked member) has ``|corr| >= corr_threshold`` with it, else it starts its own
    singleton cluster. Mirrors the DCD single-pass anchor rule, but self-contained (no fitted-MRMR
    dependency) since this is a free function over an arbitrary ranked list.

    Returns ``{anchor_name: [members in rank order]}``.
    """
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            cols = [c for c in ranked_features if c in data.columns]
            mat = data[cols].to_numpy(dtype=np.float64)
        else:
            cols = list(ranked_features)
            mat = np.asarray(data, dtype=np.float64)
    except ImportError:
        cols = list(ranked_features)
        mat = np.asarray(data, dtype=np.float64)

    n_cols = mat.shape[1] if mat.ndim == 2 else 0
    if n_cols != len(cols) or n_cols == 0:
        # data doesn't line up with the requested features -- every feature is its own cluster (no pruning).
        clusters: dict = {}
        for f in ranked_features:
            clusters[f] = [f]
        return clusters

    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(mat, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    idx_of = {name: i for i, name in enumerate(cols)}

    # Profiled (bench_varying_size_top_k_subsets.py, diversify path): the anchor-assignment loop below is
    # O(n_features x n_anchors) in the worst case (mostly-uncorrelated features -> every feature checks
    # against most existing anchors) and dominates wall time at n_features>=1000 (~0.6ms/call at 200
    # features vs ~650ms/call at 1000). No njit/vectorize rewrite applied: this is a one-shot setup call
    # (stacking pipelines invoke it a handful of times per run, not per-fold/per-iteration), well under the
    # "<100 calls/fit" threshold where the numerical-kernel acceleration ladder is skippable by convention.
    anchors: List[str] = []
    clusters = {}
    for f in ranked_features:
        if f not in idx_of:
            clusters[f] = [f]
            continue
        fi = idx_of[f]
        placed = False
        for a in anchors:
            ai = idx_of[a]
            if abs(corr[fi, ai]) >= corr_threshold:
                clusters[a].append(f)
                placed = True
                break
        if not placed:
            anchors.append(f)
            clusters[f] = [f]
    return clusters


def _diverse_ranking(ranked_features: Sequence[str], clusters: dict, rotation: int) -> List[str]:
    """Rebuild a full ranking that ROUND-ROBINS across clusters (one member per cluster per round, clusters
    visited in their original best-first order) instead of emitting one cluster as a solid block. A literal
    top-k prefix over the plain ranking exhausts the single dominant cluster before touching any other
    signal direction; round-robin interleaving guarantees even a SMALL prefix covers every cluster, so the
    downstream model sees all the independent signal instead of several near-duplicate views of just one.
    ``rotation`` additionally rotates which member leads within each cluster, so different subset sizes
    (each called with a different rotation) sample different representatives -- diversity ACROSS subsets on
    top of coverage WITHIN each subset."""
    anchors_in_rank_order: List[str] = []
    seen_anchor_for: dict = {}
    for anchor, members in clusters.items():
        for m in members:
            seen_anchor_for[m] = anchor
    for f in ranked_features:
        anchor = seen_anchor_for[f]
        if anchor not in anchors_in_rank_order:
            anchors_in_rank_order.append(anchor)

    rotated_members = {}
    for anchor in anchors_in_rank_order:
        members = clusters[anchor]
        r = rotation % len(members)
        rotated_members[anchor] = members[r:] + members[:r]

    ordered: List[str] = []
    round_idx = 0
    max_len = max(len(m) for m in rotated_members.values())
    while round_idx < max_len:
        for anchor in anchors_in_rank_order:
            members = rotated_members[anchor]
            if round_idx < len(members):
                ordered.append(members[round_idx])
        round_idx += 1
    return ordered


def varying_size_top_k_subsets(
    ranked_features: Sequence[str],
    sizes: Sequence[int],
    *,
    data: Optional[FloatMatrix] = None,
    diversify: bool = False,
    corr_threshold: float = 0.8,
) -> List[List[str]]:
    """Return one top-k prefix of ``ranked_features`` per size in ``sizes``.

    Parameters
    ----------
    ranked_features
        Feature names ordered best-first (e.g. by permutation-null-importance or MRMR MI-gain).
    sizes
        Subset sizes to generate (e.g. ``[200, 300, 500, 700]``); each size is capped at
        ``len(ranked_features)``.
    data
        Optional ``(n_rows, n_features)`` array or DataFrame used only when ``diversify=True`` to find
        correlated feature clusters. A DataFrame is indexed by column name against ``ranked_features``; a
        bare ndarray is assumed column-aligned with ``ranked_features`` in the same order. Ignored when
        ``diversify=False``.
    diversify
        Opt-in (default ``False`` -- byte-identical to the prior literal-top-k behavior). When ``True``,
        each subset rotates which member of a correlated cluster leads, so different sizes sample
        different representatives of the same signal cluster instead of all leading with the same one --
        broader coverage per subset and less-correlated base models for a downstream stacker. Requires
        ``data``; raises ``ValueError`` if omitted.
    corr_threshold
        Absolute-Pearson-correlation threshold (in ``[0, 1]``) above which two features are judged the
        same cluster. Only used when ``diversify=True``.

    Returns
    -------
    list of list of str
        One feature-name list per requested size, in the same order as ``sizes``.
    """
    n = len(ranked_features)
    if not diversify:
        return [list(ranked_features[: min(size, n)]) for size in sizes]

    if data is None:
        raise ValueError("varying_size_top_k_subsets(diversify=True) requires `data` to find correlated feature clusters")

    clusters = _cluster_anchors(ranked_features, data, corr_threshold)
    subsets = []
    for rotation, size in enumerate(sizes):
        ordered = _diverse_ranking(ranked_features, clusters, rotation)
        subsets.append(ordered[: min(size, n)])
    return subsets


__all__ = ["varying_size_top_k_subsets"]
