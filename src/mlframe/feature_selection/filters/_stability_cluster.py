"""Cluster Stability Selection + Complementary Pairs Stability (Wave 8).

Two opt-in stability-selection variants for MRMR:

E11 - **Cluster Stability Selection** (Faletto & Bien 2022, arXiv:2201.00494,
*JMLR* 2024). Pre-clusters highly-correlated features then applies
stability selection at the CLUSTER level; gives selection-frequency error
bounds (Shah-Samworth 2013). Solves the "correlated features split votes"
problem of standard stability selection.

E12 - **Complementary Pairs Stability Selection** (Shah & Samworth 2013,
*JRSS-B* 75(1)). Run selection on B random HALF-SPLITS + their complements;
derive a tight error bound on falsely selected features without
exchangeability assumptions. Tighter than Meinshausen-Buhlmann 2010 bound.

Both are OPT-IN -- not active by default. The MRMR ``stability_selection``
constructor knob (already present per RFECV wave 4 work) is extended to
accept dict-style config:

    stability_selection = {
        'method': 'cluster' | 'complementary_pairs' | 'classic',
        'n_bootstrap': 100,
        'pi_threshold': 0.6,
        'corr_threshold': 0.8,  # cluster only
    }
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np


def cluster_stability_selection(
    X: np.ndarray, y: np.ndarray,
    selector_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    n_bootstrap: int = 100,
    pi_threshold: float = 0.6,
    corr_threshold: float = 0.8,
    rng_seed: int = 0,
    return_clusters: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Faletto-Bien 2022 Cluster Stability Selection.

    Step 1: pre-cluster columns of X by absolute Pearson correlation
    >= ``corr_threshold``. Single-linkage on the hard-thresholded
    similarity graph (single-linkage chains are explicitly intended here
    per Faletto-Bien 2022 sec. 3.1).

    Step 2: for ``b = 1..n_bootstrap``, draw a half-sample of rows, run
    ``selector_fn`` on it, and record per-CLUSTER (not per-feature) selection
    -- a cluster is "selected" if ANY of its members is selected by the
    base selector.

    Step 3: keep clusters whose selection frequency >= ``pi_threshold``.
    Within each kept cluster, return the member with the highest base
    selection frequency.

    Args:
        X: 2-D ndarray of shape (n_samples, n_features).
        y: 1-D target.
        selector_fn: callable ``(X_sub, y_sub) -> ndarray of selected indices``.
        n_bootstrap: number of half-sample bootstrap runs.
        pi_threshold: selection-frequency cutoff for keeping a cluster.
        corr_threshold: |Pearson| cutoff for cluster merging.
        rng_seed: RNG seed.
        return_clusters: if True include the cluster assignments dict in the
            return tuple.

    Returns:
        selected_indices, per_feature_freq, info_dict
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    rng = np.random.default_rng(int(rng_seed))
    # ---- step 1: cluster by |Pearson| >= corr_threshold ----
    if p > 1:
        # Z-standardise then correlation = (1/n) * Z.T @ Z.
        Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
        C = np.abs(Z.T @ Z / n)
    else:
        C = np.ones((1, 1))
    cluster_id = np.arange(p, dtype=np.int64)
    # Single-linkage union-find.
    parent = np.arange(p, dtype=np.int64)
    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(p):
        for j in range(i + 1, p):
            if C[i, j] >= float(corr_threshold):
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj
    # Compact ids.
    raw_ids = np.array([_find(i) for i in range(p)], dtype=np.int64)
    uniq = np.unique(raw_ids)
    id_map = {int(v): k for k, v in enumerate(uniq)}
    cluster_id = np.array([id_map[int(v)] for v in raw_ids], dtype=np.int64)
    n_clusters = uniq.size
    # ---- step 2: bootstrap selection ----
    cluster_sel_freq = np.zeros(n_clusters, dtype=np.float64)
    feat_sel_freq = np.zeros(p, dtype=np.float64)
    half = n // 2
    for b in range(int(n_bootstrap)):
        idx = rng.permutation(n)[:half]
        try:
            sel = selector_fn(X[idx], y[idx])
        except Exception:
            continue
        sel = np.asarray(sel, dtype=np.int64).ravel()
        sel = sel[(sel >= 0) & (sel < p)]
        feat_sel_freq[sel] += 1
        selected_clusters = np.unique(cluster_id[sel])
        cluster_sel_freq[selected_clusters] += 1
    cluster_sel_freq /= max(float(n_bootstrap), 1.0)
    feat_sel_freq /= max(float(n_bootstrap), 1.0)
    # ---- step 3: keep clusters above threshold ----
    kept_clusters = np.where(cluster_sel_freq >= float(pi_threshold))[0]
    chosen = []
    for cid in kept_clusters:
        members = np.where(cluster_id == cid)[0]
        if members.size == 0:
            continue
        # Return the member with highest base selection freq.
        best = members[int(np.argmax(feat_sel_freq[members]))]
        chosen.append(int(best))
    info = {
        "n_clusters": int(n_clusters),
        "n_kept_clusters": int(kept_clusters.size),
        "cluster_sel_freq": cluster_sel_freq.tolist(),
        "feat_sel_freq": feat_sel_freq.tolist(),
    }
    if return_clusters:
        info["cluster_id"] = cluster_id.tolist()
    return np.asarray(chosen, dtype=np.int64), feat_sel_freq, info


def complementary_pairs_stability(
    X: np.ndarray, y: np.ndarray,
    selector_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    n_pairs: int = 50,
    pi_threshold: float = 0.6,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Shah-Samworth 2013 Complementary Pairs Stability Selection.

    For each of ``n_pairs`` half-splits ``(I_b, I_b^c)``:
      1. Run ``selector_fn`` on each half; record both selections.
      2. A feature is "complementary-selected" if it appears in BOTH halves.

    Final selection frequency = (1 / n_pairs) * count of pairs where the
    feature was complementary-selected. Keep features above ``pi_threshold``.

    The Shah-Samworth bound on expected number of falsely selected features
    is tighter than Meinshausen-Buhlmann's by leveraging the
    complementary-pair structure (no exchangeability assumption needed).
    """
    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    rng = np.random.default_rng(int(rng_seed))
    half = n // 2
    pair_complementary = np.zeros(p, dtype=np.float64)
    union_freq = np.zeros(p, dtype=np.float64)
    for b in range(int(n_pairs)):
        idx = rng.permutation(n)
        idx_b = idx[:half]
        idx_bc = idx[half:2 * half]
        try:
            sel_b = np.asarray(selector_fn(X[idx_b], y[idx_b]), dtype=np.int64).ravel()
            sel_bc = np.asarray(selector_fn(X[idx_bc], y[idx_bc]), dtype=np.int64).ravel()
        except Exception:
            continue
        sel_b = sel_b[(sel_b >= 0) & (sel_b < p)]
        sel_bc = sel_bc[(sel_bc >= 0) & (sel_bc < p)]
        # Complementary selection: feature in both halves.
        set_b = set(int(x) for x in sel_b)
        set_bc = set(int(x) for x in sel_bc)
        comp = set_b & set_bc
        union = set_b | set_bc
        for f in comp:
            pair_complementary[f] += 1
        for f in union:
            union_freq[f] += 1
    pair_complementary /= max(float(n_pairs), 1.0)
    union_freq /= max(float(n_pairs), 1.0)
    chosen = np.where(pair_complementary >= float(pi_threshold))[0]
    info = {
        "n_pairs": int(n_pairs),
        "pair_complementary_freq": pair_complementary.tolist(),
        "union_freq": union_freq.tolist(),
    }
    return chosen, pair_complementary, info


__all__ = [
    "cluster_stability_selection",
    "complementary_pairs_stability",
]
