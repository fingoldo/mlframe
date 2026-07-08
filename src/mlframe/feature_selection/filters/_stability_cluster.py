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

import logging
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


def cluster_stability_selection(
    X: Any, y: np.ndarray,
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
    _is_df = hasattr(X, "iloc")
    n, p = X.shape
    rng = np.random.default_rng(int(rng_seed))
    # Numeric view for the |Pearson| clustering. A raw categorical/string column (reaching here under skip_categorical_encoding) cannot enter a
    # correlation graph, so coerce per-column and flag the non-numeric ones -- they stay SINGLETON clusters (never merged) yet remain selectable via
    # the bootstrap selector below, which is handed dtype-preserved rows so a classic sub-MRMR factorises them itself. Pre-fix a blanket
    # ``np.asarray(X, dtype=float64)`` raised "could not convert string to float" on such a column and the caller fell back to classic, disabling
    # cluster stability entirely on any data carrying a raw categorical.
    _num_ok = np.ones(p, dtype=bool)
    if not _is_df and np.issubdtype(np.asarray(X).dtype, np.number):
        Xn = np.asarray(X, dtype=np.float64)
    else:
        Xn = np.zeros((n, p), dtype=np.float64)
        for _c in range(p):
            _cv = X.iloc[:, _c].to_numpy() if _is_df else np.asarray(X)[:, _c]
            try:
                Xn[:, _c] = np.asarray(_cv, dtype=np.float64)
            except (TypeError, ValueError):
                _num_ok[_c] = False
    # ---- step 1: cluster by |Pearson| >= corr_threshold (numeric columns only; non-numeric stay singletons) ----
    if p > 1:
        # Z-standardise then correlation = (1/n) * Z.T @ Z.
        Z = (Xn - Xn.mean(axis=0)) / (Xn.std(axis=0) + 1e-12)
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
    # Vectorise edge discovery: build the upper-triangular |corr|>=thr adjacency
    # (numeric columns only) in C, then union-find only over the actual edges.
    # Same single-linkage result as the prior O(p^2) Python double loop, but the
    # quadratic scan runs in C instead of the interpreter.
    if p > 1:
        adj = (C >= float(corr_threshold)) & _num_ok[:, None] & _num_ok[None, :]
        ii, jj = np.where(np.triu(adj, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
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
    # 2026-06-03 (audit hierarchy-stability-9): divide frequencies by the number
    # of SUCCESSFUL bootstraps, not the nominal ``n_bootstrap``. A selector_fn
    # that raises is silently ``continue``d; dividing by the nominal count then
    # scales every frequency by (n_success / n_bootstrap), systematically
    # deflating them toward 0 on an effective sample size the caller never sees
    # -- which invalidates the Faletto-Bien / Shah-Samworth bounds (parameterised
    # by the number of subsamples). Track and surface n_effective / n_failed.
    n_success = 0
    n_failed = 0
    for _b in range(int(n_bootstrap)):
        idx = rng.permutation(n)[:half]
        try:
            sel = selector_fn(X.iloc[idx] if _is_df else X[idx], y[idx])
        except Exception:
            n_failed += 1
            continue
        n_success += 1
        sel = np.asarray(sel, dtype=np.int64).ravel()
        sel = sel[(sel >= 0) & (sel < p)]
        feat_sel_freq[sel] += 1
        selected_clusters = np.unique(cluster_id[sel])
        cluster_sel_freq[selected_clusters] += 1
    if n_failed:
        logger.warning(
            "cluster_stability_selection: %d/%d bootstraps failed; frequencies "
            "are computed over the %d successful runs (effective B), so the "
            "Faletto-Bien/Shah-Samworth bound is parameterised by %d, not %d.",
            n_failed, int(n_bootstrap), n_success, n_success, int(n_bootstrap),
        )
    _denom = max(float(n_success), 1.0)
    cluster_sel_freq /= _denom
    feat_sel_freq /= _denom
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
        # 2026-06-03 (audit hierarchy-stability-9): effective subsample count
        # the frequencies (and the advertised bound) are actually based on.
        "n_effective": int(n_success),
        "n_failed": int(n_failed),
    }
    if return_clusters:
        info["cluster_id"] = cluster_id.tolist()
    return np.asarray(chosen, dtype=np.int64), feat_sel_freq, info


def complementary_pairs_stability(
    X: Any, y: np.ndarray,
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
    # complementary_pairs uses X only to feed the bootstrap selector (no correlation/clustering step), so keep the ORIGINAL dtype -- a blanket
    # float64 coercion crashed on a raw categorical column. The selector is handed dtype-preserved rows (a classic sub-MRMR factorises categoricals).
    _is_df = hasattr(X, "iloc")
    n, p = X.shape
    rng = np.random.default_rng(int(rng_seed))
    half = n // 2
    pair_complementary = np.zeros(p, dtype=np.float64)
    union_freq = np.zeros(p, dtype=np.float64)
    # 2026-06-03 (audit hierarchy-stability-9): count SUCCESSFUL pairs so the
    # frequencies divide by the effective sample size (see the cluster variant).
    n_success = 0
    n_failed = 0
    for _b in range(int(n_pairs)):
        idx = rng.permutation(n)
        idx_b = idx[:half]
        # 2026-06-03 (audit hierarchy-stability-8): the complement must be the
        # REST of the permutation (idx[half:]), not idx[half:2*half]. For odd n
        # the latter drops the middle row, so the two halves are not a true
        # partition of the sample -- violating the complementary-pair structure
        # the Shah-Samworth bound assumes (it allows |I| and |I^c| to differ by
        # one, but requires I ∪ I^c = all rows).
        idx_bc = idx[half:]
        try:
            sel_b = np.asarray(selector_fn(X.iloc[idx_b] if _is_df else X[idx_b], y[idx_b]), dtype=np.int64).ravel()
            sel_bc = np.asarray(selector_fn(X.iloc[idx_bc] if _is_df else X[idx_bc], y[idx_bc]), dtype=np.int64).ravel()
        except Exception:
            n_failed += 1
            continue
        n_success += 1
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
    if n_failed:
        logger.warning(
            "complementary_pairs_stability: %d/%d pairs failed; frequencies are "
            "computed over the %d successful pairs (effective B).",
            n_failed, int(n_pairs), n_success,
        )
    _denom = max(float(n_success), 1.0)
    pair_complementary /= _denom
    union_freq /= _denom
    chosen = np.where(pair_complementary >= float(pi_threshold))[0]
    info = {
        "n_pairs": int(n_pairs),
        "pair_complementary_freq": pair_complementary.tolist(),
        "union_freq": union_freq.tolist(),
        # 2026-06-03 (audit hierarchy-stability-9): effective pair count.
        "n_effective": int(n_success),
        "n_failed": int(n_failed),
    }
    return chosen, pair_complementary, info


__all__ = [
    "cluster_stability_selection",
    "complementary_pairs_stability",
]
