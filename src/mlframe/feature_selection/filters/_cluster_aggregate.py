"""Clustered-feature aggregation for MRMR: replace/augment a cluster of correlated "reflection"
features with a denoised aggregate.

When a hidden factor ``z`` drives several noisy reflections ``A_i = lambda_i*z + eps_i``, MRMR keeps
the single best reflection and drops the rest. A combination (mean / inverse-variance / PCA-PC1 /
factor-score / median) of the cluster has noise variance ~``sigma^2/k`` -> a cleaner estimate of
``z`` -> higher MI with the target than any single member. This module discovers such clusters
(reusing the friend-graph MI-edge metric + redundancy criterion), builds the aggregate as a k-ary FE
recipe, and gates adoption on the aggregate strictly out-scoring the best single member's MI with y.

Design notes:
- Aggregation is UNSUPERVISED (no y in the combination); only the accept/reject GATE sees y -> the
  same leakage firewall every FE recipe uses.
- Every LINEAR combiner is a weight vector ``w`` over standardized+sign-aligned members; the aggregate
  is ``Z @ w`` (``mean_z`` -> 1/k, ``mean_inv_var`` -> reliability-weighted, ``pca_pc1`` -> PC1
  eigenvector, ``factor_score`` -> Bartlett combiner). ``median`` is the only non-linear v1 method.
- Fit builds the recipe (storing train mean/std/signs/weights) then calls ``apply_recipe`` to produce
  the binned column, so the fit-time column IS the replay output -> train/test parity by construction.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .engineered_recipes import apply_recipe, build_cluster_aggregate_recipe
from .friend_graph import node_relevance, pairwise_mi_edge
from .info_theory import mi

logger = logging.getLogger(__name__)

CLUSTER_AGGREGATE_METHODS = ("mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score")


# ---------------------------------------------------------------------------
# Standardization + weight derivation (the aggregator menu)
# ---------------------------------------------------------------------------


def _standardize_align(M: np.ndarray, ref_col: int):
    """z-score each column (train stats) and sign-align to ``ref_col``. Returns (Z, mean, std, signs)."""
    mean = M.mean(axis=0)
    std = M.std(axis=0)
    std_safe = np.where(std > 0.0, std, 1.0)
    Zc = (M - mean) / std_safe
    # Sign-align each column to ref_col by the SIGN of its correlation with ref.
    # corr(j, ref) < 0  <=>  the covariance numerator sum((Zc_j - mean_j)(Zc_ref
    # - mean_ref)) < 0 (the corr denominator is a non-negative std product), so
    # we need only that numerator's sign -- computable for ALL columns at once
    # via one centered matrix-vector product, replacing the per-column
    # np.corrcoef loop (K calls, each a 2x2 corr matrix over N rows; 1.12x at
    # K=4 rising to 2.36x at K=20). A constant column has a zero numerator ->
    # sign +1, matching the loop's corrcoef-NaN -> isfinite-False -> +1 fallback
    # (and this form raises no divide-by-zero RuntimeWarning on constant cols).
    Zc_centered = Zc - Zc.mean(axis=0)
    cov_num = Zc_centered.T @ Zc_centered[:, ref_col]
    signs = np.where(cov_num < 0.0, -1.0, 1.0)
    signs[ref_col] = 1.0
    Z = Zc * signs
    return Z, mean, std, signs


def _svd_flip_pc1(Z: np.ndarray) -> np.ndarray:
    """PC1 loading vector of centered ``Z`` (correlation-matrix eigenvector), sign-pinned via the
    sklearn ``svd_flip`` convention (largest-abs loading positive) for cross-BLAS determinism."""
    Zc = Z - Z.mean(axis=0)
    _u, _s, vt = np.linalg.svd(Zc, full_matrices=False)
    v = vt[0]
    v = v * np.sign(v[np.argmax(np.abs(v))] or 1.0)
    return v


def _pc1_communalities(Z: np.ndarray) -> np.ndarray:
    """Communality_i = (corr of member_i with the PC1 score)^2 in [0,1] under a 1-factor read:
    the fraction of member_i's variance explained by the shared component. Used for reliability."""
    v = _svd_flip_pc1(Z)
    score = (Z - Z.mean(axis=0)) @ v
    comm = np.array([np.corrcoef(Z[:, j], score)[0, 1] ** 2 for j in range(Z.shape[1])], dtype=np.float64)
    return np.clip(np.nan_to_num(comm, nan=0.0), 1e-6, 1.0 - 1e-6)


def _derive_weights(Z: np.ndarray, method: str):
    """Weight vector for a LINEAR combiner over standardized+sign-aligned ``Z`` (or ``None`` for median).

    mean_z -> uniform; mean_inv_var -> reliability comm/(1-comm) (BLUE-ish under hetero noise);
    pca_pc1 -> PC1 eigenvector (variance-max, best under hetero loadings); factor_score -> Bartlett
    1-factor combiner (principal-factor loadings, w ∝ Psi^-1 L / (L' Psi^-1 L))."""
    k = Z.shape[1]
    if method == "mean_z":
        return np.full(k, 1.0 / k, dtype=np.float64)
    if method == "median":
        return None
    if method == "mean_inv_var":
        comm = _pc1_communalities(Z)
        w = comm / (1.0 - comm)  # reliability / noise ~ inverse residual variance
        return w / w.sum()
    if method == "pca_pc1":
        return _svd_flip_pc1(Z)
    if method == "factor_score":
        comm = _pc1_communalities(Z)
        load = np.sqrt(comm)  # principal-factor loadings (sign already aligned, comm>=0)
        psi = np.clip(1.0 - comm, 1e-6, None)  # idiosyncratic (uniqueness) variances
        wl = load / psi
        denom = float(load @ (load / psi))
        w = wl / denom if denom > 0 else np.full(k, 1.0 / k)
        return w
    raise ValueError(f"unknown cluster-aggregate method {method!r}")


# ---------------------------------------------------------------------------
# Cluster discovery (reuses the friend-graph MI-edge metric + redundancy criterion)
# ---------------------------------------------------------------------------


def _continuous_cols(X, names: Sequence[str]) -> np.ndarray:
    from .engineered_recipes import _extract_column

    cols = [np.nan_to_num(np.asarray(_extract_column(X, n), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0) for n in names]
    return np.column_stack(cols)


def _connected_components(n: int, edges: list) -> list:
    """Union-find connected components over ``edges`` (list of (a,b) into 0..n-1)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    comps: dict = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return [c for c in comps.values() if len(c) > 1]


def _discover_clusters(
    *, data, cols, nbins, X, target_indices, feature_names_in_, categorical_idx,
    cached_MIs, min_member_relevance, corr_threshold, min_cluster_size, max_cluster_size,
    homogeneity_tau, max_candidates, mi_eps, edge_significance, dtype,
):
    """Return a list of clusters; each = dict(members=[col_idx...], rep=col_idx, rel={idx:rel})."""
    target = np.asarray(target_indices, dtype=np.int64)
    target_set = set(int(t) for t in target_indices)
    feat_in = set(feature_names_in_) if feature_names_in_ is not None else None
    cat_set = set(int(c) for c in (categorical_idx or ()))

    # Pool: numeric RAW (feature_names_in_) columns, excluding targets/categoricals, with marginal
    # relevance above a LOW floor (captures the high-value case where every reflection is individually
    # too weak to be selected). Reuse cached_MIs[(i,)] (target-relevance key matches) where present.
    pool = []
    rel = {}
    for i in range(len(cols)):
        if i in target_set or i in cat_set:
            continue
        if feat_in is not None and cols[i] not in feat_in:
            continue  # engineered / non-raw col -> recipe src wouldn't resolve at transform
        key = (i,)
        r = cached_MIs[key] if (cached_MIs is not None and key in cached_MIs) else node_relevance(data, i, target, nbins, dtype=dtype)
        if isinstance(r, tuple):
            r = r[0]
        r = float(r)
        if r >= min_member_relevance:
            pool.append(i)
            rel[i] = r
    pool.sort(key=lambda i: (-rel[i], i))
    if len(pool) > max_candidates:
        pool = pool[:max_candidates]
    if len(pool) < min_cluster_size:
        return []

    # Continuous correlation among the pool (identifies genuine linear reflections).
    Xc = _continuous_cols(X, [cols[i] for i in pool])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(Xc, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    n_samples = max(1, int(data.shape[0]))

    edges = []
    for a in range(len(pool)):
        for b in range(a + 1, len(pool)):
            if abs(corr[a, b]) < corr_threshold:
                continue
            if pairwise_mi_edge(data, pool[a], pool[b], nbins, n_samples, mi_eps=mi_eps, edge_significance=edge_significance, dtype=dtype) is None:
                continue
            edges.append((a, b))

    clusters = []
    used = set()
    for comp in sorted(_connected_components(len(pool), edges), key=len, reverse=True):
        members = [pool[j] for j in comp if pool[j] not in used]
        if len(members) < min_cluster_size:
            continue
        members.sort(key=lambda i: (-rel[i], i))  # representative = highest-relevance member
        rep = members[0]
        if len(members) > max_cluster_size:
            # keep representative + top-|corr|-to-rep members
            rep_pos = pool.index(rep)
            members = [rep] + sorted(
                [m for m in members if m != rep], key=lambda m: -abs(corr[rep_pos, pool.index(m)])
            )[: max_cluster_size - 1]
        # Unidimensionality: PC1 explains >= tau of the standardized cluster variance. This is the
        # structural discriminator -- genuine reflections of ONE latent are unidimensional; a
        # partial-shared+distinct cluster (z + delta_i) is multi-factor -> low PC1 ratio -> rejected.
        # (The friend-graph conditional-MI "sink" test is NOT used here: for reflections it conflates
        # denoising-gain with distinct-signal info and would wrongly reject good clusters.)
        M = _continuous_cols(X, [cols[m] for m in members])
        Z, _mu, _sd, _sg = _standardize_align(M, 0)
        Zc = Z - Z.mean(axis=0)
        sv = np.linalg.svd(Zc, full_matrices=False, compute_uv=False)
        var_ratio = float((sv[0] ** 2) / max(np.sum(sv ** 2), 1e-12))
        if var_ratio < homogeneity_tau:
            continue
        clusters.append({"members": members, "rep": rep, "rel": {m: rel[m] for m in members}})
        used.update(members)
    return clusters


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_cluster_aggregate_step(
    *, data, cols, nbins, X, target_indices, feature_names_in_, categorical_idx,
    cached_MIs, engineered_recipes, quantization_nbins, quantization_method, quantization_dtype,
    methods=("mean_z",), mi_prevalence=1.0, min_member_relevance=0.0, min_cluster_size=3,
    max_cluster_size=12, corr_threshold=0.6, homogeneity_tau=0.6,
    max_candidates=200, mode="augment", is_polars_input=False, verbose=0,
    mi_eps=1e-6, edge_significance=3.0, dtype=np.int32,
):
    """Discover correlated reflection clusters, build a denoised aggregate per cluster (best gated
    method), append it to (data, cols, nbins, X) and register the recipe. Returns
    ``(data, cols, nbins, X, n_added, removed_member_names)``. ``removed_member_names`` is non-empty
    only in ``mode="replace"`` (consumed by ``_fit_impl`` before the cols->original remap)."""
    methods = tuple(m for m in methods if m in CLUSTER_AGGREGATE_METHODS) or ("mean_z",)
    target = np.asarray(target_indices, dtype=np.int64)
    quantization = {"nbins": int(quantization_nbins), "method": str(quantization_method or "quantile"), "dtype": np.dtype(quantization_dtype).str}

    clusters = _discover_clusters(
        data=data, cols=cols, nbins=nbins, X=X, target_indices=target_indices,
        feature_names_in_=feature_names_in_, categorical_idx=categorical_idx, cached_MIs=cached_MIs,
        min_member_relevance=min_member_relevance, corr_threshold=corr_threshold,
        min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size,
        homogeneity_tau=homogeneity_tau, max_candidates=max_candidates, mi_eps=mi_eps,
        edge_significance=edge_significance, dtype=dtype,
    )

    n_added = 0
    removed_member_names: list = []
    added_indices: list = []  # cols-space indices of accepted aggregates (appended to selected_vars by caller)
    summary: list = []        # JSON-serializable per-aggregate record for the log + meta_info report
    new_data_cols = []
    for cl in clusters:
        members = cl["members"]
        member_names = [cols[m] for m in members]
        rep = cl["rep"]
        M = _continuous_cols(X, member_names)
        Z, mean, std, signs = _standardize_align(M, members.index(rep))
        best_member_mi = max(cl["rel"].values())

        best = None  # (mi, recipe, binned_col, method)
        for method in methods:
            weights = _derive_weights(Z, method)
            agg_name = f"clusteragg_{method}({'+'.join(member_names)})"
            # 2026-05-30 Wave 9.1 fix (loop iter 29): compute the
            # continuous aggregate FIRST so the recipe can persist the
            # fit-time quantile edges. Pre-fix the recipe was built
            # without edges and ``apply_recipe`` later re-quantiled on
            # test data, silently shifting bin codes between fit and
            # transform under distribution drift. Sibling of iter 28.
            if method == "median":
                _agg_continuous = np.median(Z, axis=1)
            else:
                _agg_continuous = Z @ np.asarray(weights, dtype=np.float64)
            _agg_continuous = np.nan_to_num(
                _agg_continuous, copy=False, nan=0.0, posinf=0.0, neginf=0.0,
            )
            # Build a per-recipe quantization dict with fit-time edges
            # (deep-copy the caller's base dict so we don't mutate it
            # across loop iterations).
            if quantization is not None:
                _q_local = dict(quantization)
                _nb = int(_q_local.get("nbins", 0))
                if _nb >= 2:
                    if str(_q_local.get("method", "quantile")) == "quantile":
                        _q_arr = np.linspace(0.0, 100.0, _nb + 1)
                        _edges = np.nanpercentile(_agg_continuous, _q_arr)
                    else:
                        _finite = _agg_continuous[np.isfinite(_agg_continuous)]
                        if _finite.size:
                            _edges = np.linspace(
                                float(_finite.min()), float(_finite.max()),
                                _nb + 1,
                            )
                        else:
                            _edges = np.linspace(0.0, 0.0, _nb + 1)
                    _q_local["edges"] = _edges.tolist()
            else:
                _q_local = None
            recipe = build_cluster_aggregate_recipe(
                name=agg_name, src_names=tuple(member_names), method=method,
                member_mean=mean, member_std=std, signs=signs, weights=weights,
                quantization=_q_local, diagnostics={"representative": cols[rep]},
            )
            # Fit-time column IS the replay output (parity by construction).
            binned = apply_recipe(recipe, X)
            agg_mi = float(mi(np.column_stack([data, binned.astype(data.dtype)]), np.array([data.shape[1]], dtype=np.int64),
                              target, np.concatenate([np.asarray(nbins), [int(quantization_nbins)]]).astype(np.int64), dtype=dtype))
            if best is None or agg_mi > best[0]:
                best = (agg_mi, recipe, binned, method)

        agg_mi, recipe, binned, method = best
        # Gate: the denoised aggregate must STRICTLY out-score the best single member (denoising claim).
        if agg_mi < mi_prevalence * best_member_mi:
            if verbose > 1:
                logger.info("cluster_aggregate: rejected %s (MI %.5f < %.2f x best member %.5f)", recipe.name, agg_mi, mi_prevalence, best_member_mi)
            continue

        # Accept: append the binned aggregate column to the binned matrix `data` / `cols` / `nbins`
        # (what screening + the remap consume). Deliberately NOT added to the caller's `X`: it is unused
        # there (replay reconstructs the aggregate from the raw member columns via the recipe), and
        # writing to `X` would mutate the caller's frame and leak the engineered name into a later
        # fit's feature_names_in_.
        new_data_cols.append(binned.astype(quantization_dtype))
        added_indices.append(len(cols))  # cols-space index this aggregate will occupy
        cols = cols + [recipe.name]
        nbins = np.concatenate([np.asarray(nbins), [int(quantization_nbins)]]).astype(np.asarray(nbins).dtype)
        if engineered_recipes is not None:
            engineered_recipes[recipe.name] = recipe
        n_added += 1
        summary.append({
            "name": recipe.name, "method": method, "members": list(member_names),
            "aggregate_mi": round(float(agg_mi), 6), "best_member_mi": round(float(best_member_mi), 6),
            "mi_gain": round(float(agg_mi - best_member_mi), 6), "mode": mode,
        })
        if mode == "replace":
            removed_member_names.extend(member_names)
        if verbose:
            logger.info("cluster_aggregate: accepted %s (MI %.5f vs best member %.5f, k=%d, method=%s)", recipe.name, agg_mi, best_member_mi, len(members), method)

    if new_data_cols:
        data = np.append(data, np.column_stack(new_data_cols), axis=1)
    return data, cols, nbins, X, n_added, removed_member_names, added_indices, summary
