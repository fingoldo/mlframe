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
- Fit builds the recipe (storing train mean/std/signs/weights) and the fit-time binned column is
  produced by the SAME quantile-edge ``searchsorted`` reduction ``apply_recipe`` replays with, so the
  fit-time column IS the replay output -> train/test parity by construction. (Fit bins the aggregate it
  already holds directly rather than round-tripping back through ``apply_recipe`` member re-extraction;
  the no-edges fallback still defers to ``apply_recipe``.)
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from .engineered_recipes import apply_recipe, build_cluster_aggregate_recipe
from .friend_graph import node_relevance, pairwise_mi_edge
from .info_theory import mi

logger = logging.getLogger(__name__)

CLUSTER_AGGREGATE_METHODS = (
    "mean_z", "mean_inv_var", "median", "pca_pc1", "factor_score",
    # Layer 44 (2026-05-31): four additional aggregators added to the DCD
    # auto bake-off pool to give it more candidate combiners to choose from.
    # ``pca_pc2`` captures secondary cluster structure (correlated latents);
    # ``median_z`` is robust to outlier rows; ``signed_max_abs`` and
    # ``signed_l2_sum`` are non-linear combiners that surface the loudest
    # member signal / quadratic combination. The first two follow the
    # existing ``Z @ w`` linear-weight pattern; the last two are non-linear
    # (no ``weights`` -- replay re-applies the same row reduction).
    "pca_pc2", "median_z", "signed_max_abs", "signed_l2_sum",
)


# Layer 44: methods that bypass the ``Z @ weights`` pattern (non-linear or
# row-reductions). For these, ``_derive_weights`` returns None and replay
# (``_apply_cluster_aggregate``) dispatches to a per-method row-reducer.
_NONLINEAR_METHODS = frozenset({
    "median", "median_z", "signed_max_abs", "signed_l2_sum",
})


def _apply_method_nonlinear(Z: np.ndarray, method: str) -> np.ndarray:
    """Row-reducer for non-linear cluster aggregators.

    ``median`` / ``median_z``: per-row median of standardized members (robust to
        outlier members). ``median`` and ``median_z`` are the same reduction;
        the two names exist because ``median`` predates Layer 44 and is kept
        bit-identical for the legacy pin while ``median_z`` is the explicit
        z-scored alias used in the new bake-off pool.
    ``signed_max_abs``: per-row ``sign(z_j*) * max_j |z_j|`` where ``j*`` is
        the arg-max of |z_j|. Surfaces the loudest single member signal.
    ``signed_l2_sum``: per-row ``sum_j sign(z_j) * z_j**2``. A signed
        quadratic combiner -- larger-magnitude members contribute more,
        sign carried so cancellation across opposite-sign members works.
    """
    if method in ("median", "median_z"):
        return np.asarray(np.median(Z, axis=1))
    if method == "signed_max_abs":
        abs_Z = np.abs(Z)
        idx = np.argmax(abs_Z, axis=1)
        rows = np.arange(Z.shape[0])
        signs_row = np.sign(Z[rows, idx])
        # 0 sign collapses to +1 to avoid zeroing a max-abs of an exactly-zero row.
        signs_row = np.where(signs_row == 0.0, 1.0, signs_row)
        return np.asarray(signs_row * abs_Z[rows, idx])
    if method == "signed_l2_sum":
        return np.asarray(np.sum(np.sign(Z) * (Z**2), axis=1))
    raise ValueError(f"unknown non-linear cluster-aggregate method {method!r}")


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


# Layer 50 (2026-05-31): SVD-reuse cache for the DCD auto bake-off.
#
# Pre-fix: every call to ``_svd_flip_pc1`` / ``_svd_flip_pcN`` /
# ``_pc1_communalities`` independently centred ``Z`` and called
# ``np.linalg.svd``. The DCD auto bake-off (``_select_swap_method_auto``)
# evaluates 7 combiner methods on each of K folds; 4 of those methods
# (``mean_inv_var``, ``pca_pc1``, ``pca_pc2``, ``factor_score``) need the
# SVD of the SAME ``Z_train`` matrix -- so per fold we paid 4x the SVD
# work on identical input. Layer 50 profile on p=200 / n=5000 / 10
# latents attributed 0.444s tottime (#1 hotspot) to ``np.linalg.svd``
# alone, 150 calls; that's where the cache pays off.
#
# Cache shape: a plain dict ``{"vt": vt, "Zc": Zc, "comm": comm}`` passed
# explicitly to ``_derive_weights`` / ``_pc1_communalities`` /
# ``_svd_flip_pc1`` / ``_svd_flip_pcN``. The caller owns the dict
# (one per fold) and discards it after the fold finishes -- no global
# state, no thread-locals, no GC weakref shenanigans. Callers that don't
# need caching pass ``svd_cache=None`` and the helpers compute fresh.


def _svd_compute(Z: np.ndarray, svd_cache: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Zc, vt)`` for centred ``Z`` with caching. ``svd_cache`` is
    a dict the caller threads through one batch of ``_derive_weights`` calls
    sharing the same ``Z``; on first call the SVD is computed and stored,
    subsequent calls (other methods on the same ``Z``) reuse it. Pass None
    to disable caching.
    """
    if svd_cache is not None and "vt" in svd_cache:
        return svd_cache["Zc"], svd_cache["vt"]
    Zc = Z - Z.mean(axis=0)
    _u, _s, vt = np.linalg.svd(Zc, full_matrices=False)
    if svd_cache is not None:
        svd_cache["Zc"] = Zc
        svd_cache["vt"] = vt
    return Zc, vt


def _svd_flip_pc1(Z: np.ndarray, svd_cache: dict | None = None) -> np.ndarray:
    """PC1 loading vector of centered ``Z`` (correlation-matrix eigenvector), sign-pinned via the
    sklearn ``svd_flip`` convention (largest-abs loading positive) for cross-BLAS determinism."""
    _Zc, vt = _svd_compute(Z, svd_cache)
    v = vt[0]
    v = v * np.sign(v[np.argmax(np.abs(v))] or 1.0)
    return np.asarray(v)


def _svd_flip_pcN(Z: np.ndarray, idx: int, svd_cache: dict | None = None) -> np.ndarray:
    """N-th principal-component loading vector of centered ``Z`` (0-indexed), sign-pinned via the
    sklearn ``svd_flip`` convention. Layer 44: powers ``pca_pc2`` for clusters with multi-factor
    structure (when latents are correlated, PC2 captures the secondary axis of shared variation
    that PC1 misses).

    For ``k`` members the SVD returns ``min(n, k)`` components; if ``idx`` is out of range (e.g. PC2
    requested on a single-column Z) we fall back to PC1 to keep the combiner well-defined.
    """
    _Zc, vt = _svd_compute(Z, svd_cache)
    if idx >= vt.shape[0]:
        idx = 0
    v = vt[idx]
    v = v * np.sign(v[np.argmax(np.abs(v))] or 1.0)
    return np.asarray(v)


def _pc1_communalities(Z: np.ndarray, svd_cache: dict | None = None) -> np.ndarray:
    """Communality_i = (corr of member_i with the PC1 score)^2 in [0,1] under a 1-factor read:
    the fraction of member_i's variance explained by the shared component. Used for reliability.

    Layer 50 (2026-05-31): vectorised per-column corrcoef. Pre-fix the loop
    ``[np.corrcoef(Z[:,j], score)[0,1]**2 for j ...]`` paid K corrcoef
    dispatches; for K=20 members on n=4000 rows that loop alone showed up
    on the profile as the bulk of ``_pc1_communalities`` cumtime. Replace
    with one centred matmul: ``corr_j = (Zc_j . score_c) / (||Zc_j||*||score_c||)``,
    bit-equivalent up to FP-summation order to corrcoef (which uses
    ``cov(x,y)/sqrt(var(x)var(y))`` with the same n-1 normalisation that
    cancels in the ratio). The svd_cache lets the caller share the
    Zc/vt computation with sibling ``_svd_flip_pc1`` / ``_derive_weights``
    calls on the same ``Z``.
    """
    Zc, vt = _svd_compute(Z, svd_cache)
    if svd_cache is not None and "comm" in svd_cache:
        return np.asarray(svd_cache["comm"])
    v = vt[0]
    v = v * np.sign(v[np.argmax(np.abs(v))] or 1.0)
    score = Zc @ v
    score_c = score - score.mean()
    score_norm_sq = float(score_c @ score_c)
    if score_norm_sq <= 0.0:
        comm = np.zeros(Z.shape[1], dtype=np.float64)
    else:
        # ``Zc`` is already centred (column-mean removed); per-column norm = sqrt(sum_i Zc_ij^2).
        col_norm_sq = (Zc * Zc).sum(axis=0)
        numer = Zc.T @ score_c
        denom = col_norm_sq * score_norm_sq
        with np.errstate(invalid="ignore", divide="ignore"):
            comm = (numer * numer) / np.where(denom > 0.0, denom, 1.0)
        # Constant columns (col_norm_sq == 0) get corrcoef NaN -> 0 per the
        # legacy nan_to_num path below.
        comm = np.where(col_norm_sq > 0.0, comm, 0.0)
    comm = np.clip(np.nan_to_num(comm, nan=0.0), 1e-6, 1.0 - 1e-6)
    if svd_cache is not None:
        svd_cache["comm"] = comm
    return comm


def _derive_weights(Z: np.ndarray, method: str, svd_cache: dict | None = None):
    """Weight vector for a LINEAR combiner over standardized+sign-aligned ``Z`` (or ``None`` for median).

    mean_z -> uniform; mean_inv_var -> reliability comm/(1-comm) (BLUE-ish under hetero noise);
    pca_pc1 -> PC1 eigenvector (variance-max, best under hetero loadings); factor_score -> Bartlett
    1-factor combiner (principal-factor loadings, w ∝ Psi^-1 L / (L' Psi^-1 L)).

    Layer 50 (2026-05-31): ``svd_cache`` is an optional dict the caller threads
    through a batch of method evaluations on the SAME ``Z`` (e.g. the DCD auto
    bake-off's K-fold loop). Methods that need the SVD (``mean_inv_var``,
    ``pca_pc1``, ``pca_pc2``, ``factor_score``) share the cached vt / Zc /
    communalities, collapsing 4 SVDs into 1 per fold. Pass None to keep the
    pre-Layer-50 behaviour (re-SVD every call)."""
    k = Z.shape[1]
    if method == "mean_z":
        return np.full(k, 1.0 / k, dtype=np.float64)
    # Layer 44: ``median`` (legacy alias) and the four new non-linear methods
    # have no weight vector -- the aggregate is a row-reduction handled by
    # ``_apply_method_nonlinear`` at fit and replay.
    if method in _NONLINEAR_METHODS:
        return None
    if method == "mean_inv_var":
        comm = _pc1_communalities(Z, svd_cache=svd_cache)
        w = comm / (1.0 - comm)  # reliability / noise ~ inverse residual variance
        return w / w.sum()
    if method == "pca_pc1":
        return _svd_flip_pc1(Z, svd_cache=svd_cache)
    if method == "pca_pc2":
        # Layer 44: 2nd principal component. On a homogeneous cluster (single
        # latent, equal loadings) PC2 is dominated by noise; on a multi-factor
        # cluster (correlated latents L1, L2 with members loading partially on
        # each) PC2 captures the orthogonal axis that PC1 leaves on the table.
        return _svd_flip_pcN(Z, 1, svd_cache=svd_cache)
    if method == "factor_score":
        comm = _pc1_communalities(Z, svd_cache=svd_cache)
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
    """Extract ``names`` columns from ``X`` as a float64 matrix with NaN/inf sanitized to 0.0, for distance/correlation-based cluster discovery."""
    from .engineered_recipes import _extract_column

    cols = [np.nan_to_num(np.asarray(_extract_column(X, n), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0) for n in names]
    return np.column_stack(cols)


def _connected_components(n: int, edges: list) -> list:
    """Union-find connected components over ``edges`` (list of (a,b) into 0..n-1).

    2026-06-03 bench-attempt-rejected (bench_community_vs_single_linkage): replacing
    this single-linkage CC with modularity/Louvain community detection gives NO win.
    On the thresholded binned-SU graph the chaining failure mode does not occur --
    a moderate bridge (corr ~0.6 to two groups) has binned SU ~0.09, far below tau,
    so it never links groups; single-linkage already recovers planted groups at
    ARI 1.0. Binning + the SU threshold supply the separation modularity would add.
    """
    parent = list(range(n))

    def find(x):
        """Path-halving find with iterative parent-pointer compression."""
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
            members = [rep, *sorted([m for m in members if m != rep], key=lambda m: -abs(corr[rep_pos, pool.index(m)]))[: max_cluster_size - 1]]
        # Unidimensionality: PC1 explains >= tau of the standardized cluster variance. This is the
        # structural discriminator -- genuine reflections of ONE latent are unidimensional; a
        # partial-shared+distinct cluster (z + delta_i) is multi-factor -> low PC1 ratio -> rejected.
        # (The friend-graph conditional-MI "sink" test is NOT used here: for reflections it conflates
        # denoising-gain with distinct-signal info and would wrongly reject good clusters.)
        M = _continuous_cols(X, [cols[m] for m in members])
        Z, mu, sd, sg = _standardize_align(M, 0)
        Zc = Z - Z.mean(axis=0)
        sv = np.linalg.svd(Zc, full_matrices=False, compute_uv=False)
        var_ratio = float((sv[0] ** 2) / max(np.sum(sv**2), 1e-12))
        if var_ratio < homogeneity_tau:
            continue
        # ``members[0] == rep`` always (set two lines above, or forced to index 0 by the max_cluster_size
        # truncation branch), so Z/mu/sd/sg (standardized against ref_col=0) are exactly what
        # run_cluster_aggregate_step would rebuild for this cluster's aggregate -- stash them so it reuses
        # this array instead of re-extracting + re-standardizing the same member columns.
        clusters.append({
            "members": members, "rep": rep, "rel": {m: rel[m] for m in members},
            "Z": Z, "mean": mu, "std": sd, "signs": sg,
        })
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

    # Target block + its nbins are invariant across every cluster/method scored below: `data`'s target
    # columns and `nbins` are only ever appended to at the tail (after this whole loop), so the target
    # indices' values never shift while clusters are being scored. Hoisted out of both the per-cluster
    # and per-method loops (was rebuilt on every method iteration of every cluster).
    _tcols = target
    _target_block = data[:, _tcols]
    _n_t = _tcols.shape[0]
    _compact_nbins = np.concatenate([np.asarray(nbins)[_tcols], [int(quantization_nbins)]]).astype(np.int64)

    n_added = 0
    removed_member_names: list = []
    added_indices: list = []  # cols-space indices of accepted aggregates (appended to selected_vars by caller)
    summary: list = []  # JSON-serializable per-aggregate record for the log + meta_info report
    new_data_cols = []
    for cl in clusters:
        members = cl["members"]
        member_names = [cols[m] for m in members]
        rep = cl["rep"]
        # members[0] == rep by construction (_discover_clusters), so this is exactly the Z/mean/std/signs
        # _discover_clusters already built (against the same ref_col=0) for the homogeneity gate -- reuse
        # it instead of re-extracting + re-standardizing the same member columns from X.
        Z, mean, std, signs = cl["Z"], cl["mean"], cl["std"], cl["signs"]
        best_member_mi = max(cl["rel"].values())

        best = None  # (mi, recipe, binned_col, method)
        # Layer 50 SVD-reuse: every SVD-needing method (mean_inv_var, pca_pc1,
        # pca_pc2, factor_score) shares the SAME ``Z`` per cluster, so thread one
        # cache through the method loop -> the SVD + PC1 communalities of ``Z``
        # are computed once and reused, collapsing up to 4 SVDs (+2 comm
        # re-derivations) per cluster into 1 each. Bit-identical: the cache
        # returns the exact (Zc, vt, comm) the per-method call would recompute
        # from the identical ``Z``. Discarded after the cluster finishes.
        _svd_cache: dict = {}
        for method in methods:
            weights = _derive_weights(Z, method, svd_cache=_svd_cache)
            agg_name = f"clusteragg_{method}({'+'.join(member_names)})"
            # 2026-05-30 Wave 9.1 fix (loop iter 29): compute the
            # continuous aggregate FIRST so the recipe can persist the
            # fit-time quantile edges. Pre-fix the recipe was built
            # without edges and ``apply_recipe`` later re-quantiled on
            # test data, silently shifting bin codes between fit and
            # transform under distribution drift. Sibling of iter 28.
            if method in _NONLINEAR_METHODS:
                # Layer 44: row-reduction for non-linear combiners; replay uses
                # the same ``_apply_method_nonlinear`` so fit/transform parity
                # holds without persisting weights for these methods.
                _agg_continuous = _apply_method_nonlinear(Z, method)
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
            # Fit-time column IS the replay output (parity by construction). The
            # discarded-work fast path: ``apply_recipe`` would re-extract the
            # member columns from ``X``, re-nan_to_num, re-standardize with the
            # stored mean/std/signs, re-combine into the SAME ``_agg_continuous``
            # we already hold, then bin it. We already have ``_agg_continuous``
            # and the fit-time ``edges``, so bin directly with the SAME
            # ``searchsorted(edges[1:-1], side="right")`` reduction
            # ``_apply_cluster_aggregate`` uses -> bit-identical column, skipping
            # the full member re-extract/re-standardize/re-combine round trip
            # (9 methods x k cols of pandas getitem + nan_to_num + column_stack
            # per cluster). Only the edges path (the default whenever
            # quantization nbins>=2) is shortcut; the no-quantization / no-edges
            # branch still defers to ``apply_recipe`` for exact behaviour.
            _edges_local = None if _q_local is None else _q_local.get("edges")
            if _edges_local is not None:
                _edges_arr = np.asarray(_edges_local, dtype=np.float64)
                binned = np.searchsorted(
                    _edges_arr[1:-1] if _edges_arr.size >= 2 else _edges_arr,
                    _agg_continuous, side="right",
                ).astype(np.dtype(_q_local["dtype"]))
            else:
                binned = apply_recipe(recipe, X)
            # ``mi`` reads ONLY columns x (the binned aggregate) and y (``target``)
            # via ``merge_vars``; the rest of ``data`` is copied-then-discarded.
            # Stack just the target columns + the binned aggregate into a compact
            # (n, |target|+1) matrix and remap x/y into it -> bit-identical MI
            # (merge_vars depends only on the read columns' per-sample values + their
            # nbins, not on column position), skipping the full (n, n_features) copy
            # that was rebuilt EVERY method-iteration. ~25-89x on this score at
            # realistic shapes (bench: _benchmarks/bench_cluster_aggregate_mi_compact_stack.py).
            _compact = np.column_stack([_target_block, binned.astype(data.dtype)])
            agg_mi = float(mi(_compact, np.array([_n_t], dtype=np.int64), np.arange(_n_t, dtype=np.int64), _compact_nbins, dtype=dtype))
            if best is None or agg_mi > best[0]:
                best = (agg_mi, recipe, binned, method)

        if best is None:
            raise ValueError("_cluster_aggregate: methods must be a non-empty sequence")
        agg_mi, recipe, binned, method = best
        # Gate: the denoised aggregate must STRICTLY out-score the best single member (denoising claim).
        # 2026-06-03 (audit cluster-aggregate-9): reject on a TIE (``<=``), not
        # only on a strict loss (``<``). The docstring/contract says "strictly
        # out-scoring"; with the prior ``<`` an aggregate that merely matches the
        # best member's MI (mi_prevalence=1.0) was accepted, which in
        # mode="replace" swaps real members for a no-gain aggregate. A genuine
        # denoising win must clear the bar with margin.
        if agg_mi <= mi_prevalence * best_member_mi:
            if verbose > 1:
                logger.info("cluster_aggregate: rejected %s (MI %.5f <= %.2f x best member %.5f)", recipe.name, agg_mi, mi_prevalence, best_member_mi)
            continue

        # Accept: append the binned aggregate column to the binned matrix `data` / `cols` / `nbins`
        # (what screening + the remap consume). Deliberately NOT added to the caller's `X`: it is unused
        # there (replay reconstructs the aggregate from the raw member columns via the recipe), and
        # writing to `X` would mutate the caller's frame and leak the engineered name into a later
        # fit's feature_names_in_.
        new_data_cols.append(binned.astype(quantization_dtype))
        added_indices.append(len(cols))  # cols-space index this aggregate will occupy
        cols = [*cols, recipe.name]
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
