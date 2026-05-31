"""Layer 61 (2026-05-31): PER-CLUSTER SHARED-BASIS FE.

Why this layer
--------------

Layer 21 emits ``basis_n(x)`` per individual source column. When several
source columns are noisy reflections of one shared latent (the standard
cluster scenario: ``s_i = z + epsilon_i`` for i in cluster), the per-
member ``He_n(x_i)`` carries the SAME signal n times -- each diluted by
epsilon_i. Layer 27 (collinear-dedup) avoids the literal duplicate
inflation, and Layer 7's ``cluster_aggregate`` swaps the cluster down to
its PC1 / mean_z aggregate as a NEW raw feature; but neither path runs an
orthogonal-polynomial basis expansion ON the aggregate.

The aggregate ``a = mean_z(cluster_members)`` denoises by averaging out
the epsilon_i terms; ``He_n(a)`` then catches non-linear signal in the
LATENT that survives the denoising. That signal dominates the per-member
``He_n(s_i)`` columns because the noise has been averaged out before the
basis expansion is applied.

Concretely: when y depends on ``He_2(z)`` and the cluster has size 3 with
noise_sd 0.5, the per-member ``He_2(s_i)`` has signal-to-noise ratio
1/3 of ``He_2(mean_z(s_0, s_1, s_2))`` (the variance of the noise in the
aggregate is 1/3 of the per-member noise variance under iid noise).

What this layer adds
--------------------

* :func:`detect_clusters_by_correlation` -- lightweight cluster discovery
  from the X DataFrame: bulk Pearson corrcoef + connected-components on
  the |corr| >= threshold edge set. Mirrors L59's pair detection but
  groups members into transitive components instead of pairs.

* :func:`compute_cluster_aggregate` -- single-column per-cluster
  reduction. ``mean_z`` (default) z-standardises every member then
  averages; ``median_z`` does median over z-scores (robust against
  outlier members); ``pc1`` returns the leading SVD direction in z-space
  (the unidimensional latent under the iid-noise reflection model).

* :func:`generate_cluster_basis_features` -- per cluster, evaluate
  ``basis_d(preprocess(aggregate))`` for each requested degree, run the
  MI-uplift + noise-aware-MAD floor two-gate selection, and keep top-K.
  Per-aggregate baseline is ``max(MI(member; y) for member in cluster)``
  -- the aggregate basis only wins when it beats the BEST individual
  member's marginal MI.

* :func:`hybrid_orth_mi_cluster_basis_fe`,
  :func:`hybrid_orth_mi_cluster_basis_fe_with_recipes` -- the standard
  pair-of-entry-points pattern shared with Layer 21 / 25 / 59. The
  ``_with_recipes`` variant returns ``orth_cluster_basis`` recipes whose
  ``src_names`` is the full member tuple; replay at transform time
  re-computes the aggregate from the SAME members and evaluates the same
  basis_degree.

NOT wired into ``MRMR.fit`` by default -- explicit opt-in via
``fe_hybrid_orth_cluster_basis_enable=True``. The wiring lives in
``_mrmr_fit_impl.py`` (alongside the Layer 59 diff-basis block).

Recipe replay
-------------

Each emitted column is backed by an ``orth_cluster_basis`` recipe whose
``extra`` carries ``{basis, degree, aggregator}``. ``src_names`` is the
ordered tuple of cluster members (sorted by name for deterministic
identity). The aggregate is ALWAYS computed via the recipe-stored
aggregator so the test-time orientation matches fit time exactly.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .hermite_fe import _POLY_BASES
from ._orthogonal_univariate_fe import (
    _evaluate_basis_column,
    _mi_classif_batch,
    _BASIS_CODE,
)

logger = logging.getLogger(__name__)

__all__ = [
    "detect_clusters_by_correlation",
    "compute_cluster_aggregate",
    "generate_cluster_basis_features",
    "hybrid_orth_mi_cluster_basis_fe",
    "hybrid_orth_mi_cluster_basis_fe_with_recipes",
]

_VALID_AGGREGATORS = ("mean_z", "median_z", "pc1")


def _cluster_col_name(anchor: str, aggregator: str, basis: str, degree: int) -> str:
    """Stable engineered column name for a cluster-basis column.

    Format: ``cluster_{anchor}__agg_{aggregator}__{code}{degree}``. The
    ``cluster_`` prefix makes the kind visible at a glance and prevents
    collisions with Layer 21's ``{col}__He{d}`` naming.
    """
    code = _BASIS_CODE.get(basis, basis)
    return f"cluster_{anchor}__agg_{aggregator}__{code}{int(degree)}"


def _connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Union-find connected components over n nodes given an edge list.

    Returns components of length >= 2 (singletons -- nodes with no edges --
    are silently dropped because we only want true clusters).
    """
    parent = list(range(n))

    def find(x: int) -> int:
        # Path compression for amortised near-constant find.
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return [c for c in comps.values() if len(c) >= 2]


def detect_clusters_by_correlation(
    X: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    *,
    corr_threshold: float = 0.7,
    min_cluster_size: int = 2,
    max_cluster_size: int = 20,
) -> dict[str, list[str]]:
    """Lightweight cluster discovery: connected components of the |corr|
    >= threshold graph over the dense numeric columns of X.

    Bulk ``np.corrcoef`` once, then a Python-level union-find pass --
    O(p^2) edges in the worst case but only one C call for the correlation
    matrix. At p=200 this is ~50 ms.

    Parameters
    ----------
    X : DataFrame
        Source frame.
    cols : sequence of column names, optional
        Restrict the scan. ``None`` = all numeric columns.
    corr_threshold : float, default 0.7
        Minimum absolute Pearson correlation for an edge. The cluster
        scenario this layer targets (noisy reflections of one latent)
        produces |corr| in [0.85, 0.99]; lower thresholds let in
        spuriously-correlated pairs that add basis noise.
    min_cluster_size : int, default 2
        Drop components smaller than this. ``2`` keeps every detected
        pair as a candidate cluster; raise to ``3`` to require at least
        triple confirmation before running an aggregate basis.
    max_cluster_size : int, default 20
        Cap on members per cluster; if a component exceeds this, keep
        the ``max_cluster_size`` members with the highest mean |corr| to
        the rest of the cluster. Bounds the per-aggregate row-reduction
        cost on adversarially-collinear frames.

    Returns
    -------
    dict mapping ``anchor_name -> sorted list of member column names``.
    The anchor is the lexicographically-smallest member of the cluster
    (deterministic, no y reference). Empty dict if no cluster meets
    ``min_cluster_size``.
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cols = [
        c for c in cols
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]
    if len(cols) < min_cluster_size:
        return {}
    dense_arrays: list[np.ndarray] = []
    dense_names: list[str] = []
    for c in cols:
        arr = np.asarray(X[c].to_numpy(), dtype=np.float64)
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        if not finite.all():
            arr = np.where(finite, arr, float(np.nanmean(arr[finite])))
        if float(arr.std()) <= 1e-12:
            continue
        dense_arrays.append(arr)
        dense_names.append(c)
    if len(dense_names) < min_cluster_size:
        return {}
    mat = np.vstack(dense_arrays)
    corr = np.corrcoef(mat)
    if corr.ndim == 0:
        return {}
    abs_corr = np.abs(corr)
    p = len(dense_names)
    edges: list[tuple[int, int]] = []
    for i in range(p):
        for j in range(i + 1, p):
            c_ij = float(abs_corr[i, j])
            if not np.isfinite(c_ij):
                continue
            # Guard against literal duplicates -- their aggregate equals
            # any member, so He_n(aggregate) collides with He_n(member)
            # already emitted by Layer 21 (when enabled).
            if c_ij >= corr_threshold and c_ij < 1.0 - 1e-12:
                edges.append((i, j))
    comps = _connected_components(p, edges)
    out: dict[str, list[str]] = {}
    for comp in comps:
        members = sorted(dense_names[idx] for idx in comp)
        if len(members) < min_cluster_size:
            continue
        if len(members) > int(max_cluster_size):
            # Rank by mean |corr| to the rest of the cluster, keep the
            # top max_cluster_size. Deterministic tie-break on name.
            comp_indices = [dense_names.index(m) for m in members]
            mean_corr = []
            for idx in comp_indices:
                others = [k for k in comp_indices if k != idx]
                mean_corr.append((
                    -float(abs_corr[idx, others].mean()),
                    dense_names[idx],
                ))
            mean_corr.sort()
            members = sorted(
                dense_names[dense_names.index(name)]
                for (_, name) in mean_corr[: int(max_cluster_size)]
            )
        anchor = members[0]
        out[anchor] = members
    return out


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Standardise (x - mean) / std with NaN-safe fill and constant guard."""
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.all():
        fill = float(np.nanmean(arr[finite])) if finite.any() else 0.0
        arr = np.where(finite, arr, fill)
    mu = float(arr.mean())
    sd = float(arr.std())
    if sd <= 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def compute_cluster_aggregate(
    X: pd.DataFrame,
    members: Sequence[str],
    *,
    aggregator: str = "mean_z",
) -> np.ndarray:
    """Reduce a cluster of member columns to one column via ``aggregator``.

    Parameters
    ----------
    X : DataFrame
        Source frame; ``members`` must all be present and numeric.
    members : sequence of column names
        The cluster members to aggregate.
    aggregator : {'mean_z', 'median_z', 'pc1'}, default 'mean_z'
        * mean_z   : average over standardised (z-scored) members. Denoises
                     the shared latent ``z`` from iid additive noise; the
                     standard reflection-cluster reduction.
        * median_z : median over standardised members; robust against
                     outlier members carrying a non-shared signal.
        * pc1      : leading SVD direction in z-space (per-member std-aligned
                     to the FIRST member's sign so the orientation is
                     deterministic across pickle / clone).

    Returns
    -------
    1-D ndarray of length ``len(X)`` with the aggregated column.
    """
    if aggregator not in _VALID_AGGREGATORS:
        raise ValueError(
            f"compute_cluster_aggregate: unknown aggregator {aggregator!r}; "
            f"expected one of {_VALID_AGGREGATORS}."
        )
    if not members:
        return np.zeros(len(X), dtype=np.float64)
    zs = [_zscore(np.asarray(X[m].to_numpy(), dtype=np.float64)) for m in members]
    Z = np.column_stack(zs)
    if aggregator == "mean_z":
        return Z.mean(axis=1)
    if aggregator == "median_z":
        return np.median(Z, axis=1)
    # pc1
    # Center then SVD; leading right-singular-vector projection.
    Zc = Z - Z.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    except np.linalg.LinAlgError:
        return Z.mean(axis=1)
    if S.size == 0 or float(S[0]) <= 1e-12:
        return Z.mean(axis=1)
    pc1 = U[:, 0] * S[0]
    # Sign-align so the leading correlation with the first member is
    # positive; deterministic orientation across reruns / pickles.
    if float(np.dot(pc1, Z[:, 0])) < 0.0:
        pc1 = -pc1
    return pc1


def generate_cluster_basis_features(
    X: pd.DataFrame,
    y: np.ndarray,
    cluster_members: dict[str, Sequence[str]],
    *,
    basis: str = "hermite",
    degrees: Sequence[int] = (2, 3),
    aggregator: str = "mean_z",
    top_k: int = 3,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """For each cluster, compute the aggregate column then evaluate
    ``basis_degree(preprocess(aggregate))`` for each requested degree.

    Rank by MI uplift against the BEST individual cluster member's
    marginal MI; apply the same two-gate (relative uplift + noise-aware
    MAD floor) selection used by Layer 21 / 25 / 59. Returns the global
    top-K winners across all (cluster, degree) cells.

    Parameters
    ----------
    X : DataFrame
        Source frame.
    y : array-like (n,)
        Discrete (binary or low-cardinality int) target. Continuous y
        must be binned by the caller (the MRMR wiring uses ``pd.qcut``).
    cluster_members : dict[str, sequence[str]]
        ``anchor_name -> list of member column names``. Comes from either
        :func:`detect_clusters_by_correlation` (the fully-internal path)
        or from ``MRMR.cluster_members_`` after a DCD fit (the wired
        path).
    basis : {'hermite', 'legendre', 'chebyshev', 'laguerre'}, default 'hermite'
        Polynomial family. ``hermite`` is the right choice for a
        z-scored aggregate (the standardised input matches the Hermite
        weight). The other bases are kept for explicit caller selection.
    degrees : sequence of int, default ``(2, 3)``
        Polynomial degrees to emit per cluster. Degree 1 is the
        aggregate itself (modulo z-scoring) and adds no signal beyond
        what the aggregate column would already provide.
    aggregator : {'mean_z', 'median_z', 'pc1'}, default 'mean_z'
        Cluster aggregation rule. See :func:`compute_cluster_aggregate`.
    top_k : int, default 3
        Global cap on the number of winners. Per-cluster ties are
        broken by uplift; on identical uplift, lexicographic ordering
        on the engineered name produces a deterministic order.
    min_uplift, min_abs_mi_frac, nbins
        Standard two-gate selection thresholds, mirroring Layer 21 / 59
        semantics.

    Returns
    -------
    (engineered_X, meta)
        engineered_X : DataFrame of surviving top-K cluster-basis columns.
        meta : dict mapping each emitted column name to its full
            ``{anchor, members, basis, degree, aggregator, engineered_mi,
            baseline_mi, uplift}`` payload for recipe replay and
            diagnostics.
    """
    if basis not in _POLY_BASES:
        raise ValueError(
            f"generate_cluster_basis_features: unknown basis {basis!r}; "
            f"expected one of {sorted(_POLY_BASES.keys())}."
        )
    if aggregator not in _VALID_AGGREGATORS:
        raise ValueError(
            f"generate_cluster_basis_features: unknown aggregator "
            f"{aggregator!r}; expected one of {_VALID_AGGREGATORS}."
        )
    degrees = tuple(int(d) for d in degrees)
    if not degrees or not cluster_members:
        return pd.DataFrame(index=X.index), {}

    y_arr = (
        np.asarray(y).astype(np.int64)
        if not np.issubdtype(np.asarray(y).dtype, np.integer)
        else np.asarray(y, dtype=np.int64)
    )

    # Filter clusters down to members that are actually present and numeric
    # in X. Sort members deterministically (lexicographic) so the recipe
    # src_names tuple is stable across pickle / clone.
    cleaned: dict[str, tuple[str, ...]] = {}
    for anchor, members in cluster_members.items():
        if not members:
            continue
        kept = tuple(sorted(
            m for m in members
            if m in X.columns and pd.api.types.is_numeric_dtype(X[m])
        ))
        if len(kept) < 2:
            continue
        cleaned[str(anchor)] = kept
    if not cleaned:
        return pd.DataFrame(index=X.index), {}

    # Baseline MI per member (one batch call across the union of all members).
    touched = sorted({m for members in cleaned.values() for m in members})
    raw_mat = X[touched].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(raw_mat)
    if not finite.all():
        col_means = np.where(
            finite.any(axis=0),
            np.where(finite, raw_mat, 0.0).sum(axis=0)
            / np.maximum(finite.sum(axis=0), 1),
            0.0,
        )
        raw_mat = np.where(finite, raw_mat, col_means[None, :])
    raw_mi = _mi_classif_batch(raw_mat, y_arr, nbins=nbins)
    raw_mi_map = dict(zip(touched, raw_mi.tolist()))

    # ---- Step 2: enumerate (cluster, degree) cells.
    cand_cols: list[str] = []
    cand_values: list[np.ndarray] = []
    cand_meta: list[dict] = []
    for anchor, members in cleaned.items():
        try:
            agg = compute_cluster_aggregate(X, members, aggregator=aggregator)
        except Exception as exc:
            logger.warning(
                "generate_cluster_basis_features: aggregator=%r on cluster "
                "%r raised %r; skipping cluster.",
                aggregator, anchor, exc,
            )
            continue
        if float(np.std(agg)) <= 1e-12:
            continue
        for d in degrees:
            try:
                vals = _evaluate_basis_column(agg, basis, int(d))
            except Exception as exc:
                logger.warning(
                    "generate_cluster_basis_features: basis=%r degree=%d on "
                    "cluster %r raised %r; skipping cell.",
                    basis, d, anchor, exc,
                )
                continue
            if not np.isfinite(vals).all():
                continue
            if float(np.std(vals)) <= 1e-12:
                continue
            cand_cols.append(_cluster_col_name(anchor, aggregator, basis, int(d)))
            cand_values.append(vals)
            cand_meta.append({
                "anchor": str(anchor),
                "members": tuple(members),
                "basis": str(basis),
                "degree": int(d),
                "aggregator": str(aggregator),
            })

    if not cand_cols:
        return pd.DataFrame(index=X.index), {}

    # ---- Step 3: ONE batch MI call across every candidate.
    cand_mat = np.column_stack(cand_values).astype(np.float64, copy=False)
    eng_mi = _mi_classif_batch(cand_mat, y_arr, nbins=nbins)

    # ---- Step 4: uplift gate (vs MAX cluster-member baseline) + noise-aware
    # MAD floor + global top-K.
    raw_baselines = np.asarray(list(raw_mi_map.values()), dtype=np.float64)
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max_raw_baseline
    n_baselines = int(raw_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * max(n_baselines, 1)))) + 1.5),
    )
    if n_baselines >= 16:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        noise_floor = med + sigma_thresh * 1.4826 * mad
    else:
        noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor)

    survivors: list[dict] = []
    for j, info in enumerate(cand_meta):
        emi = float(eng_mi[j])
        if not np.isfinite(emi):
            continue
        baseline = max(
            (float(raw_mi_map.get(m, 0.0)) for m in info["members"]),
            default=0.0,
        )
        uplift = emi / (baseline + 1e-12)
        if uplift < float(min_uplift):
            continue
        if emi < abs_floor:
            continue
        survivors.append({
            "engineered_col": cand_cols[j],
            "values_idx": j,
            "engineered_mi": emi,
            "baseline_mi": baseline,
            "uplift": float(uplift),
            **info,
        })

    # Deterministic order: uplift desc, then engineered_col asc to break ties.
    survivors.sort(key=lambda d: (-d["uplift"], d["engineered_col"]))
    winners = survivors[: int(top_k)]

    out_cols: dict = {}
    meta: dict = {}
    for info in winners:
        name = str(info["engineered_col"])
        vals = cand_values[int(info["values_idx"])]
        out_cols[name] = vals
        meta[name] = {
            "anchor": str(info["anchor"]),
            "members": tuple(info["members"]),
            "basis": str(info["basis"]),
            "degree": int(info["degree"]),
            "aggregator": str(info["aggregator"]),
            "engineered_mi": float(info["engineered_mi"]),
            "baseline_mi": float(info["baseline_mi"]),
            "uplift": float(info["uplift"]),
        }
    return pd.DataFrame(out_cols, index=X.index), meta


def hybrid_orth_mi_cluster_basis_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cluster_members: Optional[dict[str, Sequence[str]]] = None,
    cols: Optional[Sequence[str]] = None,
    basis: str = "hermite",
    degrees: Sequence[int] = (2, 3),
    aggregator: str = "mean_z",
    corr_threshold: float = 0.7,
    min_cluster_size: int = 2,
    max_cluster_size: int = 20,
    top_k: int = 3,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end cluster-basis hybrid.

    When ``cluster_members`` is None, run the internal
    :func:`detect_clusters_by_correlation` on ``X`` (restricted to ``cols``
    if provided); otherwise use the caller-supplied groupings (e.g. from
    ``MRMR.cluster_members_`` after a DCD fit).

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the surviving top-K cluster-basis columns
            appended.
        scores : DataFrame with columns ``[engineered_col, anchor,
            members, basis, degree, aggregator, baseline_mi,
            engineered_mi, uplift]`` ordered by ``uplift`` descending.
    """
    if cluster_members is None:
        cluster_members = detect_clusters_by_correlation(
            X, cols,
            corr_threshold=corr_threshold,
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
        )
    engineered, meta = generate_cluster_basis_features(
        X, y, cluster_members,
        basis=basis,
        degrees=degrees,
        aggregator=aggregator,
        top_k=top_k,
        min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        nbins=nbins,
    )
    if engineered.empty:
        scores_empty = pd.DataFrame(columns=[
            "engineered_col", "anchor", "members", "basis", "degree",
            "aggregator", "baseline_mi", "engineered_mi", "uplift",
        ])
        return X.copy(), scores_empty
    rows = []
    for name, info in meta.items():
        rows.append({
            "engineered_col": name,
            "anchor": info["anchor"],
            "members": info["members"],
            "basis": info["basis"],
            "degree": info["degree"],
            "aggregator": info["aggregator"],
            "baseline_mi": info["baseline_mi"],
            "engineered_mi": info["engineered_mi"],
            "uplift": info["uplift"],
        })
    scores = pd.DataFrame(rows)
    if not scores.empty:
        scores = scores.sort_values("uplift", ascending=False).reset_index(drop=True)
    keep = list(engineered.columns)
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_cluster_basis_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cluster_members: Optional[dict[str, Sequence[str]]] = None,
    cols: Optional[Sequence[str]] = None,
    basis: str = "hermite",
    degrees: Sequence[int] = (2, 3),
    aggregator: str = "mean_z",
    corr_threshold: float = 0.7,
    min_cluster_size: int = 2,
    max_cluster_size: int = 20,
    top_k: int = 3,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_cluster_basis_fe` but additionally
    returns a list of ``EngineeredRecipe`` objects (kind
    ``"orth_cluster_basis"``) so ``MRMR.transform`` can replay each
    appended column without re-running cluster detection or MI scoring.

    Returns
    -------
    (X_augmented, scores, recipes)
    """
    from .engineered_recipes import build_orth_cluster_basis_recipe
    X_aug, scores = hybrid_orth_mi_cluster_basis_fe(
        X, y,
        cluster_members=cluster_members,
        cols=cols,
        basis=basis,
        degrees=degrees,
        aggregator=aggregator,
        corr_threshold=corr_threshold,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        top_k=top_k,
        min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    if not appended:
        return X_aug, scores, []
    name_to_row = {
        str(row["engineered_col"]): row for _, row in scores.iterrows()
    }
    recipes = []
    for name in appended:
        row = name_to_row.get(name)
        if row is None:
            logger.warning(
                "hybrid_orth_mi_cluster_basis_fe_with_recipes: appended "
                "column %r missing from scores; skipping recipe.", name,
            )
            continue
        recipes.append(build_orth_cluster_basis_recipe(
            name=name,
            members=tuple(str(m) for m in row["members"]),
            basis=str(row["basis"]),
            degree=int(row["degree"]),
            aggregator=str(row["aggregator"]),
        ))
    return X_aug, scores, recipes


def _apply_orth_cluster_basis(recipe, X) -> np.ndarray:
    """Replay a cluster-basis column. Lazy-imported by the recipes
    dispatcher to keep the recipes module under the LOC budget.

    Stateless given ``(members, basis, degree, aggregator)``; no y
    reference. The aggregate is recomputed from the SAME members and
    the same aggregator so train/test orientation parity holds row-by-
    row.
    """
    from .engineered_recipes import _extract_column
    if len(recipe.src_names) < 2:
        raise ValueError(
            f"orth_cluster_basis recipe '{recipe.name}' must have >=2 "
            f"src_names (cluster members); got {len(recipe.src_names)}"
        )
    for key in ("basis", "degree", "aggregator"):
        if key not in recipe.extra:
            raise KeyError(
                f"orth_cluster_basis recipe '{recipe.name}' missing "
                f"'{key}' in extra. Re-fit MRMR to regenerate."
            )
    basis = str(recipe.extra["basis"])
    degree = int(recipe.extra["degree"])
    aggregator = str(recipe.extra["aggregator"])
    if aggregator not in _VALID_AGGREGATORS:
        raise ValueError(
            f"orth_cluster_basis recipe '{recipe.name}' references unknown "
            f"aggregator {aggregator!r}; expected one of {_VALID_AGGREGATORS}."
        )
    # Reconstruct a minimal frame with only the cluster members so the
    # aggregate helper can z-score each column independently. We pull each
    # member via the recipe-aware ``_extract_column`` helper to support
    # pandas / polars / structured arrays uniformly.
    member_cols = {}
    for m in recipe.src_names:
        member_cols[str(m)] = np.asarray(
            _extract_column(X, str(m)), dtype=np.float64,
        )
    # Build a DataFrame-like dict with stable column order matching
    # recipe.src_names for byte-exact reproducibility.
    frame = pd.DataFrame(member_cols)
    agg = compute_cluster_aggregate(
        frame, list(recipe.src_names), aggregator=aggregator,
    )
    return _evaluate_basis_column(agg, basis, degree)
