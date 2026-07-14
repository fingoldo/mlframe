"""``multi_decomposition_feature_bank``: SVD/PCA/ICA/GRP/SRP projections as ADDITIVE stacking input.

Source: 3rd_mercedes-benz-greener-manufacturing.md -- TruncatedSVD, PCA, FastICA, GaussianRandomProjection,
SparseRandomProjection, each producing 10 components, all 50 appended as extra features feeding the base
learners (not just the meta-model). Distinct use from a single PCA-for-dim-reduction step: here the
decompositions are ADDITIVE signal alongside the original features, not a replacement for them -- each method
captures a different geometric structure (SVD/PCA: variance-maximizing linear directions; ICA:
statistically-independent components; GRP/SRP: distance-preserving random projections, cheap and
regularization-like), so their concatenation gives downstream models several complementary "views" of the
same feature space to draw from.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_VALID_METHODS = ("svd", "pca", "ica", "grp", "srp")


def multi_decomposition_feature_bank(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    n_components: int = 10,
    methods: Sequence[str] = _VALID_METHODS,
    random_state: int = 42,
    column_prefix: str = "decomp",
    y: Optional[np.ndarray] = None,
    prune_uninformative_methods: bool = False,
    prune_tolerance: float = 0.02,
    auto_k: bool = False,
    auto_k_variance_ratio: float = 0.95,
) -> pd.DataFrame:
    """Fit several unsupervised decomposition methods and concatenate their projections as new feature columns.

    Parameters
    ----------
    df
        Source frame.
    columns
        Numeric columns to decompose; defaults to every numeric column of ``df``.
    n_components
        Number of components per method (capped at ``min(n_components, len(columns), len(df) - 1)`` for
        methods with a hard rank ceiling -- SVD/PCA/ICA).
    methods
        Subset of ``{"svd", "pca", "ica", "grp", "srp"}``.
    random_state
        Seed shared across all stochastic methods (ICA, GRP, SRP).
    column_prefix
        Output column-name prefix.
    y
        Optional binary target, same row order as ``df``. Required (and only used) when
        ``prune_uninformative_methods=True``.
    prune_uninformative_methods
        Opt-in target-aware pruning pass, OFF by default (default behavior is bit-identical to the plain
        unsupervised bank). When ``True``, each method's whole column GROUP is screened at once with
        ``feature_selection.drop_near_noise_univariate_auc`` against ``y``: a method is dropped entirely only
        when EVERY one of its components independently comes back nea-chance-AUC (no single component beats a
        near-chance AUC band of ``0.5 +/- prune_tolerance``) -- so a method contributing even one informative component survives
        intact. This targets the common failure mode where a shotgun of 5 decompositions includes methods
        (typically ICA on data without independent latent sources, or GRP/SRP on small data) that are pure
        noise for a *given* target, diluting downstream models with useless columns.
    prune_tolerance
        Passed through to ``drop_near_noise_univariate_auc`` -- a component is "near noise" when
        ``abs(auc - 0.5) <= prune_tolerance``.
    auto_k
        Opt-in per-method automatic component-count selection, OFF by default (default behavior is
        bit-identical to passing a single fixed ``n_components`` to every method). When ``True``,
        ``n_components`` becomes a CEILING each method is fit at once, and the number of components actually
        kept is chosen per method: SVD/PCA keep the smallest prefix whose cumulative
        ``explained_variance_ratio_`` clears ``auto_k_variance_ratio`` (closed-form, no extra fitting); ICA/GRP/SRP
        have no such closed-form ratio, so they use a reconstruction-error elbow instead -- inverse-transforming
        successively longer zero-padded component prefixes and picking the point of maximum perpendicular
        distance from the chord joining the 1-component and full-ceiling error (the classic diminishing-returns
        "knee"). This replaces one uniform hand-picked k for every decomposer with a k that matches each
        method's own recovered structure -- e.g. a manifold with true rank 8 needs ~8 SVD/PCA components to hit
        a 0.9 variance-ratio bar, not whatever fixed k happened to be passed for all five methods at once.
    auto_k_variance_ratio
        Only used when ``auto_k=True``. Target cumulative explained-variance ratio for SVD/PCA's closed-form
        selection; unused by ICA/GRP/SRP, whose elbow heuristic needs no target since it picks the curve's
        knee regardless of scale.

    Returns
    -------
    pd.DataFrame
        ``n_components * len(methods)`` columns, named ``{column_prefix}_{method}_{i}``, same row count/order
        as ``df``, MINUS any method dropped by pruning. Meant to be CONCATENATED alongside the original
        features (additive signal), not used as a replacement for them.
    """
    invalid = set(methods) - set(_VALID_METHODS)
    if invalid:
        raise ValueError(f"multi_decomposition_feature_bank: unsupported methods {invalid}, expected subset of {_VALID_METHODS}")
    if prune_uninformative_methods and y is None:
        raise ValueError("multi_decomposition_feature_bank: prune_uninformative_methods=True requires y")
    if auto_k and not (0.0 < auto_k_variance_ratio <= 1.0):
        raise ValueError(f"multi_decomposition_feature_bank: auto_k_variance_ratio must be in (0, 1], got {auto_k_variance_ratio}")

    cols = list(columns) if columns is not None else list(df.select_dtypes(include=[np.number]).columns)
    X = df[cols].to_numpy(dtype=np.float64)
    # Unconditional df.fillna(df.median()) pays a full-frame median reduction even when no NaN is present
    # (same wasted-work pattern as np.nanquantile's unconditional slow path, fixed earlier this session in
    # row_wise_summary.py) -- only compute/apply the median fill when a NaN is actually there.
    if np.isnan(X).any():
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X = np.where(nan_mask, col_medians[np.newaxis, :], X)
    n_samples, n_features = X.shape
    k = max(1, min(n_components, n_features, n_samples - 1))

    out: Dict[str, np.ndarray] = {}
    method_columns: Dict[str, List[str]] = {}
    for method in methods:
        if auto_k:
            projection = _fit_transform_auto_k(method, X, k, random_state, auto_k_variance_ratio)
        else:
            projection = _fit_transform(method, X, k, random_state)
        method_cols = []
        for i in range(projection.shape[1]):
            col_name = f"{column_prefix}_{method}_{i}"
            out[col_name] = projection[:, i]
            method_cols.append(col_name)
        method_columns[method] = method_cols

    bank = pd.DataFrame(out, index=df.index)
    if prune_uninformative_methods:
        from mlframe.feature_selection.drop_near_noise_univariate_auc import drop_near_noise_univariate_auc

        assert y is not None  # guaranteed by the prune_uninformative_methods/y validation above
        near_noise = set(drop_near_noise_univariate_auc(bank, y, columns=list(out.keys()), tolerance=prune_tolerance))
        keep_cols: List[str] = []
        for method_cols in method_columns.values():
            if all(c in near_noise for c in method_cols):
                continue  # every component of this method is near-chance -- drop the whole method group
            keep_cols.extend(method_cols)
        bank = bank[keep_cols]

    return bank


def _fit_transform(method: str, X: np.ndarray, k: int, random_state: int) -> np.ndarray:
    """Fit the named decomposition method with a fixed ``k`` components and return the projection."""
    if method == "svd":
        from sklearn.decomposition import TruncatedSVD

        return np.asarray(TruncatedSVD(n_components=k, random_state=random_state).fit_transform(X))
    if method == "pca":
        from sklearn.decomposition import PCA

        return np.asarray(PCA(n_components=k, random_state=random_state).fit_transform(X))
    if method == "ica":
        from sklearn.decomposition import FastICA

        return np.asarray(FastICA(n_components=k, random_state=random_state, max_iter=500).fit_transform(X))
    if method == "grp":
        from sklearn.random_projection import GaussianRandomProjection

        return np.asarray(GaussianRandomProjection(n_components=k, random_state=random_state).fit_transform(X))
    if method == "srp":
        from sklearn.random_projection import SparseRandomProjection

        return np.asarray(SparseRandomProjection(n_components=k, random_state=random_state).fit_transform(X))
    raise ValueError(f"_fit_transform: unknown method {method!r}")  # pragma: no cover -- guarded by caller


def _select_k_by_variance_threshold(cum_ratios: np.ndarray, threshold: float) -> int:
    """Smallest k (1-indexed) whose cumulative explained-variance ratio clears ``threshold``, capped at ``len(cum_ratios)``."""
    idx = int(np.searchsorted(cum_ratios, threshold))
    return min(idx + 1, len(cum_ratios))


def _select_k_by_reconstruction_elbow(errors: np.ndarray) -> int:
    """Kneedle-style elbow: 1-indexed k of the point farthest (perpendicular distance) from the chord joining
    the first and last points of a monotonic reconstruction-error curve -- the classic diminishing-returns
    "knee", used for ICA/GRP/SRP which have no closed-form explained-variance-ratio attribute."""
    n = len(errors)
    if n <= 2:
        return n
    x = np.arange(1, n + 1, dtype=np.float64)
    y = errors.astype(np.float64)
    x_span = x[-1] - x[0]
    y_span = y[-1] - y[0]
    x_n = (x - x[0]) / x_span if x_span > 1e-12 else np.zeros_like(x)
    y_n = (y - y[0]) / y_span if abs(y_span) > 1e-12 else np.zeros_like(y)
    p1 = np.array([x_n[0], y_n[0]])
    line_vec = np.array([x_n[-1], y_n[-1]]) - p1
    line_len = float(np.linalg.norm(line_vec))
    if line_len < 1e-12:
        return n
    line_unit = line_vec / line_len
    points = np.stack([x_n, y_n], axis=1) - p1
    proj_len = points @ line_unit
    closest = np.outer(proj_len, line_unit)
    dists = np.linalg.norm(points - closest, axis=1)
    return int(np.argmax(dists)) + 1


def _reconstruction_errors(X: np.ndarray, projection: np.ndarray, inverse_transform: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Per-prefix-length total squared reconstruction error (index i -> k=i+1 components kept, rest zeroed), decreasing in k."""
    k_max = projection.shape[1]
    errors = np.empty(k_max, dtype=np.float64)
    padded = np.zeros_like(projection)
    for k in range(1, k_max + 1):
        padded[:, :k] = projection[:, :k]
        X_hat = np.asarray(inverse_transform(padded))
        errors[k - 1] = float(np.sum((X - X_hat) ** 2))
    return errors


def _fit_transform_auto_k(method: str, X: np.ndarray, k_max: int, random_state: int, variance_ratio: float) -> np.ndarray:
    """Fit the named decomposition method at ``k_max`` components, then trim to the auto-selected k."""
    if method == "svd":
        from sklearn.decomposition import TruncatedSVD

        model_svd = TruncatedSVD(n_components=k_max, random_state=random_state)
        projection = np.asarray(model_svd.fit_transform(X))
        k_selected = _select_k_by_variance_threshold(np.cumsum(model_svd.explained_variance_ratio_), variance_ratio)
        return projection[:, :k_selected]
    if method == "pca":
        from sklearn.decomposition import PCA

        model_pca = PCA(n_components=k_max, random_state=random_state)
        projection = np.asarray(model_pca.fit_transform(X))
        k_selected = _select_k_by_variance_threshold(np.cumsum(model_pca.explained_variance_ratio_), variance_ratio)
        return projection[:, :k_selected]
    if method == "ica":
        from sklearn.decomposition import FastICA

        model_ica = FastICA(n_components=k_max, random_state=random_state, max_iter=500)
        projection = np.asarray(model_ica.fit_transform(X))
        errors = _reconstruction_errors(X, projection, model_ica.inverse_transform)
        return projection[:, : _select_k_by_reconstruction_elbow(errors)]
    if method == "grp":
        from sklearn.random_projection import GaussianRandomProjection

        model_grp = GaussianRandomProjection(n_components=k_max, random_state=random_state, compute_inverse_components=True)
        projection = np.asarray(model_grp.fit_transform(X))
        errors = _reconstruction_errors(X, projection, model_grp.inverse_transform)
        return projection[:, : _select_k_by_reconstruction_elbow(errors)]
    if method == "srp":
        from sklearn.random_projection import SparseRandomProjection

        model_srp = SparseRandomProjection(n_components=k_max, random_state=random_state, compute_inverse_components=True)
        projection = np.asarray(model_srp.fit_transform(X))
        errors = _reconstruction_errors(X, projection, model_srp.inverse_transform)
        return projection[:, : _select_k_by_reconstruction_elbow(errors)]
    raise ValueError(f"_fit_transform_auto_k: unknown method {method!r}")  # pragma: no cover -- guarded by caller


__all__ = ["multi_decomposition_feature_bank"]
