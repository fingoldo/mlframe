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

from typing import Dict, Optional, Sequence

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

    Returns
    -------
    pd.DataFrame
        ``n_components * len(methods)`` columns, named ``{column_prefix}_{method}_{i}``, same row count/order
        as ``df``. Meant to be CONCATENATED alongside the original features (additive signal), not used as a
        replacement for them.
    """
    invalid = set(methods) - set(_VALID_METHODS)
    if invalid:
        raise ValueError(f"multi_decomposition_feature_bank: unsupported methods {invalid}, expected subset of {_VALID_METHODS}")

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
    for method in methods:
        projection = _fit_transform(method, X, k, random_state)
        for i in range(projection.shape[1]):
            out[f"{column_prefix}_{method}_{i}"] = projection[:, i]

    return pd.DataFrame(out, index=df.index)


def _fit_transform(method: str, X: np.ndarray, k: int, random_state: int) -> np.ndarray:
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


__all__ = ["multi_decomposition_feature_bank"]
