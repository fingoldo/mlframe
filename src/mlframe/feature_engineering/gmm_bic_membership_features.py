"""``gmm_bic_membership_features``: BIC-selected GMM cluster-membership-probability features.

Source: 8th_instant-gratification.md -- "I used GMM with 4 and 6 components as additional features... I
chose those parameters based on BIC." A general, label-free unsupervised featurizer: fit a
``GaussianMixture`` over a range of candidate component counts, pick the count minimizing BIC (the standard
model-complexity-penalized selection criterion, avoiding an arbitrary/overfit component count), and emit
each row's per-component posterior membership probability as a new feature block -- a density-based view of
the feature space complementary to distance/neighbor-based featurizers.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def gmm_bic_membership_features(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    n_components_range: Sequence[int] = (2, 3, 4, 5, 6, 8),
    random_state: int = 42,
    column_prefix: str = "gmm",
) -> pd.DataFrame:
    """Fit a BIC-selected ``GaussianMixture`` and emit per-component membership-probability columns.

    Parameters
    ----------
    df
        Source frame.
    columns
        Numeric columns to fit the GMM on; defaults to every numeric column of ``df``.
    n_components_range
        Candidate component counts to sweep; the one minimizing BIC is selected.
    random_state
        Seed for the GMM fit.
    column_prefix
        Output column-name prefix.

    Returns
    -------
    pd.DataFrame
        ``n_components`` columns (the BIC-selected count), named ``{column_prefix}_component_{i}``, each
        row's posterior probability of belonging to that mixture component (rows sum to 1 across columns).
    """
    from sklearn.mixture import GaussianMixture

    cols = list(columns) if columns is not None else list(df.select_dtypes(include=[np.number]).columns)
    X = df[cols].to_numpy(dtype=np.float64)

    best_bic = np.inf
    best_gmm = None
    for k in n_components_range:
        if k > len(X):
            continue
        gmm = GaussianMixture(n_components=k, random_state=random_state)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    if best_gmm is None:
        raise ValueError("gmm_bic_membership_features: no valid n_components in n_components_range fits the data (all exceed n_rows)")

    membership = best_gmm.predict_proba(X)
    out: Dict[str, np.ndarray] = {f"{column_prefix}_component_{i}": membership[:, i] for i in range(membership.shape[1])}
    return pd.DataFrame(out, index=df.index)


__all__ = ["gmm_bic_membership_features"]
