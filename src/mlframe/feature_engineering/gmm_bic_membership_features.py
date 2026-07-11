"""``gmm_bic_membership_features``: BIC-selected GMM cluster-membership-probability features.

Source: 8th_instant-gratification.md -- "I used GMM with 4 and 6 components as additional features... I
chose those parameters based on BIC." A general, label-free unsupervised featurizer: fit a
``GaussianMixture`` over a range of candidate component counts, pick the count minimizing BIC (the standard
model-complexity-penalized selection criterion, avoiding an arbitrary/overfit component count), and emit
each row's per-component posterior membership probability as a new feature block -- a density-based view of
the feature space complementary to distance/neighbor-based featurizers.

``new_df`` (opt-in) lets the GMM be fit once on ``df`` and then applied to held-out/new data, with a
train-vs-new average-log-likelihood shift diagnostic attached via ``out.attrs["gmm_shift_diagnostics"]`` --
without it, GMM membership probabilities on out-of-distribution rows silently degrade into noise with no
signal that anything went wrong.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd


def gmm_bic_membership_features(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    n_components_range: Sequence[int] = (2, 3, 4, 5, 6, 8),
    random_state: int = 42,
    column_prefix: str = "gmm",
    new_df: Optional[pd.DataFrame] = None,
    shift_zscore_threshold: float = 2.0,
) -> pd.DataFrame:
    """Fit a BIC-selected ``GaussianMixture`` and emit per-component membership-probability columns.

    Parameters
    ----------
    df
        Source frame the GMM is fit on.
    columns
        Numeric columns to fit the GMM on; defaults to every numeric column of ``df``.
    n_components_range
        Candidate component counts to sweep; the one minimizing BIC is selected.
    random_state
        Seed for the GMM fit.
    column_prefix
        Output column-name prefix.
    new_df
        Opt-in: held-out/new data to score with the ``df``-fitted GMM instead of ``df`` itself (e.g. a
        test/prod batch). When given, the returned membership columns describe ``new_df`` rows, and
        ``out.attrs["gmm_shift_diagnostics"]`` is populated with a train-vs-new average-log-likelihood
        comparison (``train_avg_loglik``, ``new_avg_loglik``, ``shift_zscore``,
        ``distribution_shift_detected``) so silently-unreliable membership probabilities under covariate
        shift are surfaced rather than passed through unflagged. When ``None`` (default), behavior is
        unchanged from before this parameter existed: fit and transform ``df`` itself, no diagnostics.
    shift_zscore_threshold
        Only used when ``new_df`` is given. The z-score (train-vs-new avg-log-likelihood gap, normalized
        by the standard error of the training per-row log-likelihoods) above which
        ``distribution_shift_detected`` is flagged ``True``.

    Returns
    -------
    pd.DataFrame
        ``n_components`` columns (the BIC-selected count), named ``{column_prefix}_component_{i}``, each
        row's posterior probability of belonging to that mixture component (rows sum to 1 across columns).
        Describes ``df`` rows by default, or ``new_df`` rows when ``new_df`` is given.
    """
    from sklearn.mixture import GaussianMixture

    cols = list(columns) if columns is not None else list(df.select_dtypes(include=[np.number]).columns)
    X = df[cols].to_numpy(dtype=np.float64)

    best_bic = np.inf
    best_gmm: Optional[GaussianMixture] = None
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

    if new_df is None:
        membership = best_gmm.predict_proba(X)
        out: Dict[str, np.ndarray] = {f"{column_prefix}_component_{i}": membership[:, i] for i in range(membership.shape[1])}
        return pd.DataFrame(out, index=df.index)

    X_new = new_df[cols].to_numpy(dtype=np.float64)
    membership = best_gmm.predict_proba(X_new)
    out = {f"{column_prefix}_component_{i}": membership[:, i] for i in range(membership.shape[1])}
    result = pd.DataFrame(out, index=new_df.index)

    train_scores = best_gmm.score_samples(X)
    new_scores = best_gmm.score_samples(X_new)
    train_avg_loglik = float(train_scores.mean())
    new_avg_loglik = float(new_scores.mean())
    train_std = float(train_scores.std(ddof=1)) if len(train_scores) > 1 else 0.0
    standard_error = train_std / np.sqrt(len(new_scores)) if len(new_scores) > 0 else 0.0
    shift_zscore = (train_avg_loglik - new_avg_loglik) / standard_error if standard_error > 0 else 0.0

    diagnostics: Dict[str, Any] = {
        "train_avg_loglik": train_avg_loglik,
        "new_avg_loglik": new_avg_loglik,
        "shift_zscore": float(shift_zscore),
        "distribution_shift_detected": bool(shift_zscore > shift_zscore_threshold),
    }
    result.attrs["gmm_shift_diagnostics"] = diagnostics
    return result


__all__ = ["gmm_bic_membership_features"]
