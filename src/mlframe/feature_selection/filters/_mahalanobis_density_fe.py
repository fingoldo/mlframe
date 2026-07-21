"""Multivariate Mahalanobis / Gaussian-copula joint density anomaly score (mrmr_audit_2026-07-20
fe_expansion.md "Multivariate Mahalanobis / Gaussian-copula joint density anomaly score").

Computes a single new feature ``d(row) = sqrt((x-mu)^T Sigma^-1 (x-mu))`` over a correlated
cluster (or all) of numeric raw columns jointly -- the classical multivariate-normal quadratic
form, with the mean/covariance Ledoit-Wolf shrunk (reused from ``sklearn.covariance.LedoitWolf``,
not reimplemented) to avoid p-close-to-n ill-conditioning.

Why this catches a shape the catalog misses: y can depend on whether a row sits inside or outside
an ELLIPSOIDAL level-set of the joint distribution of p=15-30 correlated numeric columns (e.g. a
multivariate process-control / fraud "jointly-typical vs jointly-atypical" target) where NO single
column, pair, triplet, or even a quadruplet arity-4 cross-basis is individually extreme -- each
column can sit comfortably within its own marginal range while the JOINT combination is far in
Mahalanobis distance (the classic multivariate-outlier-invisible-to-univariate-checks scenario).
The existing group_distance / conditional-dispersion families condition one column's deviation on
ONE other (binned) column; this is the p-way generalization using the FULL covariance structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

__all__ = [
    "mahalanobis_density_feature",
    "engineered_name_mahalanobis_density",
    "generate_mahalanobis_density_features",
    "apply_mahalanobis_density",
    "build_mahalanobis_density_recipe",
    "hybrid_mahalanobis_density_fe",
]


def mahalanobis_density_feature(
    X: np.ndarray,
    *,
    X_fit: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mahalanobis distance of every row of ``X`` to a Ledoit-Wolf shrunk mean/covariance.

    Parameters
    ----------
    X : (n, p) array
        Rows to score.
    X_fit : (n_fit, p) array, optional
        The rows to fit ``mu``/``Sigma`` on. ``None`` (default) fits on ``X`` itself. Pass the
        TRAIN rows explicitly and ``X`` as the full (train+test) set for a leak-safe fit-once/
        apply-to-all-rows contract, mirroring the existing K-fold-fit-then-apply discipline used
        elsewhere in this codebase.

    Returns
    -------
    (n,) float64 array of Mahalanobis distances (``>= 0``). Degenerate input (n_fit < p+1, p < 1,
    non-finite X or X_fit) returns an all-NaN ``(n,)`` array rather than raising -- Ledoit-Wolf
    itself needs at least a modest sample-to-dimension ratio to shrink meaningfully.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    fit_arr = np.asarray(X_fit, dtype=np.float64) if X_fit is not None else X
    if fit_arr.ndim == 1:
        fit_arr = fit_arr[:, None]
    n_fit = fit_arr.shape[0]

    if p < 1 or n_fit < p + 1:
        return np.full(n, np.nan, dtype=np.float64)
    if not (np.isfinite(X).all() and np.isfinite(fit_arr).all()):
        return np.full(n, np.nan, dtype=np.float64)

    lw = LedoitWolf().fit(fit_arr)
    mu = lw.location_
    Sigma_inv = lw.get_precision()

    delta = X - mu
    d2 = np.einsum("ni,ij,nj->n", delta, Sigma_inv, delta)
    return np.asarray(np.sqrt(np.maximum(d2, 0.0)))


def engineered_name_mahalanobis_density(cols: Sequence[str]) -> str:
    """Deterministic engineered-column name for the joint Mahalanobis-density score."""
    return "mahal__" + "_".join(str(c) for c in cols)


def generate_mahalanobis_density_features(X: "pd.DataFrame", cols: Sequence[str]) -> "tuple[pd.DataFrame, dict]":
    """Fit one joint Mahalanobis-density score over ``cols``: Ledoit-Wolf shrink the mean/precision
    on the fit rows, score every fit row, and freeze ``mu``/``Sigma_inv`` for leak-safe replay
    (both are pure functions of X, no ``y`` reference). Returns ``(enc_df, payload)``."""
    cols = [c for c in cols if c in X.columns]
    if len(cols) < 1:
        return pd.DataFrame(index=X.index), {}
    X_cols = X[cols].to_numpy(dtype=np.float64)
    n, p = X_cols.shape
    if p < 1 or n < p + 1 or not np.isfinite(X_cols).all():
        return pd.DataFrame(index=X.index), {}

    lw = LedoitWolf().fit(X_cols)
    mu = lw.location_
    Sigma_inv = lw.get_precision()
    delta = X_cols - mu
    d2 = np.einsum("ni,ij,nj->n", delta, Sigma_inv, delta)
    scores = np.sqrt(np.maximum(d2, 0.0))

    name = engineered_name_mahalanobis_density(cols)
    enc = pd.DataFrame({name: scores}, index=X.index)
    payload = {"cols": tuple(cols), "mu": np.asarray(mu, dtype=np.float64).copy(), "Sigma_inv": np.asarray(Sigma_inv, dtype=np.float64).copy()}
    return enc, payload


def apply_mahalanobis_density(X_test: "pd.DataFrame", payload: dict) -> "pd.DataFrame":
    """Replay a fitted Mahalanobis-density score on new rows: closed-form quadratic form against
    the frozen ``mu``/``Sigma_inv`` -- reads only X, no ``y``."""
    cols = list(payload["cols"])
    missing = [c for c in cols if c not in X_test.columns]
    if missing:
        raise KeyError(f"apply_mahalanobis_density: missing column(s) {missing} from X_test")
    X_cols = X_test[cols].to_numpy(dtype=np.float64)
    delta = X_cols - payload["mu"]
    d2 = np.einsum("ni,ij,nj->n", delta, payload["Sigma_inv"], delta)
    scores = np.sqrt(np.maximum(d2, 0.0))
    name = engineered_name_mahalanobis_density(cols)
    return pd.DataFrame({name: scores}, index=X_test.index)


def build_mahalanobis_density_recipe(*, name: str, cols: Sequence[str], mu: np.ndarray, Sigma_inv: np.ndarray) -> "EngineeredRecipe":
    """Frozen recipe for the Mahalanobis-density score column. Stores the frozen ``mu``/
    ``Sigma_inv``; replay reads only X, so transform() is leakage-free."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="mahalanobis_density",
        src_names=tuple(str(c) for c in cols),
        extra={"cols": tuple(str(c) for c in cols), "mu": np.asarray(mu, dtype=np.float64).copy(), "Sigma_inv": np.asarray(Sigma_inv, dtype=np.float64).copy()},
    )


def _apply_mahalanobis_density_recipe(recipe, X) -> np.ndarray:
    """Adapter consumed by ``engineered_recipes.apply_recipe``."""
    cols = list(recipe.extra["cols"])
    if isinstance(X, pd.DataFrame):
        X_cols = X[cols].to_numpy(dtype=np.float64)
    else:
        try:
            import polars as _pl

            if isinstance(X, _pl.DataFrame):
                X_cols = X.select(cols).to_numpy().astype(np.float64)
            else:
                X_cols = np.column_stack([np.asarray(X[c], dtype=np.float64) for c in cols])
        except ImportError:
            X_cols = np.column_stack([np.asarray(X[c], dtype=np.float64) for c in cols])
    delta = X_cols - recipe.extra["mu"]
    d2 = np.einsum("ni,ij,nj->n", delta, recipe.extra["Sigma_inv"], delta)
    return np.asarray(np.sqrt(np.maximum(d2, 0.0)))


def hybrid_mahalanobis_density_fe(
    X: "pd.DataFrame",
    y: np.ndarray,
    *,
    num_cols: Optional[Sequence[str]] = None,
    max_cols_for_block: int = 20,
    top_k: int = 1,
    mi_gate: bool = True,
    mi_gate_top_k: Optional[int] = None,
    random_state: int = 0,
    reject_sink: Optional[Callable[..., None]] = None,
) -> "tuple[pd.DataFrame, list[str], list[EngineeredRecipe], pd.DataFrame]":
    """End-to-end joint Mahalanobis-density feature: bound the column pool by top raw-MI (the
    family's own point is a p=15-30-way joint ellipsoidal level-set, so the pool is capped, not
    combinatorially enumerated), compute the single Ledoit-Wolf-shrunk Mahalanobis score, MI-gate
    it against the raw baseline.

    Returns ``(X_aug, appended, recipes, enc_df)``. ``y`` is consumed only by the column-pool
    ranking + the MI gate; recipes carry the frozen ``mu``/``Sigma_inv``, never ``y``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"hybrid_mahalanobis_density_fe: X must be a pandas DataFrame; got {type(X).__name__}")
    if num_cols is None or len(num_cols) == 0:
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    else:
        num_cols = [c for c in num_cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    if len(num_cols) < 1:
        return X.copy(), [], [], pd.DataFrame()

    if y is not None:
        from ._extra_fe_families import _top_mi_num_cols  # lazy: break parent<->sibling cycle

        num_cols = _top_mi_num_cols(X, num_cols, y, max_cols_for_block)
    else:
        num_cols = list(num_cols)[: int(max_cols_for_block)]
    if len(num_cols) < 1:
        return X.copy(), [], [], pd.DataFrame()

    enc_df, payload = generate_mahalanobis_density_features(X, num_cols)
    if enc_df.empty or not payload:
        return X.copy(), [], [], pd.DataFrame()

    winners = list(enc_df.columns)
    if mi_gate and y is not None:
        from ._unified_fe_gate import local_mi_gate

        _gate_top_k = int(mi_gate_top_k) if mi_gate_top_k else int(top_k)
        winners = local_mi_gate(enc_df, y, raw_X=X, top_k=_gate_top_k, reject_sink=reject_sink)
    else:
        winners = winners[: int(top_k)]
    if not winners:
        return X.copy(), [], [], pd.DataFrame()

    X_aug = pd.concat([X, enc_df[winners]], axis=1)
    recipes = [build_mahalanobis_density_recipe(name=name, cols=payload["cols"], mu=payload["mu"], Sigma_inv=payload["Sigma_inv"]) for name in winners]
    return X_aug, winners, recipes, enc_df
