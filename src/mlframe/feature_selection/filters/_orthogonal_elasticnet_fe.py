"""Layer 82 (2026-06-01): ELASTIC NET (L1 + L2) coefficient-based ranking +
MUTUAL-RANK fusion for hybrid orth-poly FE.

Why this layer
--------------

Layer 81 ships Lasso (pure L1) coefficient-based pre-selection. Lasso is the
right tool when the support is genuinely sparse and the candidate columns are
WEAKLY correlated -- in that regime the L1 path pins each truly-informative
column with a non-zero coefficient and drives the rest to exactly zero.

Lasso's known failure mode is CORRELATED CANDIDATE GROUPS: when two columns
carry near-identical predictive content, the L1 path arbitrarily picks ONE of
the pair (whichever the coordinate-descent solver visits first) and shrinks
the other to zero. The choice is solver-dependent and seed-fragile -- exactly
the kind of brittle behaviour pre-selection should avoid. Zou & Hastie (2005)
introduced Elastic Net to fix this: the L2 penalty term shares coefficient
mass among correlated columns ("grouping effect"), so a correlated PAIR is
either kept together or dropped together rather than arbitrarily split.

This module:

* ``score_features_by_elasticnet_coef`` -- the Layer 81 Lasso scorer with the
  L1-only fit swapped for sklearn ``ElasticNet(alpha, l1_ratio)``. Same I/O
  shape as Layer 81; the ``engineered_mi`` column carries ``|coef|`` from the
  Elastic Net fit instead of from Lasso.

* ``hybrid_orth_mi_elasticnet_fe`` -- generator + Elastic-Net ranking + the
  same two-gate filter Layer 81 uses (uplift + abs-coef floor).

* ``MUTUAL-RANK fusion`` strategy added to the Layer 69 ensemble path
  (``_orthogonal_scorer_auto_fe.py``): ``mutual_top_k`` keeps a candidate
  only if it is in the top-K of EVERY participating scorer. This is the
  strict-conjunction complement of the existing ``mean_rank`` / ``borda_count``
  aggregators -- high-precision selection at the cost of recall.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the SAME
kind Layers 21 / 65-81 use -- because the engineered VALUES are bit-equal;
only the SCORING changes.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_elasticnet_enable=True`` on the MRMR ctor.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._orthogonal_univariate_fe import generate_univariate_basis_features

logger = logging.getLogger(__name__)

__all__ = [
    "score_features_by_elasticnet_coef",
    "hybrid_orth_mi_elasticnet_fe",
    "hybrid_orth_mi_elasticnet_fe_with_recipes",
]


def _fit_elasticnet_abs_coefs(
    X_stack: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    l1_ratio: float,
    standardize: bool,
    random_state: int = 0,
) -> np.ndarray:
    """Fit ElasticNet(alpha, l1_ratio) on the stacked design matrix and return ``|coef|``.

    Standardisation mirrors Layer 81: constant columns are zeroed before the
    fit so they cannot inflate or NaN through the coordinate descent solver.
    NaN / inf are scrubbed to zero. l1_ratio=1.0 reproduces pure Lasso;
    l1_ratio=0.0 reproduces pure Ridge -- sklearn's ElasticNet handles both
    edges via internal dispatch.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import ElasticNet

    X_arr = np.asarray(X_stack, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n_rows, n_cols = X_arr.shape
    if n_rows == 0 or n_cols == 0:
        return np.zeros(n_cols, dtype=np.float64)
    X_clean = np.where(np.isfinite(X_arr), X_arr, 0.0)
    if standardize:
        col_mean = X_clean.mean(axis=0, keepdims=True)
        col_std = X_clean.std(axis=0, keepdims=True)
        safe_std = np.where(col_std > 1e-12, col_std, 1.0)
        X_scaled = (X_clean - col_mean) / safe_std
        const_mask = (col_std <= 1e-12).ravel()
        if const_mask.any():
            X_scaled[:, const_mask] = 0.0
    else:
        X_scaled = X_clean
    model = ElasticNet(
        alpha=float(alpha),
        l1_ratio=float(l1_ratio),
        fit_intercept=True,
        max_iter=5000,
        random_state=int(random_state),
    )
    # Surface (do NOT blanket-ignore) ConvergenceWarning: a non-converged
    # ElasticNet returns coefficients that silently drive feature selection, so
    # the caller must at least see a WARN. Other warning categories pass through
    # the normal filters unchanged.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=ConvergenceWarning)
        model.fit(X_scaled, y_arr)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        logger.warning(
            "ElasticNet did not converge in max_iter=5000 (alpha=%s, l1_ratio=%s); "
            "|coef| feature scores may be unreliable.", alpha, l1_ratio,
        )
    return np.abs(np.asarray(model.coef_, dtype=np.float64)).ravel()


def score_features_by_elasticnet_coef(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    standardize: bool = True,
    random_state: int = 0,
) -> pd.DataFrame:
    """Stack ``[raw_X, engineered_X]`` and rank by absolute Elastic-Net coefficient.

    Parameters
    ----------
    raw_X : DataFrame
        Original source columns (used to compute the per-source baseline
        ``|coef|``). Positionally aligned with ``y``.
    engineered_X : DataFrame
        Output of :func:`generate_univariate_basis_features`. Column names
        carry the ``"{source}__{basis_code}{degree}"`` suffix so the
        baseline can be looked up by source.
    y : array-like (n,)
        Target. Treated as regression target on float64; binary y is
        handled identically to Layer 81 (Lasso of indicator-coded y has
        the same |coef| ranking as the logistic-EN path at typical SNR).
    alpha : float
        Joint regularisation strength (same scale as sklearn ElasticNet).
        Higher -> sparser support.
    l1_ratio : float
        Mixing parameter in [0, 1]. 1.0 = pure Lasso, 0.0 = pure Ridge.
        Default 0.5 splits the penalty evenly; 0.3 - 0.7 is the typical
        production range when correlated candidate groups are expected.
    standardize : bool
        Whether to z-score the design matrix before the fit. Elastic Net
        is not scale-invariant.

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``engineered_mi`` (= |coef|)
    descending. ``baseline_mi`` / ``engineered_mi`` are |coef| values
    from the joint Elastic-Net fit (not MI); names reused for cross-
    layer consistency.
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_elasticnet_coef: raw_X has {len(raw_X)} "
            f"rows but engineered_X has {len(engineered_X)}; positional "
            f"row alignment required."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(
            f"score_features_by_elasticnet_coef: raw_X has {len(raw_X)} " f"rows but y has {len(np.asarray(y))}; positional row " f"alignment required."
        )
    raw_cols = list(raw_X.columns)
    eng_cols = list(engineered_X.columns)
    if not eng_cols:
        return pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    # Single Elastic-Net fit on the JOINT stack so |coef| values are
    # comparable across raw and engineered columns (mirrors Layer 81's
    # rationale: two separate fits would double-count shared signal).
    # f64 kept: covariance/coordinate-descent stability (f32 sums lose precision here) -- NOT routed through _crit_np_dtype.
    stack_arr = np.column_stack([
        raw_X.to_numpy(dtype=np.float64),
        engineered_X.to_numpy(dtype=np.float64),
    ])
    abs_coefs = _fit_elasticnet_abs_coefs(
        stack_arr, np.asarray(y).ravel(),
        alpha=float(alpha), l1_ratio=float(l1_ratio),
        standardize=bool(standardize),
        random_state=int(random_state),
    )
    raw_coef_map = dict(zip(raw_cols, abs_coefs[: len(raw_cols)].tolist()))
    eng_coefs = abs_coefs[len(raw_cols) :]
    rows = []
    for j, eng_name in enumerate(eng_cols):
        source = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        baseline = float(raw_coef_map.get(source, 0.0))
        emi = float(eng_coefs[j])
        uplift = emi / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col": source,
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["engineered_mi", "uplift"], ascending=[False, False],
        ).reset_index(drop=True)
    return df


def hybrid_orth_mi_elasticnet_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    standardize: bool = True,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elastic-Net-coefficient variant of :func:`hybrid_orth_mi_fe`.

    Same two-gate selection as Layer 81 (uplift + abs-|coef| floor); the
    only structural difference is the L2 penalty term that lets the
    coefficient mass distribute across CORRELATED candidate columns rather
    than concentrate on whichever one the solver visits first. On a
    correlated candidate pair the Elastic-Net keeps BOTH with similar
    coefficients, while Lasso keeps ONE and zeroes the other.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the EN-ranked top-K winners appended.
        scores : the full EN-coef ranking DataFrame (winners + rejects).
    """
    engineered = generate_univariate_basis_features(
        X, cols=cols, degrees=degrees, basis=basis,
    )
    if engineered.empty:
        return X, pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    raw_X = X[[
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]]
    scores = score_features_by_elasticnet_coef(
        raw_X, engineered, y,
        alpha=float(alpha), l1_ratio=float(l1_ratio),
        standardize=bool(standardize),
        random_state=int(random_state),
    )
    if scores.empty:
        return X, scores
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max(0.0, max_raw_baseline)
    qualified = scores[(scores["engineered_mi"] > 0.0) & (scores["engineered_mi"] >= legacy_floor) & (scores["uplift"] >= float(min_uplift))]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_elasticnet_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    alpha: float = 0.01,
    l1_ratio: float = 0.5,
    standardize: bool = True,
    random_state: int = 0,
):
    """Same as :func:`hybrid_orth_mi_elasticnet_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so that
    ``MRMR.transform`` can recompute each engineered column on test data
    without re-running the Elastic-Net fit.

    Recipes are byte-identical to Layer 21 because the engineered VALUES
    are byte-identical -- only the SCORING (and therefore the selection)
    differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe
    from ._orthogonal_univariate_fe import _evaluate_basis_column

    X_aug, scores = hybrid_orth_mi_elasticnet_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        alpha=float(alpha), l1_ratio=float(l1_ratio),
        standardize=bool(standardize),
        random_state=int(random_state),
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    code_to_basis = {
        "He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre",
    }
    for name in appended:
        if "__" not in name:
            continue
        src, suffix = name.split("__", 1)
        chosen_basis = None
        chosen_degree = None
        for code in ("LL", "He", "T", "L"):
            if suffix.startswith(code):
                rest = suffix[len(code) :]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_elasticnet_fe_with_recipes: cannot parse " "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        # mrmr_audit_2026-07-20 B-17: freeze the fit-time basis-preprocess params (mirrors the
        # canonical Layer-21 hybrid_orth_mi_fe_with_recipes fix); recomputing on the FULL fit-time
        # source column is safe/exact -- it reproduces, not refits, the fit-time params.
        _pp = None
        try:
            _col_full = np.asarray(X[src].to_numpy(), dtype=np.float64)
            _, _pp = _evaluate_basis_column(_col_full, chosen_basis, int(chosen_degree), return_params=True)
        except Exception:
            _pp = None
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
            preprocess_params=_pp,
        ))
    return X_aug, scores, recipes
