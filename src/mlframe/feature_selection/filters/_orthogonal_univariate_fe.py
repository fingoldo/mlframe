"""Univariate orthogonal-polynomial FE + MI-greedy selector for MRMR (2026-05-31).

Three pieces:

1. ``generate_univariate_basis_features`` -- for each source column, fit the
   per-basis preprocess (z-score for Hermite, min-max for Legendre/Chebyshev,
   non-negative shift for Laguerre), then emit ``He_n(z)`` / ``L_n(z)`` /
   ``T_n(z)`` / ``L^Lag_n(z)`` for n in ``degrees`` as new columns. Basis is
   auto-routed per column via ``basis_route_by_moments`` when ``basis='auto'``.

2. ``score_features_by_mi_uplift`` -- batch-score each emitted column against
   y via the existing ``_plugin_mi_classif_batch_njit`` path (or sklearn KSG
   for regression-mode y). Returns ranked DataFrame with raw-column baseline,
   emitted MI, and ``uplift = MI / baseline_MI``.

3. ``hybrid_orth_mi_fe`` -- pipeline: (a) generate univariate basis features
   for the user-selected source columns, (b) rank by MI uplift, (c) emit the
   top-K winners. Optionally appends user-requested pairwise outer products
   ``He_a(x_i) * He_b(x_j)`` for the strongest single-column winners.

Why this lives outside of polynom_pair_fe:

* polynom_pair_fe is a PAIR optimisation (learns coef_a, coef_b together via
  CMA-ES on a 2-arg bin_func), excellent for discovering interaction signal
  but expensive (~1000 optimisation steps per pair) and gated by
  ``fe_smart_polynom_iters > 0``. The univariate path is O(p * max_degree)
  evaluations + one MI ranking pass -- 100-1000x cheaper -- and complements
  the pair optimiser for single-feature non-linearities (y = sign(He_2(x_i)))
  that the pair path never explores.

* The hybrid is the user-requested combination: orthogonal-polynomial basis
  expansion FIRST (cheap, covers most low-degree non-linearities), MI-greedy
  ranking SECOND (filters to the actually-useful ones). Result feeds straight
  back into MRMR's standard relevance/redundancy gates as ordinary numeric
  columns.

NOT wired into MRMR.fit by default -- explicit opt-in via direct call. The
existing fe_smart_polynom_iters / fe_max_polynoms knobs cover the auto-wired
path. Users who want univariate orthogonal expansion call
``hybrid_orth_mi_fe`` themselves and pass the augmented DataFrame to fit.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .hermite_fe import _POLY_BASES, basis_route_by_moments, polyeval_dispatch

logger = logging.getLogger(__name__)

__all__ = [
    "generate_univariate_basis_features",
    "score_features_by_mi_uplift",
    "hybrid_orth_mi_fe",
]


def _evaluate_basis_column(x: np.ndarray, basis: str, degree: int) -> np.ndarray:
    """Preprocess x to the basis domain, then evaluate the single basis function
    of given degree via a one-hot coefficient vector. Returns shape (n,).

    The preprocess ``fit`` functions return a (z, params) tuple where z is the
    domain-mapped values - reuse z directly rather than calling apply with the
    untyped params dict (which can vary per basis: zscore -> mean/std; minmax
    -> lo/hi; shift -> lo).
    """
    basis_info = _POLY_BASES[basis]
    fit_fn = basis_info["fit"]
    z, _params = fit_fn(x)
    z = np.ascontiguousarray(z, dtype=np.float64)
    # One-hot coefficient vector: He_n / L_n / T_n / L^Lag_n at the chosen degree.
    coef = np.zeros(degree + 1, dtype=np.float64)
    coef[degree] = 1.0
    return polyeval_dispatch(basis, z, coef)


def generate_univariate_basis_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
) -> pd.DataFrame:
    """For each column in cols, emit ``basis_n(x)`` columns for n in degrees.

    Parameters
    ----------
    X : DataFrame
        Source frame. Only numeric columns are processed; non-numeric are
        silently skipped.
    cols : sequence of column names, optional
        Columns to expand. None = all numeric columns.
    degrees : sequence of int
        Polynomial degrees to emit. degree=1 is the identity-after-preprocess
        and rarely uplifts MI, so the default starts at 2.
    basis : {'auto', 'hermite', 'legendre', 'chebyshev', 'laguerre'}
        'auto' routes per column via the moment fingerprint at
        ``basis_route_by_moments`` (skew>1.5 + one-sided -> laguerre; near-
        Gaussian -> hermite; bounded -> chebyshev; else chebyshev).

    Returns
    -------
    DataFrame of new columns named ``"{col}__{basis_code}{degree}"`` (e.g.
    ``"x1__He2"``, ``"x2__T3"``).
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    code = {"hermite": "He", "legendre": "L", "chebyshev": "T", "laguerre": "LL"}
    out_cols: dict = {}
    for col in cols:
        x = np.asarray(X[col].to_numpy(), dtype=np.float64)
        finite_mask = np.isfinite(x)
        if not finite_mask.all():
            x = np.where(finite_mask, x, np.nanmean(x[finite_mask]) if finite_mask.any() else 0.0)
        chosen_basis = basis_route_by_moments(x) if basis == "auto" else basis
        if chosen_basis not in _POLY_BASES:
            logger.warning("generate_univariate_basis_features: unknown basis %r for col %r; skipping", chosen_basis, col)
            continue
        for d in degrees:
            try:
                vals = _evaluate_basis_column(x, chosen_basis, int(d))
                out_cols[f"{col}__{code.get(chosen_basis, chosen_basis)}{d}"] = vals
            except Exception as exc:
                logger.warning("generate_univariate_basis_features: basis=%r degree=%d on col=%r raised %r; skipping",
                               chosen_basis, d, col, exc)
                continue
    return pd.DataFrame(out_cols, index=X.index)


def _mi_classif_batch(X: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> np.ndarray:
    """Batch MI(X_j; y) for classification target. Uses quantile-binning of
    each column to nbins, then sklearn's mutual_info_score on the joint
    histogram. Returns shape (p,) of MI values in nats."""
    from sklearn.metrics import mutual_info_score
    n, p = X.shape
    mis = np.zeros(p, dtype=np.float64)
    for j in range(p):
        col = X[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            mis[j] = 0.0
            continue
        col_f = col[finite]
        try:
            edges = np.quantile(col_f, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
            edges = np.unique(edges)
            if edges.size == 0:
                mis[j] = 0.0
                continue
            binned = np.searchsorted(edges, col_f)
            mis[j] = float(mutual_info_score(binned, y[finite]))
        except Exception:
            mis[j] = 0.0
    return mis


def score_features_by_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    nbins: int = 10,
) -> pd.DataFrame:
    """Score each engineered column by MI uplift vs its raw source column.

    Parameters
    ----------
    raw_X : DataFrame
        Original source columns.
    engineered_X : DataFrame
        Output of ``generate_univariate_basis_features``. Column names must
        carry the ``"{source}__{basis_code}{degree}"`` suffix so the source
        baseline can be looked up.
    y : array-like (n,)
        Target. Must be discrete (binary or multiclass int codes); for
        continuous y, bin via ``pd.qcut`` first.
    nbins : int
        Quantile bins for column binning before MI computation.

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``uplift`` descending.
    """
    y_arr = np.asarray(y).astype(np.int64) if not np.issubdtype(np.asarray(y).dtype, np.integer) else np.asarray(y, dtype=np.int64)
    raw_cols = list(raw_X.columns)
    raw_mi = _mi_classif_batch(raw_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    eng_mi = _mi_classif_batch(engineered_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        source = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        baseline = float(raw_mi_map.get(source, 0.0))
        emi = float(eng_mi[j])
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
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    return df


def hybrid_orth_mi_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    top_k: int = 5,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hybrid pipeline: univariate orthogonal-polynomial expansion + MI-greedy
    selection.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the selected top-K MI-uplifted basis columns
            appended. Index preserved.
        scores : the full ranking DataFrame (winners + rejects), useful for
            debugging which transforms uplifted vs which didn't.

    The selection rule is ``uplift >= min_uplift`` then top-K by uplift. A
    basis column with engineered_MI < its source baseline never enters the
    output even if it makes the top-K -- the uplift gate dominates.

    Example
    -------
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> x1 = rng.standard_normal(n)
    >>> x2 = rng.uniform(-1, 1, n)
    >>> X = pd.DataFrame({"x1": x1, "x2": x2})
    >>> y = (x1 ** 2 + x2 ** 3 > 1.0).astype(int)  # He_2(x1) + L_3(x2) signal
    >>> X_aug, scores = hybrid_orth_mi_fe(X, y, degrees=(2, 3))
    >>> # X_aug now has x1__He2 and x2__L3 appended (assuming uplift > 1.05)
    """
    engineered = generate_univariate_basis_features(X, cols=cols, degrees=degrees, basis=basis)
    if engineered.empty:
        return X.copy(), pd.DataFrame(columns=["engineered_col", "source_col", "baseline_mi", "engineered_mi", "uplift"])
    raw_X = X[[c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]]
    scores = score_features_by_mi_uplift(raw_X, engineered, y, nbins=nbins)
    # Two-gate selection:
    # 1. relative: uplift >= min_uplift (default 1.05 = require 5% MI gain vs raw source)
    # 2. absolute: engineered_mi >= min_abs_mi_frac * max(raw_baseline_mi) -- prevents noise
    #    columns from sneaking in via tiny-baseline ratio inflation
    #    (e.g. noise raw MI 0.003 * noise He3 MI 0.004 = 1.4x uplift but absolute MI is
    #    still noise floor; absolute gate rejects it).
    max_raw_baseline = float(scores["baseline_mi"].max()) if not scores.empty else 0.0
    abs_floor = float(min_abs_mi_frac) * max_raw_baseline
    qualified = scores[
        (scores["uplift"] >= float(min_uplift))
        & (scores["engineered_mi"] >= abs_floor)
    ]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores
