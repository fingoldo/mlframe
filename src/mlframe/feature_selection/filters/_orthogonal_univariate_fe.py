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
    "generate_pair_cross_basis_features",
    "score_pair_cross_basis_by_mi_uplift",
    "hybrid_orth_mi_pair_fe",
    # Layer 23: recipe-aware entry points wired into MRMR.fit auto-pipeline.
    "hybrid_orth_mi_fe_with_recipes",
    "hybrid_orth_mi_pair_fe_with_recipes",
]

_BASIS_CODE = {"hermite": "He", "legendre": "L", "chebyshev": "T", "laguerre": "LL"}


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


def _dedup_collinear_source_cols(
    X: pd.DataFrame, cols: Sequence[str], *, corr_threshold: float = 0.999,
) -> list[str]:
    """Drop near-duplicate source columns BEFORE basis enumeration.

    Layer 27 incident (2026-05-31): on 10 collinear sources (x2..x10 = x1 +
    1% jitter), the constructor emitted 10 He_2 columns and every one
    survived MRMR's redundancy gate because their CMI-residuals under
    quantile binning differed by tiny amounts above the relevance floor.
    Hybrid stage exploded the candidate set 10x and MRMR couldn't
    distinguish the duplicates.

    Fix: a cheap source-side dedup pass. Walks cols in order, computes the
    abs Pearson correlation against every column already kept; drops the
    candidate if it correlates above ``corr_threshold`` with anything in
    the kept set. ``0.999`` matches the 1% jitter test fixture while
    leaving real-world near-duplicates (corr in [0.95, 0.99]) untouched.

    Non-numeric / constant / all-NaN columns are passed through (not
    deduped, not dropped) so downstream basis evaluation handles them as
    before.
    """
    if not cols:
        return list(cols)
    kept: list[str] = []
    kept_arrays: list[np.ndarray] = []
    for c in cols:
        if c not in X.columns or not pd.api.types.is_numeric_dtype(X[c]):
            # Pass-through: not a numeric column, let downstream skip it.
            kept.append(c)
            continue
        arr = np.asarray(X[c].to_numpy(), dtype=np.float64)
        finite = np.isfinite(arr)
        if not finite.any() or arr[finite].std() <= 1e-12:
            # Constant or all-NaN: pass-through.
            kept.append(c)
            kept_arrays.append(arr)
            continue
        is_dup = False
        for prev in kept_arrays:
            prev_finite = np.isfinite(prev)
            mask = finite & prev_finite
            if mask.sum() < 8:
                continue
            a = arr[mask]
            b = prev[mask]
            if a.std() <= 1e-12 or b.std() <= 1e-12:
                continue
            # Pearson abs correlation. Numerically stable for finite-mask slices.
            corr = abs(float(np.corrcoef(a, b)[0, 1]))
            if not np.isfinite(corr):
                continue
            if corr >= corr_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
            kept_arrays.append(arr)
    return kept


def generate_univariate_basis_features(
    X: pd.DataFrame,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    basis: str = "auto",
    dedup_collinear_sources: bool = True,
    dedup_corr_threshold: float = 0.999,
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
    dedup_collinear_sources : bool, default True
        When True, drop near-duplicate source columns (Pearson |corr| >=
        ``dedup_corr_threshold`` against an already-kept source) BEFORE
        basis enumeration. Defaults ON because the alternative emits N
        copies of the same basis column for N collinear sources and
        downstream MRMR cannot distinguish them (Layer 27 incident).

    Returns
    -------
    DataFrame of new columns named ``"{col}__{basis_code}{degree}"`` (e.g.
    ``"x1__He2"``, ``"x2__T3"``).
    """
    if cols is None:
        cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if dedup_collinear_sources:
        cols = _dedup_collinear_source_cols(
            X, list(cols), corr_threshold=dedup_corr_threshold,
        )
    code = _BASIS_CODE
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
    # 2. absolute: engineered_mi >= max(
    #        min_abs_mi_frac * max(raw_baseline_mi),    # legacy floor
    #        mean(raw_baseline_mi) + 3 * std(raw_baseline_mi),  # noise-aware floor
    #    )
    # Layer 27 incident (2026-05-31): on all-noise frames every raw col has
    # MI in a tight band around the noise floor and ``0.1 * max_raw`` is
    # itself in that band -- so ANY engineered_mi clears the floor and the
    # top-K fills with FPs. The noise-aware ``mean + 3*std`` reference is
    # statistical: a column drawn from the same noise distribution will
    # exceed it only on extreme tail, knocking the false-positive rate
    # below 5% per slot. On real-signal frames the max raw_baseline_mi is
    # multiple std above the mean, so the legacy floor dominates and
    # selection is unchanged.
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = float(raw_baselines.max()) if raw_baselines.size else 0.0
    legacy_floor = float(min_abs_mi_frac) * max_raw_baseline
    # Layer 27 noise-aware floor: use MEDIAN-based stats (robust to a few
    # real signals dragging the mean up). On all-noise frames every
    # baseline_mi sits in a tight band, median + 3*MAD bounds the band
    # tightly. On signal frames the true signal is an outlier above the
    # noise band and median+3*MAD remains in the noise band, so the
    # legacy ``frac * max(raw_baseline)`` dominates and legitimate signals
    # qualify as before.
    # Bonferroni-aware sigma scale: pure sqrt(2 ln 2p) under-counts the chi-
    # square-like right tail of plug-in MI's noise distribution. Empirically
    # n=1500 binary y with 10 bins produces noise MIs that exceed Gaussian
    # tails by ~1-1.5 sigma worth of probability mass. Anchor at the larger
    # of: (a) Bonferroni for the candidate count, (b) a 5-sigma floor that
    # bounds the chi-square right tail at ~1e-7 per slot. For 40 candidates
    # this gives 5.0; for 1000 it gives ~5.8 (asymptotically Bonferroni-driven).
    n_cands = int(raw_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    if raw_baselines.size >= 4:
        med = float(np.median(raw_baselines))
        mad = float(np.median(np.abs(raw_baselines - med)))
        # 1.4826 * MAD ~= std for a normal distribution.
        noise_floor = med + sigma_thresh * 1.4826 * mad
    else:
        noise_floor = 0.0
    # Layer 27 follow-up: also compute a noise floor on the ENGINEERED MI
    # distribution. On all-noise frames the engineered cols inherit the same
    # noise structure; the top engineered_mi can be artifactually 2-4x the
    # median by pure tail sampling. Bound engineered_mi above the engineered
    # median+sigma*MAD too -- legitimate signals are statistical outliers in
    # the engineered distribution AS WELL.
    eng_mis = scores["engineered_mi"].to_numpy()
    if eng_mis.size >= 4:
        med_e = float(np.median(eng_mis))
        mad_e = float(np.median(np.abs(eng_mis - med_e)))
        eng_noise_floor = med_e + sigma_thresh * 1.4826 * mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = scores[
        (scores["uplift"] >= float(min_uplift))
        & (scores["engineered_mi"] >= abs_floor)
    ]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


# ---------------------------------------------------------------------------
# Cross-basis pair features: He_a(x_i) * He_b(x_j)
# ---------------------------------------------------------------------------
#
# Motivation: the univariate path captures y = f(x_i) -- single-feature non-
# linearities. The pair-CMA-ES path in hermite_fe.py learns FULL pair coeffs
# (c_a, c_b) jointly via Optuna / CMA-ES, which is expensive (~1000 steps).
# The cross-basis pair path is the cheap middle ground: enumerate all
# (deg_a, deg_b) cells in the bilinear product grid up to ``max_degree``,
# emit basis_a(x_i) * basis_b(x_j) as a column, then rank by MI uplift vs
# the better of the two source columns.
#
# Captures the XOR / saddle / circle family without any optimisation:
# * XOR     y = sign(x_i * x_j)         -> He_1(z_i) * He_1(z_j)   = z_i*z_j
# * Saddle  y = sign(x_i^2 - x_j^2)     -> He_2(z_i) * He_0(z_j) and
#                                          He_0(z_i) * He_2(z_j)
#                                          (the linear combination of the two
#                                          is the saddle; the stronger of
#                                          the two MI-wise enters the support)
# * Circle  y = sign(x_i^2 + x_j^2 - r) -> same two terms as saddle
#
# Selection mirrors the univariate two-gate:
#   1. uplift >= min_uplift vs max(MI(x_i; y), MI(x_j; y))
#   2. engineered_mi >= min_abs_mi_frac * max_raw_baseline


def _pair_eng_col_name(col_i: str, col_j: str, basis: str, deg_a: int, deg_b: int) -> str:
    """Stable naming: ``"{col_i}*{col_j}__He{a}_He{b}"``.

    Both legs share the same basis code (e.g. He_a * He_b). The cross-basis
    enumeration intentionally fixes one basis family per pair -- mixing
    families (He_a * T_b) blows up combinatorially without measurable signal
    gain on the standard XOR / saddle / circle targets.
    """
    code = _BASIS_CODE.get(basis, basis)
    return f"{col_i}*{col_j}__{code}{deg_a}_{code}{deg_b}"


def generate_pair_cross_basis_features(
    X: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    max_degree: int = 2,
    basis: str = "auto",
    min_degree: int = 1,
) -> pd.DataFrame:
    """For each (col_i, col_j) pair and each (deg_a, deg_b) in
    [min_degree..max_degree]^2, emit ``basis(x_i)_a * basis(x_j)_b`` as a new
    column.

    Parameters
    ----------
    X : DataFrame
        Source frame. Both legs of every pair must be numeric.
    pairs : sequence of (col_i, col_j)
        Column pairs to expand. Order matters for the name but not the value
        (multiplication is commutative); pass each unordered pair once.
    max_degree : int
        Maximum degree per leg. Default 2 covers XOR (1,1), partial saddle
        (1,2)/(2,1), and pure quadratic interaction (2,2) -- enough for the
        classic non-linear pair targets without combinatorial blowup.
    basis : {'auto', 'hermite', 'legendre', 'chebyshev', 'laguerre'}
        Routed per-column via ``basis_route_by_moments`` when ``'auto'``. The
        two legs of a pair may end up on different bases under 'auto' -- the
        name reflects each leg's chosen basis only via the suffix; we keep
        the join-token consistent (``He{a}_He{b}`` even when leg basis
        differ) so callers can group by name prefix.
    min_degree : int
        Minimum degree per leg. Default 1 -- degree 0 produces the constant
        column (= identity for the OTHER leg's transform), already covered
        by the univariate path.

    Returns
    -------
    DataFrame of new pair-cross-basis columns named via ``_pair_eng_col_name``.
    """
    if not pairs:
        return pd.DataFrame(index=X.index)
    cache: dict[tuple[str, int, str], np.ndarray] = {}
    out_cols: dict = {}
    max_d = int(max_degree)
    min_d = max(0, int(min_degree))
    for col_i, col_j in pairs:
        if col_i == col_j:
            continue
        if col_i not in X.columns or col_j not in X.columns:
            logger.warning("generate_pair_cross_basis_features: missing column %r or %r; skipping", col_i, col_j)
            continue
        if not (pd.api.types.is_numeric_dtype(X[col_i]) and pd.api.types.is_numeric_dtype(X[col_j])):
            continue
        x_i = np.asarray(X[col_i].to_numpy(), dtype=np.float64)
        x_j = np.asarray(X[col_j].to_numpy(), dtype=np.float64)
        for x in (x_i, x_j):
            finite_mask = np.isfinite(x)
            if not finite_mask.all():
                fill = float(np.nanmean(x[finite_mask])) if finite_mask.any() else 0.0
                np.copyto(x, np.where(finite_mask, x, fill))
        basis_i = basis_route_by_moments(x_i) if basis == "auto" else basis
        basis_j = basis_route_by_moments(x_j) if basis == "auto" else basis
        if basis_i not in _POLY_BASES or basis_j not in _POLY_BASES:
            logger.warning(
                "generate_pair_cross_basis_features: unknown basis %r/%r for pair (%r,%r); skipping",
                basis_i, basis_j, col_i, col_j,
            )
            continue
        for deg_a in range(min_d, max_d + 1):
            for deg_b in range(min_d, max_d + 1):
                if deg_a == 0 and deg_b == 0:
                    continue
                try:
                    key_a = (col_i, deg_a, basis_i)
                    if key_a not in cache:
                        cache[key_a] = _evaluate_basis_column(x_i, basis_i, deg_a)
                    h_a = cache[key_a]
                    key_b = (col_j, deg_b, basis_j)
                    if key_b not in cache:
                        cache[key_b] = _evaluate_basis_column(x_j, basis_j, deg_b)
                    h_b = cache[key_b]
                    name = _pair_eng_col_name(col_i, col_j, basis_i if basis_i == basis_j else basis_i, deg_a, deg_b)
                    out_cols[name] = h_a * h_b
                except Exception as exc:
                    logger.warning(
                        "generate_pair_cross_basis_features: basis=%r/%r deg=%d/%d on pair (%r,%r) raised %r; skipping",
                        basis_i, basis_j, deg_a, deg_b, col_i, col_j, exc,
                    )
                    continue
    return pd.DataFrame(out_cols, index=X.index)


def score_pair_cross_basis_by_mi_uplift(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    nbins: int = 10,
) -> pd.DataFrame:
    """Score each pair-cross-basis column by MI uplift vs the BETTER of the
    two raw source columns. Mirrors ``score_features_by_mi_uplift`` but the
    name carries a pair prefix ``"{col_i}*{col_j}__..."``.

    Returns
    -------
    DataFrame with columns
    ``[engineered_col, source_col_i, source_col_j, baseline_mi_i,
    baseline_mi_j, baseline_mi, engineered_mi, uplift]`` sorted by
    ``uplift`` descending. ``baseline_mi`` is ``max(baseline_mi_i,
    baseline_mi_j)`` -- the cross-basis term must beat the BETTER individual
    leg, not just the worse one, to count as genuine interaction signal.
    """
    y_arr = (
        np.asarray(y).astype(np.int64)
        if not np.issubdtype(np.asarray(y).dtype, np.integer)
        else np.asarray(y, dtype=np.int64)
    )
    raw_cols = list(raw_X.columns)
    raw_mi = _mi_classif_batch(raw_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    raw_mi_map = dict(zip(raw_cols, raw_mi.tolist()))
    if engineered_X.empty:
        return pd.DataFrame(columns=[
            "engineered_col", "source_col_i", "source_col_j",
            "baseline_mi_i", "baseline_mi_j", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    eng_mi = _mi_classif_batch(engineered_X.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
    rows = []
    for j, eng_name in enumerate(engineered_X.columns):
        # parse "{col_i}*{col_j}__..."
        head = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        if "*" not in head:
            # not a pair column -- skip
            continue
        col_i, col_j = head.split("*", 1)
        baseline_i = float(raw_mi_map.get(col_i, 0.0))
        baseline_j = float(raw_mi_map.get(col_j, 0.0))
        baseline = max(baseline_i, baseline_j)
        emi = float(eng_mi[j])
        uplift = emi / (baseline + 1e-12)
        rows.append({
            "engineered_col": eng_name,
            "source_col_i": col_i,
            "source_col_j": col_j,
            "baseline_mi_i": baseline_i,
            "baseline_mi_j": baseline_j,
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    return df


def hybrid_orth_mi_pair_fe(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    pair_max_degree: int = 2,
    basis: str = "auto",
    top_k: int = 5,
    top_pair_count: int = 3,
    top_pair_seed_k: int = 4,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    pair_min_uplift: float = 1.05,
    pair_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two-stage hybrid: (1) univariate orthogonal-poly FE + MI-greedy, then
    (2) cross-basis pair FE on the top-N univariate source columns, also
    MI-greedy.

    Stage 1 reuses ``hybrid_orth_mi_fe`` to pick top-N univariate winners.
    The source columns of those winners (plus any explicit raw columns the
    user wants to force into the pair pool via ``cols``) form the pair seed
    pool. Stage 2 enumerates all unordered pairs over the seed pool, calls
    ``generate_pair_cross_basis_features``, ranks via
    ``score_pair_cross_basis_by_mi_uplift``, and applies the same two-gate
    selection.

    Parameters
    ----------
    X, y, cols, degrees, basis, top_k, min_uplift, min_abs_mi_frac, nbins
        Forwarded to the univariate ``hybrid_orth_mi_fe`` stage.
    pair_max_degree : int
        Max degree per leg in the cross-basis enumeration. Default 2.
    top_pair_count : int
        How many cross-basis pair winners to append after the univariate
        winners. Default 3.
    top_pair_seed_k : int
        How many top univariate source columns to pull into the pair-seed
        pool. With N sources we enumerate ``N*(N-1)/2`` pairs. Default 4
        gives 6 pairs * (pair_max_degree^2) cross-basis cells = bounded
        cost.
    pair_min_uplift, pair_min_abs_mi_frac : float
        Two-gate selection thresholds for the pair stage. Same semantics as
        the univariate gates but compared against
        ``max(MI(x_i; y), MI(x_j; y))`` as the baseline.

    Returns
    -------
    (X_augmented, univariate_scores, cross_scores)
        X_augmented : ``X`` with univariate winners THEN cross-basis pair
            winners appended, in that order. Index preserved.
        univariate_scores : ranking DataFrame from the stage-1 univariate
            pass (same shape as ``hybrid_orth_mi_fe`` returns).
        cross_scores : ranking DataFrame from the stage-2 cross-basis pair
            pass (output of ``score_pair_cross_basis_by_mi_uplift``).
    """
    # Stage 1: univariate hybrid. Use the SAME caller-facing knobs so the
    # univariate winners on the joint frame are reproducible bit-identical
    # to a direct ``hybrid_orth_mi_fe`` call.
    X_aug_uni, uni_scores = hybrid_orth_mi_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac, nbins=nbins,
    )

    # Build the pair seed pool: top univariate winners' SOURCE columns,
    # plus a fallback to the raw column MI ranking when uplift-based winners
    # are sparse (e.g. when y has no useful univariate non-linear signal but
    # has a XOR cross-term, the seed pool would otherwise be empty).
    raw_cols_all = [c for c in (cols or X.columns) if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
    seed_sources: list[str] = []
    if not uni_scores.empty:
        # Source columns of top univariate winners by uplift, deduped, order-preserving.
        for src in uni_scores["source_col"].tolist():
            if src not in seed_sources and src in raw_cols_all:
                seed_sources.append(src)
            if len(seed_sources) >= int(top_pair_seed_k):
                break
    if len(seed_sources) < 2 and len(raw_cols_all) >= 2:
        # Fallback: rank raw columns by MI(x; y), take top N. Required for
        # pure-XOR targets where no univariate basis term uplifts (all
        # univariate MIs are near-zero for y = sign(x_i * x_j)).
        y_arr = np.asarray(y).astype(np.int64) if not np.issubdtype(np.asarray(y).dtype, np.integer) else np.asarray(y, dtype=np.int64)
        raw_X_all = X[raw_cols_all]
        raw_mi_arr = _mi_classif_batch(raw_X_all.to_numpy(dtype=np.float64), y_arr, nbins=nbins)
        order = np.argsort(-raw_mi_arr)
        fallback = [raw_cols_all[i] for i in order[: int(top_pair_seed_k)]]
        for src in fallback:
            if src not in seed_sources:
                seed_sources.append(src)
            if len(seed_sources) >= int(top_pair_seed_k):
                break

    cross_scores_empty_cols = [
        "engineered_col", "source_col_i", "source_col_j",
        "baseline_mi_i", "baseline_mi_j", "baseline_mi",
        "engineered_mi", "uplift",
    ]
    if len(seed_sources) < 2 or int(top_pair_count) <= 0:
        return X_aug_uni, uni_scores, pd.DataFrame(columns=cross_scores_empty_cols)

    pairs = [
        (seed_sources[i], seed_sources[j])
        for i in range(len(seed_sources))
        for j in range(i + 1, len(seed_sources))
    ]
    pair_eng = generate_pair_cross_basis_features(
        X, pairs, max_degree=pair_max_degree, basis=basis,
    )
    if pair_eng.empty:
        return X_aug_uni, uni_scores, pd.DataFrame(columns=cross_scores_empty_cols)

    raw_X_seed = X[seed_sources]
    cross_scores = score_pair_cross_basis_by_mi_uplift(
        raw_X_seed, pair_eng, y, nbins=nbins,
    )
    # Two-gate selection mirrors the univariate stage. The absolute floor is
    # max(raw_baseline_max, cross_engineered_mi_max) * frac. The second
    # term matters for pure-interaction targets (XOR / saddle): all
    # univariate / raw baselines are noise-floor (~0.003), but the true
    # cross-basis winner sits at 0.6 nats; without taking the cross-scores
    # max into account, ALL noise cross-terms with engineered_mi ~ 0.006
    # would clear an abs_floor of 0.0003 and pollute the output. Using
    # max(.) as the reference correctly raises the bar to 0.06 in that
    # regime so only the true XOR term qualifies.
    max_raw_baseline = float(cross_scores["baseline_mi"].max()) if not cross_scores.empty else 0.0
    if not uni_scores.empty:
        max_raw_baseline = max(max_raw_baseline, float(uni_scores["baseline_mi"].max()))
    max_cross_engineered = float(cross_scores["engineered_mi"].max()) if not cross_scores.empty else 0.0
    legacy_floor = float(pair_min_abs_mi_frac) * max(max_raw_baseline, max_cross_engineered)
    # Layer 27 (2026-05-31) noise-aware floor: see hybrid_orth_mi_fe for
    # the rationale. The pair stage is even more prone to noise pollution
    # (O(p^2) candidates vs O(p) for univariate); the noise-aware
    # mean+3*std reference protects the all-noise frame's contract.
    _baselines = cross_scores["baseline_mi"].to_numpy() if not cross_scores.empty else np.array([])
    # Bonferroni-aware sigma (see hybrid_orth_mi_fe for derivation): pair
    # candidate counts are much larger than univariate so the per-candidate
    # threshold must be tighter. Anchor at max(5.0, sqrt(2 ln 2p) + 1.5).
    n_cands = int(_baselines.size)
    sigma_thresh = max(
        5.0,
        float(np.sqrt(2.0 * np.log(max(2.0, 2.0 * n_cands))) + 1.5),
    )
    if _baselines.size >= 4:
        _med = float(np.median(_baselines))
        _mad = float(np.median(np.abs(_baselines - _med)))
        noise_floor = _med + sigma_thresh * 1.4826 * _mad
    else:
        noise_floor = 0.0
    # Also bound vs engineered MI distribution.
    _eng_mis = cross_scores["engineered_mi"].to_numpy() if not cross_scores.empty else np.array([])
    if _eng_mis.size >= 4:
        _med_e = float(np.median(_eng_mis))
        _mad_e = float(np.median(np.abs(_eng_mis - _med_e)))
        eng_noise_floor = _med_e + sigma_thresh * 1.4826 * _mad_e
    else:
        eng_noise_floor = 0.0
    abs_floor = max(legacy_floor, noise_floor, eng_noise_floor)
    qualified = cross_scores[
        (cross_scores["uplift"] >= float(pair_min_uplift))
        & (cross_scores["engineered_mi"] >= abs_floor)
    ]
    winners = qualified.head(int(top_pair_count))
    keep_pair = list(winners["engineered_col"])
    if keep_pair:
        X_aug = pd.concat([X_aug_uni, pair_eng[keep_pair]], axis=1)
    else:
        X_aug = X_aug_uni
    return X_aug, uni_scores, cross_scores


# ---------------------------------------------------------------------------
# Layer 23 (2026-05-31): recipe-emitting wrappers
# ---------------------------------------------------------------------------
#
# The vanilla ``hybrid_orth_mi_fe`` / ``hybrid_orth_mi_pair_fe`` return a
# DataFrame + scores. For MRMR.fit auto-wiring we ALSO need ``EngineeredRecipe``
# objects so that ``MRMR.transform`` can replay each appended column on test
# data deterministically (no y reference -- the recipe carries only basis +
# degree per source column). The wrappers below re-derive the per-col basis
# the same way ``generate_univariate_basis_features`` did, build one recipe
# per appended column, and return them alongside the existing outputs.


def _col_basis_for_recipe(x: np.ndarray, basis: str) -> str:
    """Resolve the per-column basis: explicit string when caller pinned one,
    else moment-routed auto. Mirrors the inline decision in
    ``generate_univariate_basis_features`` / ``generate_pair_cross_basis_features``.
    """
    if basis == "auto":
        return basis_route_by_moments(x)
    return basis


def hybrid_orth_mi_fe_with_recipes(
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
):
    """Same as :func:`hybrid_orth_mi_fe` but additionally returns a list of
    ``EngineeredRecipe`` objects -- one per appended univariate column --
    so that ``MRMR.transform`` can recompute each engineered column on
    test data without re-running the MI ranking.

    Returns
    -------
    (X_augmented, scores, recipes)
    """
    from .engineered_recipes import build_orth_univariate_recipe
    X_aug, scores = hybrid_orth_mi_fe(
        X, y, cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac, nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    recipes = []
    for name in appended:
        # Re-derive (src, degree, basis) from the appended frame: src is the
        # prefix before ``__``; basis/degree are encoded in the suffix. Cross-
        # check by also routing the source column via the same auto rule we
        # used at fit time so the recipe replays identically.
        src = name.split("__", 1)[0]
        suffix = name.split("__", 1)[1]
        # _BASIS_CODE = {"hermite":"He","legendre":"L","chebyshev":"T","laguerre":"LL"}
        # Order longest-first to avoid 'L' matching the start of 'LL'.
        code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}
        chosen_basis = None
        chosen_degree = None
        for code in ("LL", "He", "T", "L"):
            if suffix.startswith(code):
                rest = suffix[len(code):]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_fe_with_recipes: cannot parse basis/degree "
                "from column name %r; skipping recipe build.", name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes


def hybrid_orth_mi_pair_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    pair_max_degree: int = 2,
    basis: str = "auto",
    top_k: int = 5,
    top_pair_count: int = 3,
    top_pair_seed_k: int = 4,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    pair_min_uplift: float = 1.05,
    pair_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
):
    """Same as :func:`hybrid_orth_mi_pair_fe` but additionally returns a
    flat list of recipes (univariate + pair, in append order) for replay.
    """
    from .engineered_recipes import (
        build_orth_univariate_recipe,
        build_orth_pair_cross_recipe,
    )
    X_aug, uni_scores, cross_scores = hybrid_orth_mi_pair_fe(
        X, y, cols=cols, degrees=degrees, basis=basis,
        pair_max_degree=pair_max_degree,
        top_k=top_k, top_pair_count=top_pair_count,
        top_pair_seed_k=top_pair_seed_k,
        min_uplift=min_uplift, min_abs_mi_frac=min_abs_mi_frac,
        pair_min_uplift=pair_min_uplift,
        pair_min_abs_mi_frac=pair_min_abs_mi_frac,
        nbins=nbins,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]
    code_to_basis = {"He": "hermite", "LL": "laguerre", "T": "chebyshev", "L": "legendre"}
    recipes = []
    for name in appended:
        if "*" in name.split("__", 1)[0]:
            # pair cross: "{col_i}*{col_j}__{code}{deg_a}_{code}{deg_b}"
            head, suffix = name.split("__", 1)
            col_i, col_j = head.split("*", 1)
            # parse "{code_a}{deg_a}_{code_b}{deg_b}"
            try:
                left, right = suffix.split("_", 1)
            except ValueError:
                logger.warning(
                    "hybrid_orth_mi_pair_fe_with_recipes: cannot parse pair "
                    "suffix %r in %r; skipping recipe.", suffix, name,
                )
                continue
            def _parse_code_deg(s: str):
                for code in ("LL", "He", "T", "L"):
                    if s.startswith(code):
                        rest = s[len(code):]
                        if rest.isdigit():
                            return code_to_basis[code], int(rest)
                return None, None
            basis_a, deg_a = _parse_code_deg(left)
            basis_b, deg_b = _parse_code_deg(right)
            if basis_a is None or basis_b is None:
                logger.warning(
                    "hybrid_orth_mi_pair_fe_with_recipes: cannot parse code/deg "
                    "from %r; skipping recipe.", name,
                )
                continue
            # For a cross-basis pair the generator emits a single basis code
            # for both legs (basis_i if basis_i == basis_j else basis_i).
            # When ``basis='auto'`` and basis_route_by_moments disagrees
            # between legs, the name is built with basis_i's code, but the
            # ACTUAL leg-2 evaluation used basis_j. Re-route per-column at
            # recipe-build time and prefer the moment-routed basis when in
            # auto mode so replay matches fit-time evaluation.
            if basis == "auto":
                try:
                    x_i = X[col_i].to_numpy(dtype=np.float64)
                    x_j = X[col_j].to_numpy(dtype=np.float64)
                    basis_a = basis_route_by_moments(x_i)
                    basis_b = basis_route_by_moments(x_j)
                except Exception:
                    pass
            recipes.append(build_orth_pair_cross_recipe(
                name=name, src_a_name=col_i, src_b_name=col_j,
                basis_i=basis_a, basis_j=basis_b,
                deg_a=deg_a, deg_b=deg_b,
            ))
        else:
            # univariate: "{col}__{code}{degree}"
            src = name.split("__", 1)[0]
            suffix = name.split("__", 1)[1]
            chosen_basis = None
            chosen_degree = None
            for code in ("LL", "He", "T", "L"):
                if suffix.startswith(code):
                    rest = suffix[len(code):]
                    if rest.isdigit():
                        chosen_basis = code_to_basis[code]
                        chosen_degree = int(rest)
                        break
            if chosen_basis is None or chosen_degree is None:
                logger.warning(
                    "hybrid_orth_mi_pair_fe_with_recipes: cannot parse basis/"
                    "degree from %r; skipping recipe.", name,
                )
                continue
            recipes.append(build_orth_univariate_recipe(
                name=name, src_name=src,
                basis=chosen_basis, degree=chosen_degree,
            ))
    return X_aug, uni_scores, cross_scores, recipes

