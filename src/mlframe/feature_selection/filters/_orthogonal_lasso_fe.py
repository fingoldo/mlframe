"""Layer 81 (2026-06-01): LASSO (L1) coefficient-based ranking for hybrid
orth-poly FE.

Why this layer
--------------

Layers 21 / 65-74 score candidate engineered columns via mutual-information
or dependence metrics (plug-in MI, KSG k-NN MI, copula MI, dCor, HSIC,
JMIM, TC, CMIM). All of those are NON-PARAMETRIC: they make no assumption
about the functional form linking candidate to ``y``; they reward any
statistical dependence.

L1-regularised linear regression (Lasso, Tibshirani 1996) is the dual
PARAMETRIC approach. We stack the raw and engineered columns into one
design matrix, fit a single Lasso, and rank features by ``|coef|``.
Engineered columns whose linear contribution to ``y`` survives the L1
shrinkage are SELECTED; columns the L1 path drives to exactly zero are
REJECTED.

What Lasso wins / loses against MI
----------------------------------

* WINS on linear-additive signals. When ``y = a * He_2(x_1) + b * He_3(x_2)``
  (orth-poly engineered columns enter the target ADDITIVELY and linearly),
  Lasso recovers ``(a, b)`` directly and ranks the two columns by their
  contribution magnitude. MI estimators on the same fixture rank them
  correctly too, but Lasso gives a sharper margin against noise columns
  because L1 explicitly drives noise coefficients to zero.

* LOSES on non-monotone, non-linear targets that have ZERO Pearson
  correlation but non-zero MI: ``y = cos(x)`` is the canonical example.
  ``|cov(cos(x), x)|`` is small (cosine is even-symmetric on standard
  Gaussian x), so Lasso drives the linear coefficient to zero and the
  column drops out. The plug-in MI on the same column posts a large
  value because mutual information is invariant under monotone
  transformations of either variable AND captures the |cos(x)| pattern
  through binning. This is the documented LIMITATION of Lasso pre-
  selection -- the COST of the parametric assumption.

The two pre-selection paths are COMPLEMENTARY -- a user can opt into
both and union the winners. Production tilts toward Lasso when the
ground-truth signal is known to be linear-additive (price / quantity
forecasts, additive utility models), MI when the signal is unknown
or known to be non-monotone (anomaly detection, oscillatory targets).

Why this is "Lasso PRE-SELECTION", not "Lasso feature engineering"
------------------------------------------------------------------

The L1 path here REPLACES the MI ranking inside the hybrid orth-poly
pipeline. The orth-poly basis generator (``He_n``, ``L_n``, ``T_n``,
``L^Lag_n``) still emits exactly the same engineered VALUES Layer 21
ships -- only the SCORING (and therefore the selection) changes. So
recipe replay uses the existing ``orth_univariate`` kind; no new replay
infrastructure is needed.

Standardisation
---------------

Lasso is NOT scale-invariant: a column with std=100 and a column with
std=0.01 carrying identical predictive content will receive coefficients
that differ by 4 orders of magnitude. We standardise (``StandardScaler``)
the stacked design matrix BEFORE the Lasso fit by default so the
``|coef|`` ranking is comparable across columns. ``standardize=False``
preserves the raw scale for callers who pre-standardised upstream.

Recipe replay
-------------

Each emitted column is backed by an ``orth_univariate`` recipe -- the
SAME kind Layers 21 / 65-74 use -- because the engineered VALUES are
bit-equal; only the SCORING differs.

NOT wired into ``MRMR.fit`` by default -- opt-in via
``fe_hybrid_orth_lasso_enable=True`` on the MRMR ctor.
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ._orthogonal_univariate_fe import generate_univariate_basis_features

logger = logging.getLogger(__name__)

# Below this baseline |coef| a source is treated as no-signal: the uplift ratio
# is suppressed (it would otherwise explode) and an absolute |coef| floor is
# required instead -- the same guard JMIM applies (Layer 21/65+).
_BASELINE_EPS = 1e-6
_ABS_MI_FLOOR = 1e-3

__all__ = [
    "score_features_by_lasso_coef",
    "hybrid_orth_mi_lasso_fe",
    "hybrid_orth_mi_lasso_fe_with_recipes",
]


def _fit_lasso_abs_coefs(
    X_stack: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    standardize: bool,
    random_state: int = 0,
) -> np.ndarray:
    """Fit Lasso(alpha) on the stacked design matrix and return ``|coef|``.

    Returns shape ``(n_features,)``. Constant columns receive coefficient
    zero by construction (the StandardScaler would divide by zero -- we
    short-circuit by replacing constant columns with zeros before scaling).
    Handles binary y by treating it as a regression target on
    ``{0.0, 1.0}``: Lasso of indicator-coded y against the design matrix
    has the same |coef| ranking as the canonical logistic-Lasso path at
    the typical sample sizes / signal strengths Layer 21's downstream
    gates work with, AND is dramatically cheaper.
    Multiclass y (>2 discrete classes) is one-hot/binarised and a Lasso is
    fit per class indicator; the per-feature score is the MAX ``|coef|``
    across the one-vs-rest fits. This is invariant to class-label
    ordinality, unlike regressing on the raw ordinal class integers (which
    would let a meaningless label ordering drive selection).
    """
    from sklearn.linear_model import Lasso

    X_arr = np.asarray(X_stack, dtype=np.float64)
    y_raw = np.asarray(y).ravel()
    y_arr = y_raw.astype(np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    n_rows, n_cols = X_arr.shape
    if n_rows == 0 or n_cols == 0:
        return np.zeros(n_cols, dtype=np.float64)

    # Multiclass detection: a small set of discrete (integer-valued) labels with >2 classes is a classification target whose label INTEGERS carry no
    # ordinal meaning. Regressing Lasso on those integers yields coefficients driven by the arbitrary class numbering. One-hot each class and take the
    # max |coef| over the one-vs-rest fits so selection is invariant to relabelling.
    uniq = np.unique(y_raw[np.isfinite(y_arr)]) if np.issubdtype(y_arr.dtype, np.floating) else np.unique(y_raw)
    is_discrete = uniq.size <= max(20, int(0.05 * n_rows)) and np.all(y_arr[np.isfinite(y_arr)] == np.round(y_arr[np.isfinite(y_arr)]))
    if is_discrete and uniq.size > 2:
        classes = np.unique(y_raw)
        best = np.zeros(n_cols, dtype=np.float64)
        for cls in classes:
            indicator = (y_raw == cls).astype(np.float64)
            coefs = _fit_lasso_abs_coefs(
                X_arr, indicator, alpha=alpha, standardize=standardize, random_state=random_state,
            )
            best = np.maximum(best, coefs)
        return best
    # NaN / inf scrub: Lasso's coordinate-descent does not tolerate NaN.
    X_clean = np.where(np.isfinite(X_arr), X_arr, 0.0)
    if standardize:
        col_mean = X_clean.mean(axis=0, keepdims=True)
        col_std = X_clean.std(axis=0, keepdims=True)
        # Constant columns -> std == 0; leave as zero contribution after
        # centring so the Lasso fit cannot inflate / NaN through them.
        safe_std = np.where(col_std > 1e-12, col_std, 1.0)
        X_scaled = (X_clean - col_mean) / safe_std
        # Force constant columns to literal zero post-scaling so their
        # |coef| is exactly zero regardless of solver path.
        const_mask = (col_std <= 1e-12).ravel()
        if const_mask.any():
            X_scaled[:, const_mask] = 0.0
    else:
        X_scaled = X_clean
    with warnings.catch_warnings():
        # ConvergenceWarning at small n with many candidates is expected
        # and not actionable from the caller's side -- silence at fit-time.
        warnings.simplefilter("ignore")
        model = Lasso(
            alpha=float(alpha),
            fit_intercept=True,
            max_iter=5000,
            random_state=int(random_state),
        )
        model.fit(X_scaled, y_arr)
    return np.abs(np.asarray(model.coef_, dtype=np.float64)).ravel()


def score_features_by_lasso_coef(
    raw_X: pd.DataFrame,
    engineered_X: pd.DataFrame,
    y: np.ndarray,
    *,
    alpha: float = 0.01,
    standardize: bool = True,
    random_state: int = 0,
) -> pd.DataFrame:
    """Stack ``[raw_X, engineered_X]`` and rank by absolute Lasso coefficient.

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
        Target. Continuous, binary {0, 1}, or multiclass. Multiclass
        (>2 discrete classes) is binarised one-vs-rest and the per-feature
        score is the MAX ``|coef|`` across the per-class Lasso fits, so the
        ranking is invariant to the arbitrary class-label ordering (binary
        remains the dominant production path).
    alpha : float
        Lasso L1 strength. Higher -> sparser support, more candidates
        driven to zero. Default 0.01 matches the orth-poly fixture SNR.
    standardize : bool
        Whether to z-score the design matrix before the fit. Lasso is
        not scale-invariant; standardising is the canonical recipe for
        ranking by ``|coef|``.

    Returns
    -------
    DataFrame with columns ``[engineered_col, source_col, baseline_mi,
    engineered_mi, uplift]`` sorted by ``engineered_mi`` (= |coef|)
    descending. Column names use ``baseline_mi`` / ``engineered_mi`` for
    cross-layer consistency with Layers 21 / 65 / 66 / 67; the VALUES
    are absolute Lasso coefficients (not MI), and the ranking semantics
    are identical (higher = better).
    """
    if len(raw_X) != len(engineered_X):
        raise ValueError(
            f"score_features_by_lasso_coef: raw_X has {len(raw_X)} rows "
            f"but engineered_X has {len(engineered_X)}; positional row "
            f"alignment required."
        )
    if len(raw_X) != len(np.asarray(y)):
        raise ValueError(
            f"score_features_by_lasso_coef: raw_X has {len(raw_X)} rows "
            f"but y has {len(np.asarray(y))}; positional row alignment "
            f"required."
        )
    raw_cols = list(raw_X.columns)
    eng_cols = list(engineered_X.columns)
    if not eng_cols:
        return pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    # Single Lasso on the JOINT stack so the |coef| values are comparable
    # across raw and engineered columns. (Two separate Lasso fits would
    # double-count the shared linear signal and bias the uplift ratio.)
    stack_cols = raw_cols + eng_cols
    # f64 kept: covariance/coordinate-descent stability (f32 sums lose precision here) -- NOT routed through _crit_np_dtype.
    stack_arr = np.column_stack([
        raw_X.to_numpy(dtype=np.float64),
        engineered_X.to_numpy(dtype=np.float64),
    ])
    abs_coefs = _fit_lasso_abs_coefs(
        stack_arr, np.asarray(y).ravel(),
        alpha=float(alpha), standardize=bool(standardize),
        random_state=int(random_state),
    )
    raw_coef_map = dict(zip(raw_cols, abs_coefs[: len(raw_cols)].tolist()))
    eng_coefs = abs_coefs[len(raw_cols):]
    rows = []
    for j, eng_name in enumerate(eng_cols):
        source = eng_name.split("__", 1)[0] if "__" in eng_name else eng_name
        baseline = float(raw_coef_map.get(source, 0.0))
        emi = float(eng_coefs[j])
        # Near-zero baseline |coef| makes the uplift ratio explode past the
        # gate even on a no-signal source; suppress the ratio there and let the
        # absolute |coef| floor decide (mirrors the JMIM guard).
        if baseline < _BASELINE_EPS:
            uplift = 0.0 if emi < _ABS_MI_FLOOR else float("inf")
        else:
            uplift = emi / baseline
        rows.append({
            "engineered_col": eng_name,
            "source_col": source,
            "baseline_mi": baseline,
            "engineered_mi": emi,
            "uplift": uplift,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # Primary ranking by raw |coef| magnitude: a column the L1 path
        # admitted with high weight is more useful than a column with
        # high RATIO of low absolute weights. This differs from the MI
        # variants (which rank by uplift) because uplift is symmetric in
        # the denominator while |coef| is a direct quality score.
        df = df.sort_values(
            ["engineered_mi", "uplift"], ascending=[False, False],
        ).reset_index(drop=True)
    return df


def hybrid_orth_mi_lasso_fe(
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
    standardize: bool = True,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lasso-coefficient variant of :func:`hybrid_orth_mi_fe`.

    Replaces the plug-in quantile-binned MI estimator with absolute
    L1-regularised linear-regression coefficients and applies a two-gate
    selection:

    1. relative: ``|coef_eng| / |coef_source_raw| >= min_uplift`` -- the
       engineered column must improve on the raw source's own linear
       contribution by at least ``min_uplift``.
    2. absolute: ``|coef_eng| > 0`` AND
       ``|coef_eng| >= min_abs_mi_frac * max(raw|coef|)`` -- the column
       must survive L1 shrinkage AND clear a fraction of the strongest
       raw-source coefficient. The ``> 0`` floor is the L1 hallmark:
       Lasso drives noise coefficients to EXACTLY zero, so a column with
       ``|coef| == 0`` is a noise reject by construction.

    The non-monotone LOSS (the documented expected behaviour): if the
    target is ``y = cos(x_1)``, every engineered column receives a Lasso
    coefficient of zero because Pearson correlation between cosine of
    Gaussian and Gaussian is zero. Lasso-pre-selection is the wrong
    scorer for that fixture; the user should pick a non-parametric
    Layer 21 / 65-74 variant instead.

    Returns
    -------
    (X_augmented, scores)
        X_augmented : ``X`` with the Lasso-ranked top-K winners appended.
        scores : the full Lasso-coef ranking DataFrame (winners + rejects).
    """
    engineered = generate_univariate_basis_features(
        X, cols=cols, degrees=degrees, basis=basis,
    )
    if engineered.empty:
        return X.copy(), pd.DataFrame(columns=[
            "engineered_col", "source_col", "baseline_mi",
            "engineered_mi", "uplift",
        ])
    raw_X = X[[
        c for c in (cols or X.columns)
        if c in X.columns and pd.api.types.is_numeric_dtype(X[c])
    ]]
    scores = score_features_by_lasso_coef(
        raw_X, engineered, y,
        alpha=float(alpha), standardize=bool(standardize),
        random_state=int(random_state),
    )
    if scores.empty:
        return X.copy(), scores
    raw_baselines = scores["baseline_mi"].to_numpy()
    max_raw_baseline = (
        float(raw_baselines.max()) if raw_baselines.size else 0.0
    )
    legacy_floor = float(min_abs_mi_frac) * max(0.0, max_raw_baseline)
    # L1 floor: columns Lasso drove to EXACTLY zero are guaranteed
    # noise rejects; they never enter the qualified set regardless of the
    # uplift ratio (uplift = 0 / 0 is undefined / inf; the strict
    # ``engineered_mi > 0`` gate is the safety net).
    qualified = scores[
        (scores["engineered_mi"] > 0.0)
        & (scores["engineered_mi"] >= legacy_floor)
        & (scores["uplift"] >= float(min_uplift))
    ]
    winners = qualified.head(int(top_k))
    keep = list(winners["engineered_col"])
    X_aug = pd.concat([X, engineered[keep]], axis=1) if keep else X.copy()
    return X_aug, scores


def hybrid_orth_mi_lasso_fe_with_recipes(
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
    standardize: bool = True,
    random_state: int = 0,
):
    """Same as :func:`hybrid_orth_mi_lasso_fe` plus a list of
    ``orth_univariate`` recipes -- one per appended column -- so that
    ``MRMR.transform`` can recompute each engineered column on test data
    without re-running the Lasso fit.

    Recipes are byte-identical to Layer 21 because the engineered VALUES
    are byte-identical -- only the SCORING (and therefore the selection)
    differs.
    """
    from .engineered_recipes import build_orth_univariate_recipe

    X_aug, scores = hybrid_orth_mi_lasso_fe(
        X, y,
        cols=cols, degrees=degrees, basis=basis,
        top_k=top_k, min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        alpha=float(alpha), standardize=bool(standardize),
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
                rest = suffix[len(code):]
                if rest.isdigit():
                    chosen_basis = code_to_basis[code]
                    chosen_degree = int(rest)
                    break
        if chosen_basis is None or chosen_degree is None:
            logger.warning(
                "hybrid_orth_mi_lasso_fe_with_recipes: cannot parse "
                "basis/degree from column name %r; skipping recipe build.",
                name,
            )
            continue
        recipes.append(build_orth_univariate_recipe(
            name=name, src_name=src,
            basis=chosen_basis, degree=chosen_degree,
        ))
    return X_aug, scores, recipes
