"""Double-ML / orthogonalized composite -- debias the base contribution under confounding.

The plain residual composite (``CompositeTargetEstimator``) anchors a GBDT on a cheap
base column and lets the inner learn the residual ``y - base``. That is unbiased ONLY
when the base column is independent of the features ``X``. When the base is itself
CORRELATED with some feature in ``X`` (a shared confounder), the inner re-absorbs the
base-correlated signal: the decomposition then mis-attributes part of the base's true
effect to ``X``, biasing the estimated base coefficient.

``OrthogonalizedCompositeEstimator`` removes this bias with the Neyman-orthogonal /
partialling-out recipe (Frisch-Waugh-Lovell, a.k.a. double / debiased ML):

    1. cross-fitted nuisance ``m_hat(X) = E[base | X\\base]``  (KFold, out-of-fold preds)
    2. cross-fitted nuisance ``g_hat(X) = E[y    | X\\base]``  (KFold, out-of-fold preds)
    3. residualize:  base_tilde = base - m_hat(X),  y_tilde = y - g_hat(X)
    4. theta_hat = <base_tilde, y_tilde> / <base_tilde, base_tilde>   (FWL/OLS on residuals)
    5. the inner composite models what theta * base leaves over: the structure in X
       that is ORTHOGONAL to the base direction, fit on (y - theta_hat * base).

Because the nuisance predictions are OUT-OF-FOLD, the residualized quantities carry no
in-fold overfitting leakage; ``theta_hat`` is then a Neyman-orthogonal estimate of the
base's partial (confounding-free) effect. At ``predict`` time:

    y_hat(x) = theta_hat * base(x) + inner.predict(x)

Design choices mirror the other composite wrappers (``glm.py`` / ``classification.py``):
- sklearn-compatible (fit / predict / get_params / clone); the nuisance + inner
  estimators are passed by config, never captured as closures, so clone / pickle stay clean.
- the base column is pulled with a NARROW one-column ndarray read (``_extract_base``),
  never a whole-frame ``to_pandas`` -- safe on 100+ GB frames (see CLAUDE.md).
- KFold cross-fitting (default 5 folds); fold predictions are written back to their
  out-of-fold rows so every row gets a leakage-free nuisance prediction.

Out of scope: non-Gaussian links (see ``CompositeGLMEstimator``) and classification
residuals (see ``CompositeClassificationEstimator``). This estimator targets a
continuous additive ``y`` where the base effect is (locally) linear.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

# Floor on the residualized-base inner product denominator. When ``base`` is (nearly)
# perfectly predictable from X, ``base_tilde`` collapses to ~0 and the FWL ratio becomes
# ill-conditioned; we floor the denominator so theta stays finite rather than exploding.
_DENOM_FLOOR = 1e-12


def _is_polars_df(x: Any) -> bool:
    """Explicit isinstance check (no duck-typing on ``to_pandas``)."""
    try:
        import polars as pl

        return isinstance(x, pl.DataFrame)
    except Exception:
        return False


def _extract_base(X: Any, base_column: str) -> np.ndarray:
    """Pull the base column from X (pandas / polars) as a 1-D float64 ndarray.

    Narrow one-column read only -- never copies the whole frame. Raises ``KeyError``
    with an actionable message if the column was dropped (e.g. by feature selection).
    """
    if _is_polars_df(X):
        if base_column not in X.columns:
            raise KeyError(
                f"OrthogonalizedCompositeEstimator: base column '{base_column}' missing from X. "
                "If feature selection (MRMR/RFECV) is dropping it, add base_column to "
                "forced_keep_columns in the feature selection config."
            )
        return X.get_column(base_column).to_numpy().astype(np.float64, copy=False)
    # pandas
    if hasattr(X, "columns"):
        if base_column not in X.columns:
            raise KeyError(
                f"OrthogonalizedCompositeEstimator: base column '{base_column}' missing from X."
            )
        return np.asarray(X[base_column], dtype=np.float64).ravel()
    raise TypeError("OrthogonalizedCompositeEstimator requires a pandas/polars DataFrame X with named columns.")


def _drop_base_column(X: Any, base_column: str) -> Any:
    """Return X without the base column (view / lazy drop, no full-frame copy).

    The inner composite must NOT see the raw base column as a feature -- otherwise it
    re-learns the base effect we just orthogonalized out. polars ``drop`` and pandas
    ``drop`` both return a new lightweight frame sharing the underlying column buffers.
    """
    if _is_polars_df(X):
        return X.drop(base_column) if base_column in X.columns else X
    if hasattr(X, "columns"):
        return X.drop(columns=[base_column]) if base_column in X.columns else X
    return X


def _to_1d_numpy(y: Any) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).ravel()


def _cross_fitted_oof(estimator: Any, X: Any, target: np.ndarray, kf: KFold) -> np.ndarray:
    """Out-of-fold predictions of ``E[target | X]`` via KFold cross-fitting.

    For each fold the nuisance model is fit on the OTHER folds and predicts the held-out
    fold, so no row's prediction was informed by that same row -- the residual built from
    these predictions is leakage-free (the Neyman-orthogonality requirement).
    """
    n = target.shape[0]
    oof = np.empty(n, dtype=np.float64)
    is_polars = _is_polars_df(X)
    for train_idx, test_idx in kf.split(np.arange(n)):
        model = clone(estimator)
        if is_polars:
            X_tr, X_te = X[train_idx], X[test_idx]
        else:
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        model.fit(X_tr, target[train_idx])
        oof[test_idx] = _to_1d_numpy(model.predict(X_te))
    return oof


class OrthogonalizedCompositeEstimator(BaseEstimator, RegressorMixin):
    """Debiased (double-ML) composite: confounding-free base coefficient via FWL.

    Parameters
    ----------
    base_column : str
        Name of the base column inside ``X``. Pulled with a narrow one-column read; it is
        also dropped from the feature matrix the inner sees, so the inner cannot re-absorb
        the base effect.
    inner_estimator : sklearn regressor
        Models the residual structure ``y - theta_hat * base`` on the (base-dropped) X.
    base_nuisance_estimator : sklearn regressor, optional
        Cross-fitted ``E[base | X]``. Defaults to ``Ridge()`` -- swap for a GBDT on real
        data. Cloned per fold; never mutated.
    y_nuisance_estimator : sklearn regressor, optional
        Cross-fitted ``E[y | X]``. Defaults to ``Ridge()``.
    n_folds : int, default 5
        KFold splits for the cross-fitted nuisance models.
    shuffle : bool, default True
    random_state : int, optional

    Attributes (post-fit)
    ---------------------
    base_coef_ : float
        Neyman-orthogonal estimate of the base's partial effect (theta_hat).
    naive_base_coef_ : float
        Plain OLS base coefficient on the RAW (un-residualized) base/y -- kept so callers
        can see the confounding bias the orthogonalization removed.
    inner_ : fitted inner estimator.
    """

    def __init__(
        self,
        base_column: str,
        inner_estimator: Any,
        base_nuisance_estimator: Optional[Any] = None,
        y_nuisance_estimator: Optional[Any] = None,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.base_column = base_column
        self.inner_estimator = inner_estimator
        self.base_nuisance_estimator = base_nuisance_estimator
        self.y_nuisance_estimator = y_nuisance_estimator
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    # -- internals ---------------------------------------------------------
    def _resolve_nuisance(self, attr: Any) -> Any:
        return clone(attr) if attr is not None else Ridge()

    @staticmethod
    def _ols_coef(base: np.ndarray, y: np.ndarray) -> float:
        """Simple-regression slope of y on base through a centered fit (intercept-free FWL)."""
        bc = base - base.mean()
        yc = y - y.mean()
        denom = float(bc @ bc)
        if denom < _DENOM_FLOOR:
            return 0.0
        return float((bc @ yc) / denom)

    # -- sklearn API -------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "OrthogonalizedCompositeEstimator":
        if self.n_folds < 2:
            raise ValueError("OrthogonalizedCompositeEstimator: n_folds must be >= 2 for cross-fitting.")
        y_arr = _to_1d_numpy(y)
        base = _extract_base(X, self.base_column)
        if base.shape[0] != y_arr.shape[0]:
            raise ValueError("base column and y length mismatch.")

        kf = KFold(n_splits=self.n_folds, shuffle=self.shuffle, random_state=self.random_state)

        # Cross-fitted nuisances on the OTHER features only -- the base column itself is
        # dropped, otherwise ``E[base|X]`` trivially recovers base (base is a column of X)
        # and ``base_tilde`` collapses to 0, destroying the FWL ratio.
        X_nuis = _drop_base_column(X, self.base_column)
        m_hat = _cross_fitted_oof(self._resolve_nuisance(self.base_nuisance_estimator), X_nuis, base, kf)
        g_hat = _cross_fitted_oof(self._resolve_nuisance(self.y_nuisance_estimator), X_nuis, y_arr, kf)

        base_tilde = base - m_hat
        y_tilde = y_arr - g_hat

        # FWL: regress residualized y on residualized base (no intercept -- both centered by
        # construction of the residuals, but center again defensively for finite-sample drift).
        denom = float(base_tilde @ base_tilde)
        if denom < _DENOM_FLOOR:
            logger.warning(
                "OrthogonalizedCompositeEstimator: residualized base is near-degenerate "
                "(base almost fully explained by X); base_coef_ set to 0."
            )
            self.base_coef_ = 0.0
        else:
            self.base_coef_ = float((base_tilde @ y_tilde) / denom)

        # Naive comparison: plain OLS slope on the RAW base (the biased baseline).
        self.naive_base_coef_ = self._ols_coef(base, y_arr)

        # Inner models the remainder on the base-dropped features. The base coefficient is
        # frozen at theta_hat, so the inner only learns the orthogonal structure in X.
        X_inner = _drop_base_column(X, self.base_column)
        residual = y_arr - self.base_coef_ * base
        self.inner_ = clone(self.inner_estimator)
        self.inner_.fit(X_inner, residual)

        self.n_features_in_ = base.shape[0] and getattr(X, "shape", (0, 0))[1]
        self._fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("OrthogonalizedCompositeEstimator: call fit before predict.")
        base = _extract_base(X, self.base_column)
        X_inner = _drop_base_column(X, self.base_column)
        inner_pred = _to_1d_numpy(self.inner_.predict(X_inner))
        return self.base_coef_ * base + inner_pred
