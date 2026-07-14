"""Base-vs-residual prediction attribution for composite-target estimators.

A composite target splits the signal into a BASE part (recovered from the
base column(s) by the transform alone) and a RESIDUAL part (what the inner
model learned on top). This module decomposes each fitted
:class:`CompositeTargetEstimator` prediction back into those two parts,
purely post-hoc: no refit, no frame copy, one narrow base-column pull.

Semantics
---------
For the ADDITIVE / linear-residual family the inverse is
``y = T + base_level(base)`` (e.g. ``linear_residual``:
``y = T + alpha*base + beta``; ``diff``: ``y = T + base``). The
decomposition is EXACT and additive::

    base_contribution    = inverse(T=0,    base, params)   # the y at zero residual
    residual_contribution = y_hat - base_contribution
    base_contribution + residual_contribution == y_hat       # by construction

For the MULTIPLICATIVE family (``ratio``: ``y = T*base``; ``logratio``:
``y = base*exp(T)``; ``centered_ratio`` / ``rolling_quantile_ratio`` /
``geometric_mean_residual``) the natural split is a PRODUCT of a base
FACTOR and a residual FACTOR::

    base_contribution    = inverse(T=neutral, base, params)  # the base level / factor
    residual_contribution = y_hat / base_contribution
    base_contribution * residual_contribution == y_hat       # by construction

The returned table always carries a ``mode`` column ('additive' /
'multiplicative') so a consumer knows whether the two contributions SUM or
MULTIPLY to ``y_hat``. ``base_share`` is reported on a common scale (see
:func:`attribution_summary`) so additive and multiplicative targets are
comparable.

The base level is obtained by inverting the transform at its NEUTRAL residual
-- the residual value that contributes nothing (0 for an additive inverse,
the multiplicative-identity residual for a product inverse). This reuses the
fitted transform exactly, so the attribution is consistent with what
``predict`` actually computes; no transform-specific algebra is re-derived
here.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from .transforms import get_transform
from . import _extract_groups

# Transforms whose inverse is a PRODUCT ``y = base_factor * residual_factor``.
# Value is the NEUTRAL residual T whose inverse yields the pure base factor
# (the residual factor is then ``y_hat / base_factor``):
#   ratio                    y = T * base                 -> T=1 -> factor base
#   logratio                 y = base * exp(softcap(T))   -> T=0 -> factor base
#   centered_ratio           y = T * (base + c)           -> T=1 -> factor base+c
#   rolling_quantile_ratio   y = T * rolling_median(base) -> T=1 -> factor rollmed
#   geometric_mean_residual  y = T * geomean(bases)       -> T=1 -> factor geomean
_MULTIPLICATIVE_NEUTRAL_T: dict[str, float] = {
    "ratio": 1.0,
    "logratio": 0.0,
    "centered_ratio": 1.0,
    "rolling_quantile_ratio": 1.0,
    "rolling_quantile_ratio_centered": 1.0,
    "rolling_quantile_ratio_grouped": 1.0,
    "geometric_mean_residual": 1.0,
}


def _require_fitted_with_base(estimator: Any) -> Any:
    """Return the resolved transform; raise on unfitted or base-free estimators.

    Attribution is a BASE-vs-residual split, so it is only defined for
    transforms that consume a base column (``requires_base=True``). A pure
    unary y-transform (``log_y`` / ``cbrt_y`` / ``frac_diff`` / ...) has no
    base term to attribute against and raises ``ValueError``.
    """
    if not hasattr(estimator, "estimator_") or not hasattr(estimator, "fitted_params_"):
        raise NotFittedError("explain_prediction called before fit: the CompositeTargetEstimator " "has no fitted inner model / params yet. Call fit() first.")
    transform = get_transform(estimator.transform_name)
    if not transform.requires_base:
        raise ValueError(
            f"explain_prediction: transform '{estimator.transform_name}' is a "
            "base-free unary y-transform; base-vs-residual attribution is "
            "undefined (there is no base column to attribute against)."
        )
    return transform


def _base_level(estimator: Any, transform, base_arr: np.ndarray, params: dict[str, Any], inverse_kwargs: dict[str, Any]) -> np.ndarray:
    """Pure base level / factor: invert the transform at its NEUTRAL residual.

    Additive transforms use T=0 (the base level at zero residual); the
    multiplicative family uses the residual mapping to factor 1 (see
    ``_MULTIPLICATIVE_NEUTRAL_T``). One ``inverse`` call over the predict-side
    base, no per-row Python loop.
    """
    neutral = _MULTIPLICATIVE_NEUTRAL_T.get(estimator.transform_name, 0.0)
    n = base_arr.shape[0]
    t_neutral = np.full(n, neutral, dtype=np.float64)
    base_level = np.asarray(
        transform.inverse(t_neutral, base_arr, params, **inverse_kwargs),
        dtype=np.float64,
    ).reshape(-1)
    return base_level


def explain_prediction(estimator: Any, X: Any) -> pd.DataFrame:
    """Decompose each composite prediction into base + residual contributions.

    Parameters
    ----------
    estimator
        A FITTED :class:`CompositeTargetEstimator` (additive / linear-residual
        or multiplicative transform; a base-free unary transform raises).
    X
        Predict-time feature frame (pandas / polars / ndarray) -- the same
        object accepted by ``estimator.predict``. Not copied; only the narrow
        base column(s) are pulled.

    Returns
    -------
    pandas.DataFrame with one row per input row and columns:
        ``base_contribution`` -- the y-level the transform recovers from the
        base alone (additive: ``inverse(T=0, base)``; multiplicative: the base
        factor).
        ``residual_contribution`` -- the inner model's part on the y-scale
        (additive: ``y_hat - base_contribution``, so the two SUM to ``y_hat``;
        multiplicative: ``y_hat / base_factor``, so the two MULTIPLY to
        ``y_hat``).
        ``y_hat`` -- the wrapper's point prediction.
        ``mode`` -- 'additive' or 'multiplicative', telling the consumer which
        combine rule reconstructs ``y_hat``.

    The decomposition reconstructs ``y_hat`` EXACTLY by construction (sum for
    additive, product for multiplicative), modulo the multiplicative
    edge-case where ``base_factor == 0`` (then ``residual_contribution`` is set
    to NaN -- a zero base factor cannot carry the whole prediction).
    """
    transform = _require_fitted_with_base(estimator)
    params = estimator.fitted_params_

    base_columns = estimator._resolve_base_columns()
    base_arr = estimator._extract_base_for_transform(X, base_columns)

    inverse_kwargs: dict[str, Any] = {}
    if transform.requires_groups:
        if not estimator.group_column:
            raise ValueError(f"explain_prediction: transform '{estimator.transform_name}' " "requires groups but group_column is not configured.")
        inverse_kwargs["groups"] = _extract_groups(X, estimator.group_column)

    y_hat = np.asarray(estimator.predict(X), dtype=np.float64).reshape(-1)
    base_contribution = _base_level(estimator, transform, base_arr, params, inverse_kwargs)

    multiplicative = estimator.transform_name in _MULTIPLICATIVE_NEUTRAL_T
    if multiplicative:
        # residual FACTOR; base_factor * residual_factor == y_hat. A zero base
        # factor cannot carry the prediction multiplicatively -> NaN (caller
        # decides; the additive split below is unaffected).
        with np.errstate(divide="ignore", invalid="ignore"):
            residual_contribution = np.where(
                base_contribution != 0.0, y_hat / base_contribution, np.nan,
            )
        mode = "multiplicative"
    else:
        residual_contribution = y_hat - base_contribution
        mode = "additive"

    return pd.DataFrame({
        "base_contribution": base_contribution,
        "residual_contribution": residual_contribution,
        "y_hat": y_hat,
        "mode": mode,
    })


def attribution_summary(estimator: Any, X: Any) -> dict[str, Any]:
    """Mean absolute base-vs-residual share across a dataset.

    Reports how much of the predicted signal the BASE explains vs the inner
    model, on a common [0, 1] share scale that is comparable across additive
    and multiplicative targets:

    - additive: the share is taken on each contribution's VARIATION about its
      own mean -- ``share = mean|base_c - mean(base_c)| / (mean|base_c -
      mean(base_c)| + mean|resid_c - mean(resid_c)|)``. Deviation-about-mean
      (not raw magnitude) is what isolates the base COLUMN's informative drive:
      the constant OLS intercept ``beta`` is folded into ``base_contribution``
      as a level offset and carries no base-column information, so a raw-
      magnitude share would credit the base for the whole target level even
      when the base column is pure noise (then ``base_contribution = beta``,
      a constant, contributing zero variation -> share near 0, as intended).
      A base-dominated target has ``base_share`` near 1; a residual-dominated
      target near 0.
    - multiplicative: the contributions are FACTORS, so the share is taken in
      log-magnitude space -- ``share = |log|base_factor|| / (|log|base_factor||
      + |log|residual_factor||)`` per row (rows with a non-finite / zero factor
      are dropped from the mean). A base factor that moves y far from 1 while
      the residual factor stays near 1 gives a high base share.

    Returns a dict: ``base_share`` (float in [0, 1]), ``residual_share``
    (= 1 - base_share), ``mode``, and ``n_rows`` used.
    """
    table = explain_prediction(estimator, X)
    mode = str(table["mode"].iloc[0]) if len(table) else "additive"
    bc = table["base_contribution"].to_numpy(dtype=np.float64)
    rc = table["residual_contribution"].to_numpy(dtype=np.float64)

    if mode == "multiplicative":
        # Share in log-magnitude space (factors combine multiplicatively).
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.abs(np.log(np.abs(bc)))
            lr = np.abs(np.log(np.abs(rc)))
        denom = lb + lr
        valid = np.isfinite(lb) & np.isfinite(lr) & (denom > 0)
        if not valid.any():
            base_share = float("nan")
            n_used = 0
        else:
            base_share = float(np.mean(lb[valid] / denom[valid]))
            n_used = int(valid.sum())
    else:
        # Variation about each contribution's own mean: the constant intercept
        # / level is non-informative for the base COLUMN, so a base column that
        # only sets a fixed offset (alpha~0, beta~mean(y)) contributes ~0
        # variation and gets ~0 share, while a base that drives the row-to-row
        # signal gets a high share.
        finite = np.isfinite(bc) & np.isfinite(rc)
        if not finite.any():
            base_share = float("nan")
            n_used = 0
        else:
            bcf, rcf = bc[finite], rc[finite]
            dev_b = float(np.mean(np.abs(bcf - bcf.mean())))
            dev_r = float(np.mean(np.abs(rcf - rcf.mean())))
            denom = dev_b + dev_r
            base_share = (dev_b / denom) if denom > 0 else float("nan")
            n_used = int(finite.sum())

    residual_share = (1.0 - base_share) if np.isfinite(base_share) else float("nan")
    return {
        "base_share": base_share,
        "residual_share": residual_share,
        "mode": mode,
        "n_rows": n_used,
    }
