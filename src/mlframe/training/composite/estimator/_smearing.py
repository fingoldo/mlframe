"""Smearing correction for non-linear unary target transforms (log / cbrt / Yeo-Johnson / Box-Cox / signed power).

A model fitted on ``T = f(y)`` predicts E[T | x]. Inverting that, ``f^-1(E[T | x])``, is not E[y | x] when ``f^-1`` is
curved: for ``log_y`` it is the geometric mean, systematically BELOW the mean on a right-skewed target. Judged by y-scale
RMSE (the discovery gate and the suite verdicts) such a composite therefore always loses to the raw target: a production
run dropped ``target_hourly_rate-logY`` at -2.3% honest gain and none of its log/cbrt composites beat raw.

Duan's smearing estimator averages the inverse over the empirical residual distribution: ``E[y|x] ~ mean_k f^-1(t_hat
+ r_k)``. The residuals are summarised by ``N_SMEAR_QUANTILES`` quantiles of the inner model's training residuals, so
predict costs ``N_SMEAR_QUANTILES`` inverse calls. Training residuals of a boosted model are smaller than honest ones, so
the correction errs on the side of too little, never too much.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, cast

import numpy as np

SMEARED_TRANSFORMS = frozenset({"log_y", "cbrt_y", "yeo_johnson_y", "box_cox_y", "signed_power_y"})
N_SMEAR_QUANTILES = 32
_MAX_RESIDUAL_ROWS = 50_000


def residual_quantiles(estimator: Any, X: Any, t_train: np.ndarray, seed: int = 0) -> Optional[np.ndarray]:
    """``N_SMEAR_QUANTILES`` mid-quantiles of ``t_train - estimator.predict(X)`` on a bounded row sample, or None."""
    n = int(t_train.shape[0])
    if n < N_SMEAR_QUANTILES * 4:
        return None
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(n, _MAX_RESIDUAL_ROWS), replace=False))
    try:
        X_s = X.iloc[idx] if hasattr(X, "iloc") else X[idx]  # pandas by position; polars / numpy by row index
        pred = np.asarray(estimator.predict(X_s), dtype=np.float64).reshape(-1)
    except Exception:
        return None
    r = np.asarray(t_train, dtype=np.float64)[idx] - pred
    r = r[np.isfinite(r)]
    if r.size < N_SMEAR_QUANTILES * 4:
        return None
    return cast(Optional[np.ndarray], np.quantile(r, (np.arange(N_SMEAR_QUANTILES) + 0.5) / N_SMEAR_QUANTILES))


def smeared_inverse(inverse: Callable[[np.ndarray], np.ndarray], t_hat: np.ndarray, quantiles: Optional[np.ndarray]) -> np.ndarray:
    """``mean_k inverse(t_hat + q_k)``; plain ``inverse(t_hat)`` when no residual quantiles were fitted."""
    if quantiles is None or len(quantiles) == 0:
        return np.asarray(inverse(t_hat), dtype=np.float64)
    acc = np.zeros(t_hat.shape[0], dtype=np.float64)
    for q in np.asarray(quantiles, dtype=np.float64):
        acc += np.asarray(inverse(t_hat + q), dtype=np.float64).reshape(-1)
    return acc / len(quantiles)


def smeared_prediction(transform_name: str, model: Any, x_fit: Any, t_fit: np.ndarray, t_hat: np.ndarray, inverse: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Discovery-side twin of the estimator's predict: invert ``t_hat`` with smearing for the curved unary transforms.

    Residual quantiles come from ``model`` on its own fit rows, exactly as ``CompositeTargetEstimator.fit`` does, so a
    spec is scored the way the trained composite will predict. Other transforms get the plain inverse.
    """
    q = None
    if transform_name in SMEARED_TRANSFORMS:
        r = np.asarray(t_fit, dtype=np.float64) - np.asarray(model.predict(x_fit), dtype=np.float64).reshape(-1)
        r = r[np.isfinite(r)]
        if r.size >= 4 * N_SMEAR_QUANTILES:
            q = np.quantile(r, (np.arange(N_SMEAR_QUANTILES) + 0.5) / N_SMEAR_QUANTILES)
    return smeared_inverse(inverse, t_hat, q)
