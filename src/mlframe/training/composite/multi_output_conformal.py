"""Per-output split-conformal intervals for ``CompositeMultiOutputEstimator``.

The multi-output wrapper holds one independent :class:`CompositeTargetEstimator`
per output column. Each per-column wrapper already exposes the single-output
conformal API (``calibrate_conformal`` / ``predict_interval``) backed by the
shared :func:`conformal_quantile` helper. This module lifts that to the vector
target: calibrate split-conformal INDEPENDENTLY per output column (column ``k``
gets its OWN residual radius from its own held-out residuals) and stack the
per-column ``(lower, upper)`` bands into ``(n, K)`` arrays.

Why per-column (not one shared radius)
--------------------------------------
The whole point of the multi-output composite is that each column has its own
dominant base / scale. A single pooled radius would over-cover the tight columns
and under-cover the wide ones. Calibrating each column on its own residuals gives
the standard split-conformal guarantee MARGINALLY PER COLUMN: empirical coverage
of column ``k`` is ``>= 1 - alpha`` under exchangeability of the column-``k``
calibration / test residuals, for every ``k`` -- distribution-free, no Gaussian
or homoscedastic assumption, and no cross-column coupling.

Calibration MUST run on held-out rows (the suite val split or an OOF fold) the
inner per-column estimators never trained on -- conformal validity rests on the
calibration rows being exchangeable with the test rows.

Failed columns (a fully-NaN / degenerate output recorded at fit as a constant
fallback) carry NO inner estimator, so they get a degenerate ``+/- 0`` band
around their constant fallback -- a valid (if uninformative) interval that keeps
the stacked ``(n, K)`` output rectangular instead of crashing the vector.

Design choices mirror the rest of the composite package:
- Radii live on the per-column estimators' own ``_conformal_q_`` dicts (set by
  the single-output ``calibrate_conformal``); the multi-output wrapper records
  only which alphas it calibrated in ``self._mo_conformal_alphas_`` -- a small
  set of floats, so ``sklearn.clone`` / pickle stay clean and frame-free.
- ``y`` is column-sliced per output (cheap; ``y`` is the target, not the feature
  frame); ``X_cal`` / ``X`` are passed by reference to each per-column wrapper,
  never copied (safe on a 100+ GB frame).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def calibrate_conformal(self, X_cal, y_cal, alpha=0.1):
    """Calibrate split-conformal INDEPENDENTLY per output column.

    Each fitted per-column :class:`CompositeTargetEstimator` is calibrated on its
    OWN held-out residuals (column ``k`` of ``y_cal``), so column ``k`` gets its
    own radius via the shared :func:`conformal_quantile` helper. Marginal coverage
    ``>= 1 - alpha`` holds PER output column.

    Parameters
    ----------
    X_cal
        Held-out calibration features -- rows the per-column inners did NOT train
        on (the suite val split, or an OOF fold). Passed by reference to each
        per-column wrapper (never copied).
    y_cal
        Calibration targets, shape ``(n_cal, K)`` (a 1-D vector is treated as a
        single column). Column-sliced per output.
    alpha
        Miscoverage level in ``(0, 1)``; scalar or iterable of levels. Each level
        is calibrated and cached on every per-column estimator so
        :func:`predict_interval` can serve any pre-calibrated level cheaply.

    Returns ``self`` (sklearn-style). A failed column (constant fallback, no inner
    estimator) is skipped here and served as a degenerate ``+/- 0`` band at
    predict time.
    """
    if not hasattr(self, "estimators_"):
        from sklearn.exceptions import NotFittedError

        raise NotFittedError("CompositeMultiOutputEstimator.calibrate_conformal called before fit.")
    y2d = self._to_2d_targets(y_cal)
    if y2d.shape[1] != self.n_outputs_:
        raise ValueError("calibrate_conformal: y_cal has " f"{y2d.shape[1]} columns but the estimator was fit on " f"{self.n_outputs_}.")
    alphas = [alpha] if np.isscalar(alpha) else list(alpha)
    for k, est in enumerate(self.estimators_):
        if est is None:
            # Failed column: constant fallback, no residual distribution to fit.
            continue
        est.calibrate_conformal(X_cal, y2d[:, k], alpha=alpha)
    if not hasattr(self, "_mo_conformal_alphas_") or self._mo_conformal_alphas_ is None:
        self._mo_conformal_alphas_ = set()
    for a in alphas:
        self._mo_conformal_alphas_.add(round(float(a), 6))
    return self


def predict_interval(self, X, alpha=0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-output ``(lower, upper)`` arrays of shape ``(n, K)``.

    Each column's band is its per-column estimator's ``predict_interval`` at
    ``alpha`` (point ``+/- radius_k``, clipped to that column's train envelope),
    so marginal coverage ``>= 1 - alpha`` holds PER output column. Requires a
    prior :func:`calibrate_conformal` at this ``alpha`` (a clear error otherwise).

    A failed column (constant fallback, no inner estimator) gets a degenerate
    ``+/- 0`` band around its fallback constant so the stacked ``(n, K)`` output
    stays rectangular. A single-row ``X`` yields ``(1, K)`` bands.
    """
    if not hasattr(self, "estimators_"):
        from sklearn.exceptions import NotFittedError

        raise NotFittedError("CompositeMultiOutputEstimator.predict_interval called before fit.")
    key = round(float(alpha), 6)
    calibrated = getattr(self, "_mo_conformal_alphas_", None) or set()
    if key not in calibrated:
        raise RuntimeError(
            f"predict_interval: no per-column conformal radius calibrated for "
            f"alpha={alpha}. Call calibrate_conformal(X_cal, y_cal, alpha={alpha}) "
            f"on a held-out set first (calibrated levels: {sorted(calibrated)})."
        )
    K = self.n_outputs_
    lowers: list = [None] * K
    uppers: list = [None] * K
    n_rows = None
    for k, est in enumerate(self.estimators_):
        if est is None:
            continue
        lo_k, hi_k = est.predict_interval(X, alpha=alpha)
        lo_k = np.asarray(lo_k, dtype=np.float64).reshape(-1)
        hi_k = np.asarray(hi_k, dtype=np.float64).reshape(-1)
        lowers[k] = lo_k
        uppers[k] = hi_k
        if n_rows is None:
            n_rows = lo_k.shape[0]
    if n_rows is None:
        n_rows = self._infer_n_rows(X)
    lower = np.empty((n_rows, K), dtype=np.float64)
    upper = np.empty((n_rows, K), dtype=np.float64)
    for k in range(K):
        if lowers[k] is None:
            # Failed column: degenerate band at the constant fallback.
            fb = self.column_fallbacks_[k]
            lower[:, k] = fb
            upper[:, k] = fb
        else:
            lower[:, k] = lowers[k]
            upper[:, k] = uppers[k]
    return lower, upper
