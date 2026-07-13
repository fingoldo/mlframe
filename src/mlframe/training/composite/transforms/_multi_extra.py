"""Multi-base extensions: ``asinh_residual_multi`` and ``linear_residual_multi_robust``.

``asinh_residual_multi`` is the K-column sibling of ``asinh_residual``: a joint OLS of ``arcsinh(y)`` on the arcsinh-transformed base columns, with the
same condition-number multicollinearity guard as ``linear_residual_multi`` (the fit delegates to ``_linear_residual_multi_fit`` in arcsinh space, so the
BKW-scaled cond gate, NaN row masking and collinear fallback are inherited verbatim).

``linear_residual_multi_robust`` is the trimmed-LS sibling of ``linear_residual_robust`` for the K-column joint OLS: first-pass ``_linear_residual_multi_fit``,
drop rows with ``|resid| > _LINRES_ROBUST_MAD_K * sigma_MAD``, refit on the inliers. Forward / inverse / domain are the plain multi functions.
"""
from __future__ import annotations

from typing import Any

import numpy as np

# The linear sibling and the parent-resident thresholds are imported LAZILY inside function bodies: the linear sibling top-level-imports the parent,
# which imports the registry, which imports this module -- a top-level import here would grow the whitelisted transforms import SCC.


def _asinh_residual_multi_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Joint OLS of ``arcsinh(y) ~ arcsinh(base) @ alphas + beta`` via ``_linear_residual_multi_fit`` (inherits the cond-number guard + collinear fallback)."""
    from .linear import _linear_residual_multi_fit
    base_arr = np.asarray(base, dtype=np.float64)
    if base_arr.ndim == 1:
        base_arr = base_arr.reshape(-1, 1)
    yz = np.arcsinh(np.asarray(y, dtype=np.float64))
    bz = np.arcsinh(base_arr)
    return _linear_residual_multi_fit(yz, bz, sample_weight=sample_weight)


def _asinh_residual_multi_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Apply ``T = arcsinh(y) - arcsinh(base) @ alphas - beta``."""
    from .linear import _linear_residual_multi_forward
    base_arr = np.asarray(base, dtype=np.float64)
    if base_arr.ndim == 1:
        base_arr = base_arr.reshape(-1, 1)
    yz = np.arcsinh(np.asarray(y, dtype=np.float64))
    out: np.ndarray = np.asarray(_linear_residual_multi_forward(yz, np.arcsinh(base_arr), params), dtype=np.float64)
    return out


def _asinh_residual_multi_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    """Undo the transform: ``y = sinh(T_hat + arcsinh(base) @ alphas + beta)``."""
    from .linear import _linear_residual_multi_inverse
    base_arr = np.asarray(base, dtype=np.float64)
    if base_arr.ndim == 1:
        base_arr = base_arr.reshape(-1, 1)
    z = np.asarray(_linear_residual_multi_inverse(np.asarray(t_hat, dtype=np.float64), np.arcsinh(base_arr), params), dtype=np.float64)
    return np.asarray(np.sinh(z))


def _asinh_residual_multi_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    """Every finite row is admissible (arcsinh defined on all reals); delegates to the multi finite gate."""
    from .linear import _linear_residual_multi_domain
    return _linear_residual_multi_domain(y, base)


def _linear_residual_multi_robust_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Trimmed-LS joint OLS: multi OLS -> drop ``|resid| > _LINRES_ROBUST_MAD_K * sigma_MAD`` rows -> refit.

    Returns the same params dict shape as ``_linear_residual_multi_fit`` so the multi forward / inverse apply unchanged. Mirrors
    ``_linear_residual_robust_fit``'s guards: falls back to the first pass (stamped ``is_redundant_with_linres_multi=True``) when the residual scale is
    degenerate, the trim would drop more than ``1 - _LINRES_ROBUST_MIN_KEEP_FRAC`` of rows, or no row is trimmed (second pass would be identical).
    """
    from .linear import _linear_residual_multi_fit, _linear_residual_multi_inverse
    from . import _LINRES_ROBUST_MAD_K, _LINRES_ROBUST_MIN_KEEP_FRAC
    base_arr = np.asarray(base, dtype=np.float64)
    if base_arr.ndim == 1:
        base_arr = base_arr.reshape(-1, 1)
    y_f = np.asarray(y, dtype=np.float64).reshape(-1)
    first_pass = _linear_residual_multi_fit(y_f, base_arr, sample_weight=sample_weight)
    if first_pass.get("collinear_fallback"):
        first_pass["is_redundant_with_linres_multi"] = True
        return first_pass
    row_finite = np.isfinite(y_f) & np.all(np.isfinite(base_arr), axis=1)
    yc = y_f[row_finite]
    bc = base_arr[row_finite]
    swc = np.asarray(sample_weight, dtype=np.float64).reshape(-1)[row_finite] if sample_weight is not None else None
    n = yc.size
    if n < bc.shape[1] + 2:
        first_pass["is_redundant_with_linres_multi"] = True
        return first_pass
    resid = yc - _linear_residual_multi_inverse(np.zeros(n, dtype=np.float64), bc, first_pass)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma_mad = mad * 1.4826
    if sigma_mad <= 0.0 or not np.isfinite(sigma_mad):
        first_pass["is_redundant_with_linres_multi"] = True
        return first_pass
    keep = np.abs(resid - med) <= _LINRES_ROBUST_MAD_K * sigma_mad
    n_kept = int(keep.sum())
    if n_kept < max(bc.shape[1] + 2, int(_LINRES_ROBUST_MIN_KEEP_FRAC * n)):
        first_pass["is_redundant_with_linres_multi"] = True
        return first_pass
    if n_kept == n:
        first_pass["is_redundant_with_linres_multi"] = True
        return first_pass
    sw2 = swc[keep] if swc is not None else None
    result = _linear_residual_multi_fit(yc[keep], bc[keep], sample_weight=sw2)
    result["is_redundant_with_linres_multi"] = False
    return result
