"""Soft base-shrink inverse guard for the base-additive composite transforms.

A base-additive inverse (``y = T_hat + alpha*base + beta`` and its ``diff`` /
``additive_residual`` / multi / grouped siblings) EXTRAPOLATES linearly in ``base``.
When a predict-time ``base`` lands OUTSIDE the range seen at fit -- the classic
unseen-group / out-of-distribution tail -- that extrapolation makes the point
estimate collapse. Today's only guard routes NaN/inf inverses to ``fallback_predict``;
it does nothing for the common case where the inverse is FINITE but wildly off.

This module adds two smooth, measured guards, gated default-ON via
``getattr(self, "soft_base_shrink", True)``:

1. Soft-shrink. At fit we record each base column's calibration range ``[lo, hi]``
   (the min/max actually seen) plus its robust spread (IQR). At predict a base value
   ``b`` beyond the boundary is SOFT-clipped, NOT hard-clamped::

       u = distance beyond the boundary (b - hi, or lo - b)
       base_eff = boundary + iqr * u / (iqr + u)     (equivalently boundary + u/(1+d), d = u/iqr)

   ``s(d) = 1/(1+d)`` decays with the out-of-range distance ``d`` in IQR units, so the
   contribution beyond the boundary saturates at one IQR (bounded inverse) and further-out
   values shrink MORE. The map is C1-continuous at the boundary (slope 1 on both sides, no
   jump) and monotone increasing in ``b`` everywhere -- a hard clamp would bias in-range rows
   and introduce a kink; a soft-clip does neither. In-range values (``lo <= b <= hi``) are
   returned bit-for-bit unchanged, so ``soft_base_shrink=True`` is byte-identical to the raw
   additive inverse everywhere inside the fit range.

2. Smart fallback. Rows shrunk beyond ``soft_base_shrink_severity_iqr`` IQRs (deeply OOD) are
   routed to a BETTER estimate than the shrunk inverse / global median: the causal-lag value
   when a lag column is present (the AR failsafe ``y_hat = lag``; for a persistent target an
   unseen-tail ``base`` predicts ``y`` far better than the central median), else the wrapper's
   existing ``fallback_predict``. A per-row flag is exposed on ``self.soft_shrink_info_``.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import numba as _numba
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - numba optional
    _numba = None
    _HAS_NUMBA = False

logger = logging.getLogger(__name__)

# The inverse is linear-in-base (``y = T + <linear combo of base> + const``) exactly for these
# transforms, so an out-of-range base directly scales the extrapolation. Other transforms either
# have no base, already edge-clip the base (median_residual / monotonic_residual lookups), or use a
# non-additive inverse (ratio / logratio) -- soft-shrink is scoped out of those so they stay
# byte-identical.
ADDITIVE_BASE_TRANSFORMS = frozenset({
    "diff",
    "additive_residual",
    "linear_residual",
    "linear_residual_robust",
    "theilsen_residual",
    "linear_residual_multi",
    "linear_residual_grouped",
})

# Rows shrunk more than this many IQRs beyond the calibration boundary are DEEPLY out-of-distribution
# and route to the smart fallback rather than trusting even the bounded shrunk inverse. Overridable per
# instance via ``soft_base_shrink_severity_iqr`` (the parent registers it alongside ``soft_base_shrink``).
DEFAULT_SEVERITY_IQR = 3.0

# Key under which the per-column fit-range (lo / hi / iqr arrays) is stashed inside ``fitted_params_``.
BASE_FIT_RANGE_KEY = "base_fit_range"


def _iqr_floor(col_finite: np.ndarray, lo: float, hi: float) -> float:
    """Positive robust spread for one base column: IQR, or the range / a tiny scale when IQR is 0.

    A heavily-tied or constant base gives IQR == 0, which would make the out-of-range distance ``u/iqr``
    blow up. Fall back to the full [min,max] range, then to a scale-relative floor, so the shrink always
    has a well-defined, positive length unit.
    """
    q25, q75 = (float(v) for v in np.quantile(col_finite, (0.25, 0.75)))
    spread = q75 - q25
    if spread > 0.0:
        return spread
    rng = hi - lo
    if rng > 0.0:
        return rng
    return max(abs(hi) * 1e-3, 1e-12)


def capture_base_fit_range(self, transform, base_train: np.ndarray) -> None:
    """Record the base calibration range into ``self.fitted_params_[BASE_FIT_RANGE_KEY]`` (fit-time).

    No-op for non-base / non-additive transforms and when no base column has any finite train value, so
    the 30+ other transforms keep byte-identical ``fitted_params_`` and predict stays on the raw inverse.
    """
    if not getattr(transform, "requires_base", False):
        return
    if getattr(transform, "name", "") not in ADDITIVE_BASE_TRANSFORMS:
        return
    b = np.asarray(base_train, dtype=np.float64)
    b2 = b.reshape(-1, 1) if b.ndim == 1 else b
    K = b2.shape[1]
    lo = np.empty(K, dtype=np.float64)
    hi = np.empty(K, dtype=np.float64)
    iqr = np.empty(K, dtype=np.float64)
    for j in range(K):
        col = b2[:, j]
        col = col[np.isfinite(col)]
        if col.size == 0:
            return  # a base column with no finite train value -> cannot calibrate; leave range absent.
        lo[j] = float(col.min())
        hi[j] = float(col.max())
        iqr[j] = _iqr_floor(col, lo[j], hi[j])
    params = getattr(self, "fitted_params_", None)
    if isinstance(params, dict):
        params[BASE_FIT_RANGE_KEY] = {"lo": lo, "hi": hi, "iqr": iqr}


def is_enabled(self, transform, params: dict[str, Any]) -> bool:
    """True when soft base-shrink should run for this predict call.

    Requires the gate flag ON (default True), a base-additive transform, and a fit-range captured at fit
    (absent for ``from_fitted_inner`` / legacy fitted instances -> disabled -> raw inverse preserved).
    """
    if not getattr(self, "soft_base_shrink", True):
        return False
    if not getattr(transform, "requires_base", False):
        return False
    if getattr(transform, "name", "") not in ADDITIVE_BASE_TRANSFORMS:
        return False
    rng = params.get(BASE_FIT_RANGE_KEY) if isinstance(params, dict) else None
    return isinstance(rng, dict) and "lo" in rng


def _shrink_base_numpy(
    b2: np.ndarray, lo: np.ndarray, hi: np.ndarray, iqr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference vectorised implementation of the soft-clip; ``b2`` is 2-D ``(n, K)``.

    Returns ``(base_eff2, d_row)`` where ``base_eff2`` copies ``b2`` with only out-of-range finite entries
    soft-clipped. Kept as the njit-free fallback + the parity oracle for the numba kernel.
    """
    n, K = b2.shape
    hi_b = np.broadcast_to(hi, (n, K))
    lo_b = np.broadcast_to(lo, (n, K))
    iqr_b = np.broadcast_to(iqr, (n, K))
    finite = np.isfinite(b2)
    above = finite & (b2 > hi_b)
    below = finite & (b2 < lo_b)
    base_eff2 = b2.copy()
    d_full = np.zeros((n, K), dtype=np.float64)
    if above.any():
        u = b2[above] - hi_b[above]
        d = u / iqr_b[above]
        base_eff2[above] = hi_b[above] + u / (1.0 + d)
        d_full[above] = d
    if below.any():
        u = lo_b[below] - b2[below]
        d = u / iqr_b[below]
        base_eff2[below] = lo_b[below] - u / (1.0 + d)
        d_full[below] = d
    return base_eff2, d_full.max(axis=1)


if _HAS_NUMBA:
    @_numba.njit(cache=True, parallel=True)  # fastmath OFF: the isfinite gate must survive (non-finite base = domain violation).
    def _shrink_base_kernel(b2, lo, hi, iqr, base_eff2, d_row):  # pragma: no cover - compiled
        """Parallel njit core of the soft base-shrink: per-row IQR-relative clip of ``b2`` into ``[lo, hi]``, writing the shrunk base into ``base_eff2`` and the max relative distance into ``d_row``."""
        n, K = b2.shape
        for i in _numba.prange(n):
            dmax = 0.0
            for j in range(K):
                v = b2[i, j]
                if np.isfinite(v):
                    if v > hi[j]:
                        u = v - hi[j]
                        d = u / iqr[j]
                        base_eff2[i, j] = hi[j] + u / (1.0 + d)
                        if d > dmax:
                            dmax = d
                    elif v < lo[j]:
                        u = lo[j] - v
                        d = u / iqr[j]
                        base_eff2[i, j] = lo[j] - u / (1.0 + d)
                        if d > dmax:
                            dmax = d
            d_row[i] = dmax


def shrink_base(
    base_arr: np.ndarray, lo: np.ndarray, hi: np.ndarray, iqr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-clip out-of-range base values toward the calibration boundary; return ``(base_eff, d_row)``.

    ``base_arr`` is ``(n,)`` or ``(n, K)``; ``lo`` / ``hi`` / ``iqr`` are ``(K,)``. ``d_row`` is the per-row
    worst-column out-of-range distance in IQR units (0 for in-range / non-finite rows). In-range and
    non-finite entries pass through unchanged (the in-range path is byte-identical). A fused njit kernel
    (one parallel pass, no temporaries) does the work; falls back to the vectorised numpy oracle when numba
    is unavailable -- both are bit-identical.
    """
    b2 = base_arr if base_arr.ndim == 2 else base_arr.reshape(-1, 1)
    n, _K = b2.shape
    if _HAS_NUMBA:
        b2c = np.ascontiguousarray(b2, dtype=np.float64)
        base_eff2 = b2c.copy()
        d_row = np.empty(n, dtype=np.float64)
        _shrink_base_kernel(
            b2c, np.ascontiguousarray(lo, dtype=np.float64),
            np.ascontiguousarray(hi, dtype=np.float64),
            np.ascontiguousarray(iqr, dtype=np.float64), base_eff2, d_row,
        )
    else:
        base_eff2, d_row = _shrink_base_numpy(b2, lo, hi, iqr)
    base_eff = base_eff2 if base_arr.ndim == 2 else base_eff2.reshape(-1)
    return base_eff, d_row


def compute(self, transform, base_arr: np.ndarray, params: dict[str, Any]):
    """Return ``(base_eff, shrunk_mask, deep_ood_mask)`` for the predict path.

    When soft-shrink is disabled/inapplicable returns ``(base_arr, None, None)`` (the caller then runs the
    raw inverse, byte-identical). When enabled, ``shrunk_mask`` marks rows shrunk at all and
    ``deep_ood_mask`` marks rows shrunk beyond the severity threshold (candidates for the smart fallback).
    """
    if not is_enabled(self, transform, params):
        return base_arr, None, None
    rng = params[BASE_FIT_RANGE_KEY]
    lo = np.asarray(rng["lo"], dtype=np.float64)
    hi = np.asarray(rng["hi"], dtype=np.float64)
    iqr = np.asarray(rng["iqr"], dtype=np.float64)
    K_base = 1 if base_arr.ndim == 1 else base_arr.shape[1]
    if K_base != lo.size:
        # Base column count drifted from fit (mis-wired predict frame); disable rather than mis-shrink.
        return base_arr, None, None
    base_eff, d_row = shrink_base(base_arr, lo, hi, iqr)
    severity = float(getattr(self, "soft_base_shrink_severity_iqr", DEFAULT_SEVERITY_IQR))
    shrunk = d_row > 0.0
    deep_ood = d_row > severity
    return base_eff, shrunk, deep_ood


def _resolve_lag_column(self, X) -> str | None:
    """Name of the causal-lag column for the smart fallback, or None.

    Prefers a stamped target name (via ``detect_causal_lag_column``); falls back to the wrapper's own
    ``base_column`` when it carries a causal-lag suffix (the common AR composite case where base == y_prev),
    so no target name is needed. Import is lazy to avoid an estimator<->discovery import cycle.
    """
    from ..discovery._causal_lag import detect_causal_lag_column, CAUSAL_LAG_SUFFIXES
    for attr in ("target_name_", "_target_name", "target_column", "target_name"):
        tname = getattr(self, attr, None)
        if tname:
            col = detect_causal_lag_column(X, tname)
            if col:
                return col
    bc = getattr(self, "base_column", "") or ""
    if bc and any(bc.endswith(s) for s in CAUSAL_LAG_SUFFIXES):
        return bc
    return None


def _resolve_lag_values(self, X, base_raw: np.ndarray, n: int) -> np.ndarray | None:
    """1-D causal-lag values aligned to the predict rows, or None when no lag column resolves."""
    lag_col = _resolve_lag_column(self, X)
    if lag_col is None:
        return None
    if lag_col == getattr(self, "base_column", "") and base_raw.ndim == 1 and base_raw.size == n:
        return np.asarray(base_raw, dtype=np.float64)  # base column IS the lag; reuse the raw values.
    try:
        from . import _extract_base
        vals = np.asarray(_extract_base(X, lag_col), dtype=np.float64).reshape(-1)
    except Exception as exc:
        logger.debug("[soft_base_shrink] lag column %r extraction failed (%r); using fallback_predict.", lag_col, exc)
        return None
    return vals if vals.size == n else None


def _median_constant(self, params: dict[str, Any]) -> float:
    """Finite-median fallback constant used when no other prediction source is available."""
    from ._predict import _finite_median_fallback
    return _finite_median_fallback(params)


def apply_smart_fallback(
    self, y_hat: np.ndarray, deep_ood: np.ndarray, base_raw: np.ndarray,
    domain_ok: np.ndarray, X, params: dict[str, Any],
) -> None:
    """Overwrite deeply-OOD rows in ``y_hat`` (in place) with the smart fallback.

    Deep-OOD rows use the causal-lag value where finite; otherwise (no lag column, or a non-finite lag)
    they use the wrapper's ``fallback_predict`` (median constant, or NaN under ``fallback_predict='nan'``).
    Only rows that are ALSO domain-valid are touched -- domain violations are already handled upstream.
    """
    if deep_ood is None:
        return
    rows = deep_ood & np.asarray(domain_ok, dtype=bool)
    if not rows.any():
        return
    idx = np.where(rows)[0]
    lag_vals = _resolve_lag_values(self, X, base_raw, y_hat.size)
    use_median = self.fallback_predict == "y_train_median"
    if lag_vals is not None:
        v = lag_vals[idx]
        finite = np.isfinite(v)
        y_hat[idx[finite]] = v[finite]
        rest = idx[~finite]
        if rest.size:
            y_hat[rest] = _median_constant(self, params) if use_median else np.nan
    else:
        y_hat[idx] = _median_constant(self, params) if use_median else np.nan


def record_info(self, shrunk, deep_ood, n_rows: int) -> None:
    """Expose the per-row shrink / fallback flags on ``self.soft_shrink_info_`` (last predict batch)."""
    if shrunk is None:
        self.soft_shrink_info_ = {
            "shrunk_mask": None, "fallback_mask": None,
            "n_shrunk": 0, "n_fallback": 0, "n_rows": int(n_rows),
        }
        return
    self.soft_shrink_info_ = {
        "shrunk_mask": shrunk,
        "fallback_mask": deep_ood,
        "n_shrunk": int(shrunk.sum()),
        "n_fallback": int(deep_ood.sum()) if deep_ood is not None else 0,
        "n_rows": int(n_rows),
    }
