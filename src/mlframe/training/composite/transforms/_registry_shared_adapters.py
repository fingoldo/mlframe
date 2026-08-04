"""Adapters shared between ``registry.py`` and ``_registry_extended.py``.

Both siblings previously imported top-level from each other (``registry`` -> ``_registry_extended``
for ``_TRANSFORMS_REGISTRY_EXTENDED``, ``_registry_extended`` -> ``registry`` for
``_make_unary_registry_adapter`` / ``_centered_ratio_domain_fitted``), forming a 2-node import cycle
that only resolved because ``registry``'s late import of ``_registry_extended`` executes after these
two names are already bound. Extracted here (a leaf with no dependency on either sibling) so both can
import from a common module instead of each other.
"""
from __future__ import annotations

import numpy as np

from .extended import _centered_ratio_domain


def _make_unary_registry_adapter(
    fit_fn, forward_fn, inverse_fn, domain_fn, domain_fitted_fn=None,
):
    """Adapt a unary (y, params) signature to the registry's (y, base, params) signature by ignoring ``base``. Returns (fit_adapter, forward_adapter, inverse_adapter, domain_adapter[, domain_fitted_adapter]).

    ``domain_fitted_fn`` (optional, signature ``(y, params) -> mask``) wires
    the fitted-params-aware domain hook for unary transforms whose validity
    depends on a learned parameter (e.g. ``log_y``'s ``offset``: rows with
    ``y + offset <= 0`` are out of domain only once ``offset`` is known). When
    ``None`` the returned 5th element is ``None`` and the registry entry leaves
    ``domain_check_fitted`` unset (params-free ``domain_check`` is exact)."""

    def _fit(y, base):
        """Registry-shaped fit adapter: drop ``base``, delegate to the unary ``fit_fn(y)``."""
        return fit_fn(y)

    def _forward(y, base, params):
        """Registry-shaped forward adapter: drop ``base``, delegate to the unary ``forward_fn(y, params)``."""
        return forward_fn(y, params)

    def _inverse(t_hat, base, params):
        """Registry-shaped inverse adapter: drop ``base``, delegate to the unary ``inverse_fn(t_hat, params)``."""
        return inverse_fn(t_hat, params)

    def _domain(y, base):
        """Registry-shaped domain adapter: params-free unary domain at fit time, all-True at predict time (no base constraint for unary transforms)."""
        # The unary helper accepts (y) or (y, params); the registry
        # contract is domain_check(y, base) at fit-time and (None, base)
        # at predict-time. Predict-side call passes y=None so we cannot
        # apply the unary domain on y -- gate on finite base / always-True
        # for unary which has no base constraint at predict.
        if y is None:
            return np.ones(len(base) if hasattr(base, "__len__") else 1, dtype=bool)
        return domain_fn(y)

    if domain_fitted_fn is None:
        return _fit, _forward, _inverse, _domain, None

    def _domain_fitted(y, base, params):
        """Registry-shaped fitted-domain adapter: params-aware unary domain at fit time, all-True at predict time (see docstring below)."""
        # Fitted-domain for unary: no base constraint, so at predict time
        # (y is None) the per-row domain cannot be re-checked from base
        # alone (e.g. log_y's ``y + offset > 0`` needs y). Return all-True
        # for the predict-side row count, matching ``_domain``. At fit/
        # screening time y is present and we gate on the params-aware
        # unary domain (e.g. ``y + offset > 0``).
        if y is None:
            return np.ones(len(base) if hasattr(base, "__len__") else 1, dtype=bool)
        return domain_fitted_fn(y, params)

    return _fit, _forward, _inverse, _domain, _domain_fitted


def _centered_ratio_domain_fitted(y, base, params):
    """Fitted-domain for ``centered_ratio`` (T = y / (base + c)).

    The pre-fit ``_centered_ratio_domain`` only gates on finite y / base; the
    real per-row validity depends on the learned shift ``c`` and eps-floor:
    a row whose ``base + c`` lands inside the near-zero ``[-eps, eps]`` band
    has its denominator clamped to ``+/- eps`` in ``forward``/``inverse``, so
    T no longer reflects the true ratio and the round-trip is only approximate
    on that row. Those rows are excluded from screening + fit so the divisor
    clamp never silently distorts the MI estimate / fitted scale. Mirrors the
    ``domain_check`` ``y=None`` predict-time contract: with ``y`` unknown we
    still gate the base-side ``|base + c| >= eps`` condition (knowable from
    params), so the same rows are flagged at predict time.
    """
    base_arr = np.asarray(base, dtype=np.float64)
    if params is None:
        # No fitted params yet -> fall back to the params-free domain.
        return _centered_ratio_domain(y, base)
    c = float(params.get("c", 0.0))
    eps = float(params.get("eps", 0.0))
    shifted = base_arr + c
    base_ok = np.isfinite(base_arr) & (np.abs(shifted) >= eps)
    if y is None:
        return base_ok
    return base_ok & np.isfinite(np.asarray(y, dtype=np.float64))
