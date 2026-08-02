"""Shared row/domain-mask implementations for the ``training/composite/transforms/`` family.

Many per-transform ``*_domain`` functions (registered by name in ``registry.py`` and re-exported
from ``__init__.py``, so each must keep its own name/docstring/signature) share one of a handful of
literal finite-check bodies. Consolidated here so a fix to the shared logic can't silently drift
out of sync across copies; each per-transform module keeps a thin, differently-named wrapper that
delegates to the matching implementation below.
"""
from __future__ import annotations

import numpy as np


def residual_domain_reshaped(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    """Boolean row mask (flattened via ``reshape(-1)``) where ``base`` (and ``y``, if given) are finite."""
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64).reshape(-1))
    if y is None:
        return base_ok
    return np.asarray(base_ok & np.isfinite(np.asarray(y, dtype=np.float64).reshape(-1)))


def residual_domain_plain(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    """Boolean row mask where ``base`` (and ``y``, if given) are finite; no reshape, defined on all reals."""
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64))
    if y is None:
        return base_ok
    return np.asarray(base_ok & np.isfinite(np.asarray(y, dtype=np.float64)))


def residual_domain_no_cast(y: np.ndarray | None, base: np.ndarray) -> np.ndarray:
    """Boolean row mask where ``base`` (and ``y``, if given) are finite; no dtype coercion, no positivity constraint."""
    base_ok = np.isfinite(base)
    if y is None:
        return np.asarray(base_ok)
    return np.asarray(base_ok & np.isfinite(y))


def y_domain_finite(y: np.ndarray) -> np.ndarray:
    """Domain mask for a transform defined on all finite ``y`` (no ``base`` dependence)."""
    return np.isfinite(np.asarray(y, dtype=np.float64))
