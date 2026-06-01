"""Semi-supervised feature-engineering wrapper for MRMR (Layer 80, 2026-06-01).

Production setup: callers often have a small labeled pool (e.g. 1k cold-start
rows) alongside a much larger unlabeled pool (~100k rows from the same
distribution). The current orth-poly FE family (Layers 21 / 22 / 56 / 77 / 78
+ adaptive-degree / routing / cluster / diff-basis variants) fits its
per-column basis preprocess (z-score for hermite, min-max for legendre /
chebyshev, shift for laguerre) ON THE LABELED X ONLY. With ~200-1000 labeled
rows the per-column mean / std / lo / hi estimates carry meaningful sampling
noise; downstream the basis transform `He_n(z)` / `T_n(z)` / `L_n(z)` /
`L^Lag_n(z)` inherits that noise into every emitted column.

This module ships two pieces:

1.  ``set_unlabeled_pool`` / ``get_unlabeled_pool`` / ``unlabeled_pool_active``
    -- a thread-local registry mapping column name -> 1D unlabeled value
    array. ``generate_univariate_basis_features`` (and the auto-routing branch
    inside it) consults the registry per column to decide whether to fit
    the basis preprocess on the LABELED-ONLY array or on the concatenated
    LABELED + UNLABELED pool.

2.  ``fit_with_unlabeled(mrmr, X_labeled, y_labeled, X_unlabeled, **fit_params)``
    -- thin wrapper around ``MRMR.fit`` that:

    *   verifies ``mrmr.fe_semi_supervised_enable`` is True (else falls
        back to vanilla ``fit`` and emits a UserWarning),
    *   builds the per-column unlabeled mapping from the columns shared
        between ``X_labeled`` and ``X_unlabeled``,
    *   installs the thread-local pool via the context manager,
    *   delegates to ``mrmr.fit`` so the L21 / L22 / L56 / L77 / L78 stages
        all consume the augmented pool.

y is NEVER read by the augmentation code path: MI scoring still consumes
``y_labeled`` only via the normal MRMR fit pipeline. The augmentation is
strictly preprocess-level (mean / std / lo / hi).

Default-disabled (``fe_semi_supervised_enable=False``) preserves byte-equal
behaviour: when the flag is off, the wrapper helper still works but acts as
a no-op delegator to ``mrmr.fit(X_labeled, y_labeled)`` -- X_unlabeled is
ignored. When the flag is on but no X_unlabeled is supplied, the legacy
fit-on-labeled-only path runs (the thread-local pool stays empty).
"""
from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping, Optional

import numpy as np
import pandas as pd

__all__ = [
    "set_unlabeled_pool",
    "get_unlabeled_pool",
    "unlabeled_pool_active",
    "fit_with_unlabeled",
]


# Thread-local registry. We use a threading.local() rather than a global dict
# so concurrent MRMR fits in a joblib / threading worker pool cannot
# cross-pollute each other's unlabeled pools.
_LOCAL = threading.local()


def get_unlabeled_pool() -> Optional[Dict[str, np.ndarray]]:
    """Return the currently-active unlabeled pool mapping, or None.

    Called by ``generate_univariate_basis_features`` (and any other basis-fit
    consumer) per column to decide whether to augment the preprocess pool.
    Returns ``None`` -- the legacy bit-exact path -- when no pool is
    installed in the current thread.
    """
    return getattr(_LOCAL, "pool", None)


def set_unlabeled_pool(pool: Optional[Mapping[str, np.ndarray]]) -> None:
    """Install (or clear) the thread-local unlabeled pool.

    Prefer ``with unlabeled_pool_active(pool): ...`` for stack-safe
    install / restore semantics. This direct setter exists for tests and
    advanced callers who already wrap the install in their own try/finally.
    """
    if pool is None:
        if hasattr(_LOCAL, "pool"):
            del _LOCAL.pool
        return
    # Normalise to a plain dict of float64 ndarrays so consumers can rely on
    # ``np.concatenate`` working without per-call asarray coercion.
    norm: Dict[str, np.ndarray] = {}
    for k, v in pool.items():
        arr = np.asarray(v, dtype=np.float64).ravel()
        norm[str(k)] = arr
    _LOCAL.pool = norm


@contextmanager
def unlabeled_pool_active(
    pool: Optional[Mapping[str, np.ndarray]],
) -> Iterator[None]:
    """Context manager: install ``pool`` on entry, restore prior pool on exit.

    Always restores the previous pool (None or whatever was installed
    before), even when the body raises.
    """
    prior = getattr(_LOCAL, "pool", None)
    set_unlabeled_pool(pool)
    try:
        yield
    finally:
        set_unlabeled_pool(prior)


def _build_pool_mapping(
    X_labeled: pd.DataFrame,
    X_unlabeled: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Build the per-column unlabeled mapping from the columns shared between
    the two frames. Numeric-only: non-numeric columns are silently skipped,
    matching ``generate_univariate_basis_features`` semantics.
    """
    if not isinstance(X_labeled, pd.DataFrame) or not isinstance(
        X_unlabeled, pd.DataFrame
    ):
        raise TypeError(
            "fit_with_unlabeled: both X_labeled and X_unlabeled must be "
            "pandas DataFrames; got "
            f"{type(X_labeled).__name__} / {type(X_unlabeled).__name__}."
        )
    mapping: Dict[str, np.ndarray] = {}
    for col in X_labeled.columns:
        if col not in X_unlabeled.columns:
            continue
        if not pd.api.types.is_numeric_dtype(X_labeled[col]):
            continue
        if not pd.api.types.is_numeric_dtype(X_unlabeled[col]):
            continue
        arr = np.asarray(X_unlabeled[col].to_numpy(), dtype=np.float64).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            # All-NaN column in the unlabeled pool: nothing to augment, skip.
            continue
        mapping[str(col)] = finite
    return mapping


def fit_with_unlabeled(
    mrmr: Any,
    X_labeled: pd.DataFrame,
    y_labeled: Any,
    X_unlabeled: Optional[pd.DataFrame] = None,
    **fit_params: Any,
):
    """Fit an ``MRMR`` instance with optional unlabeled-pool augmentation
    for the orth-poly basis-preprocess fitting (Layer 80, 2026-06-01).

    Parameters
    ----------
    mrmr : MRMR
        Estimator to fit. Should be constructed with
        ``fe_semi_supervised_enable=True``; if False, the unlabeled pool is
        ignored (legacy fit-on-labeled-only path runs) and a UserWarning
        notifies the caller that the flag must be enabled to take effect.
    X_labeled : pd.DataFrame
        Labeled training rows. y_labeled is aligned to these rows.
    y_labeled : array-like
        Labels for X_labeled. Forwarded to ``mrmr.fit`` unchanged.
    X_unlabeled : pd.DataFrame, optional
        Auxiliary unlabeled rows from the same distribution. Used ONLY for
        fitting per-column basis preprocess params (mean / std / lo / hi);
        no engineered values are emitted for these rows and y is never
        consulted, so leakage is impossible.
    **fit_params :
        Forwarded to ``mrmr.fit`` as-is.

    Returns
    -------
    mrmr
        The fitted estimator (same instance, for chain-style usage).
    """
    enabled = bool(getattr(mrmr, "fe_semi_supervised_enable", False))
    if not enabled:
        if X_unlabeled is not None:
            warnings.warn(
                "fit_with_unlabeled: X_unlabeled was provided but "
                "fe_semi_supervised_enable=False on the MRMR instance; "
                "the unlabeled pool will be IGNORED. Pass "
                "fe_semi_supervised_enable=True to the MRMR ctor to activate "
                "semi-supervised basis-preprocess fitting.",
                UserWarning,
                stacklevel=2,
            )
        return mrmr.fit(X_labeled, y_labeled, **fit_params)
    if X_unlabeled is None or (
        hasattr(X_unlabeled, "shape") and X_unlabeled.shape[0] == 0
    ):
        # Master switch ON but caller passed no unlabeled rows -- run the
        # legacy fit. We do NOT install an empty pool because installing it
        # would still flip the moment-router auto-routing on/off branch.
        return mrmr.fit(X_labeled, y_labeled, **fit_params)
    pool = _build_pool_mapping(X_labeled, X_unlabeled)
    if not pool:
        # Shared-column intersection was empty (every column was non-numeric
        # or all-NaN in the unlabeled frame). Nothing to augment.
        return mrmr.fit(X_labeled, y_labeled, **fit_params)
    with unlabeled_pool_active(pool):
        return mrmr.fit(X_labeled, y_labeled, **fit_params)
