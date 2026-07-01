"""Extreme-AR composite-discovery skip helpers, carved out of ``_phase_composite_discovery.py`` to keep it under the
1000-LOC house limit. Pure helpers (zoo classification, the skip predicate, a per-group lag-1 recompute fallback);
re-imported by the phase module and used unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..models import is_neural_model

# Plain (unregularised) linear models extrapolate unboundedly on unseen groups like neural nets do; Ridge is
# regularised and was the prod-safe baseline (R^2=1.00 where Identity-MLP collapsed to -326), so it is NOT unbounded.
_UNBOUNDED_LINEAR_TOKENS = ("linear", "ols", "lstsq")

# When the zoo is bounded-only and the user left ``composite_skip_when_raw_dominates_ratio`` at 0 (never skip), default
# it to this conservative ratio: fires only when the raw-y screen RMSE is < 2% of y_std (raw R^2 > 0.9996), i.e. the
# raw features already predict y near-perfectly so a composite has no headroom. This is the documented prior default,
# safe to re-enable now that the unbounded-extrapolation case is excluded by the bounded-zoo gate.
_RERANK_SKIP_RATIO_BOUNDED_DEFAULT = 0.02


def _zoo_is_bounded_only(models) -> bool:
    """True iff no model in the zoo can extrapolate unboundedly on unseen groups (no neural / plain-linear estimator).

    When the zoo is bounded-only (tree ensembles, Ridge), an extreme-AR group-aware target can safely SKIP composite
    discovery: the AR failsafe carries the signal and no model needs a residualised target to avoid catastrophic
    extrapolation. An unknown / empty zoo returns False -- conservative, never auto-skip when we cannot prove safety.
    """
    names = [str(m).lower() for m in (models or [])]
    if not names:
        return False
    for nm in names:
        if is_neural_model(nm):
            return False
        if "ridge" not in nm and any(tok in nm for tok in _UNBOUNDED_LINEAR_TOKENS):
            return False
    return True


def _extreme_ar_discovery_skip(
    *, skip_enabled: bool, group_aware_active: bool, bounded_only_zoo: bool,
    lag1_ar, is_picked_target: bool, threshold: float,
) -> bool:
    """Decide whether to SKIP the whole composite-discovery block for an extreme-AR group-aware target.

    Fires only when ALL hold: the skip is enabled; the REAL split is group-aware (``group_aware_active`` -- the analyzer
    hint OR the configured splitter, not the hint alone -- a config-only group-aware run must skip too); the zoo is
    bounded-only (no model needs a residual target to avoid extrapolation); a per-group lag-1 autocorr is available and
    ``>= threshold``; and this is the analyzer's picked target. On a strongly-AR group-aware target the residual signal
    is near-zero on unseen groups and lag_predict already carries it, so discovery is wasted compute.
    """
    return bool(
        skip_enabled
        and group_aware_active
        and bounded_only_zoo
        and lag1_ar is not None
        and is_picked_target
        and float(lag1_ar) >= threshold
    )


def _recompute_lag1_ar_per_group(y_full, group_ids, train_idx) -> Optional[float]:
    """Per-group lag-1 autocorrelation of the TARGET on the train rows, reusing the analyzer's Fisher-z kernel.

    Belt-and-suspenders fallback for the extreme-AR skip: ``metadata["target_distribution_report"]`` carries
    ``lag1_autocorr_per_group`` only on independently-gated analyzer branches, so it can be absent at discovery time
    even on a 0.9999-AR target (the skip then silently never fires). The target's own per-group AR(1) IS the skip
    signal, and discovery already holds ``y_full`` (full-data aligned) + ``group_ids`` + ``train_idx``, so recompute it
    directly. Returns None on any size/finite/degenerate issue (caller treats None as "no AR signal").
    """
    try:
        from mlframe.training.targets import _lag1_autocorr_grouped
        _y = np.asarray(y_full)
        _g = np.asarray(group_ids).reshape(-1)
        _ti = np.asarray(train_idx)
        if _ti.size == 0:
            return None
        _max = int(_ti.max())
        if _y.shape[0] <= _max or _g.shape[0] <= _max:
            return None
        yt = _y[_ti].astype(np.float64)
        gt = _g[_ti]
        finite = np.isfinite(yt)
        if int(finite.sum()) < 100:
            return None
        ar = float(_lag1_autocorr_grouped(yt[finite], gt[finite]))
        return ar if np.isfinite(ar) else None
    except Exception:  # noqa: BLE001 -- a recompute failure must never abort discovery
        return None
