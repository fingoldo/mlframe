"""Temporal-aware helpers carved out of ``_fit.fit`` (keeps the discovery fit body under the LOC limit).

Two cohesive blocks: the pre-discovery base-target leakage guard (drops same-time near-identity re-encodings of
the target that the forbidden-pattern regex misses) and the screening-sample time ordering (sorts the MI-screening
sample into time order so the downstream tiny-model CV is a genuine forward-walk).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .screening import _extract_column_array

logger = logging.getLogger(__name__)


def apply_base_leakage_guard(
    discovery: Any,
    df: Any,
    base_candidates: list,
    train_idx: np.ndarray,
    y_train: np.ndarray,
    time_ordering: Any,
) -> list:
    """Drop base candidates that are a same-time near-identity re-encoding of ``y`` (leakage).

    Only acts when ``time_ordering`` is given so the lag-probe spares a genuine ``lag(y)`` base; a no-op on
    non-temporal data (autocorrelation must not be mistaken for a leak). Sets ``discovery._leaky_bases_dropped_``
    and returns the kept base candidates (unchanged when nothing is dropped).
    """
    from ._leakage import detect_base_target_leakage

    _to_all = np.asarray(time_ordering)
    _to_train = _to_all[train_idx] if _to_all.shape[0] >= int(np.max(train_idx)) + 1 else None
    if _to_train is None:
        return base_candidates

    kept, dropped = [], []
    for _bcand in base_candidates:
        try:
            _barr = _extract_column_array(df, _bcand)[train_idx]
            _leak = detect_base_target_leakage(y_train, _barr, time_ordering=_to_train)
        except Exception:
            kept.append(_bcand)
            continue
        if _leak.get("is_leaky"):
            dropped.append((_bcand, _leak.get("reason", "")))
        else:
            kept.append(_bcand)
    if dropped:
        discovery._leaky_bases_dropped_ = dropped
        logger.warning(
            "[CompositeTargetDiscovery] dropped %d leaky base(s) (same-time near-identity of y): %s",
            len(dropped),
            dropped[:5],
        )
        return kept
    return base_candidates


def order_screen_by_time(train_idx_screen: np.ndarray, sample_idx: np.ndarray, time_ordering: Any) -> tuple:
    """Sort the MI-screening sample into time order so downstream tiny-model CV is a forward-walk.

    Returns ``(train_idx_screen, sample_idx, screen_time_ordered)``; the first two are reordered (and identical to
    the inputs when ``time_ordering`` is absent / unusable), and ``screen_time_ordered`` reports whether the sort ran.
    """
    if time_ordering is None:
        return train_idx_screen, sample_idx, False
    try:
        _time_all = np.asarray(time_ordering)
        if _time_all.shape[0] >= int(np.max(train_idx_screen)) + 1:
            _time_screen = _time_all[train_idx_screen]
            # NaT/NaN-safe stable sort: non-finite/unsortable keys keep their original relative order at the front.
            _order = np.argsort(_time_screen, kind="stable")
            return train_idx_screen[_order], sample_idx[_order], True
    except (TypeError, ValueError, IndexError) as _to_err:
        logger.warning(
            "[CompositeTargetDiscovery] time_ordering supplied but could not order the screening sample (%s); "
            "falling back to base-monotonicity time detection.",
            _to_err,
        )
    return train_idx_screen, sample_idx, False
