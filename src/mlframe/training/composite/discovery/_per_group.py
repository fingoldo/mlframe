"""Opt-in per-group/per-cluster composite-target discovery.

Reopens the REJECTED design decision documented in ``discovery/__init__.py`` (near
``CompositeTargetDiscovery.fit``): "10-15 values per cluster too few for stable
per-cluster discovery ... revisit ONLY when production data shows 500+ rows per
cluster on average". This module implements that reopened path, gated behind
``config.per_group_discovery_enabled`` (default False -- the global-only path is
untouched when this flag is off).

Delegation, not duplication: each qualifying group's discovery is run by handing a
FRESH ``CompositeTargetDiscovery`` instance (same config, with
``per_group_discovery_enabled`` forced off to prevent recursion) the group's OWN
``train_idx`` subset and calling the real ``fit()`` -- the exact same MI-screening +
Phase-B rerank + honest-RMSE-gate pipeline every other caller uses, INCLUDING that
per-group ``fit()``'s own honest-holdout carve, which is done from the group's rows
ONLY (leakage-safety: no group's screening or holdout ever sees another group's rows).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from . import CompositeTargetDiscovery

from ..estimator import _extract_groups
from ..spec import CompositeSpec
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)



def _group_val_rows(val_df: Any, val_y: Any, group_col: str, group_val: Any) -> tuple[Any, Any]:
    """The val rows of one group, so its specs are gated on the population they will be routed to.

    Returns ``(None, None)`` when the val frame is absent or lacks ``group_col``, or when the group has no val rows: the
    delegate's y-scale gate then falls back to a group-disjoint carve of its own training rows.
    """
    if val_df is None or val_y is None:
        return None, None
    try:
        g = _extract_groups(val_df, group_col)
    except Exception as exc:  # nosec B110 -- a val frame without the group column cannot be split per group; logged
        logger.info("[CompositeTargetDiscovery.per_group] val frame has no %r column (%s); group gates use their train rows.", group_col, exc)
        return None, None
    mask = np.asarray(g) == group_val
    if not mask.any():
        return None, None
    from ..ensemble._oof_split import _slice_rows

    return _slice_rows(val_df, mask), np.asarray(val_y)[mask]

def run_per_group_discovery(
    self: "CompositeTargetDiscovery",
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    test_idx: np.ndarray | None,
    time_ordering: Any,
    val_df: Any,
    val_y: np.ndarray | None,
) -> dict[Any, list[CompositeSpec]]:
    """Run discovery independently per group of ``config.per_group_column``.

    ``train_idx`` here is the ORIGINAL (pre-honest-holdout-carve) row set passed to
    the outer ``fit()`` -- each per-group delegate call carves its OWN honest holdout
    from its own subset, never sharing rows with any other group's carve or with the
    outer/global fit's carve.

    Groups with >= ``config.per_group_min_rows`` rows get their own discovered spec
    list; groups below the floor are simply absent from the returned mapping (the
    caller/predict-time router falls back to the global ``specs_`` for those).
    """
    config = self.config
    group_col = config.per_group_column
    min_rows = int(config.per_group_min_rows)
    if not group_col:
        logger.warning(
            "[CompositeTargetDiscovery.per_group] per_group_discovery_enabled=True but " "per_group_column is not set; skipping per-group discovery."
        )
        return {}

    group_values_full = _extract_groups(df, group_col)
    group_values_train = group_values_full[train_idx]

    specs_by_group: dict[Any, list[CompositeSpec]] = {}
    unique_groups = np.unique(group_values_train)
    for group_val in unique_groups:
        group_mask = group_values_train == group_val
        group_row_idx = train_idx[group_mask]
        n_rows = int(group_row_idx.size)
        if n_rows < min_rows:
            logger.info(
                "[CompositeTargetDiscovery.per_group] group=%r has %d rows < per_group_min_rows=%d; "
                "falling back to the global spec set.",
                group_val, n_rows, min_rows,
            )
            continue

        from . import CompositeTargetDiscovery as _CompositeTargetDiscoveryCls

        _group_config = config.model_copy(update={"per_group_discovery_enabled": False})
        _group_discovery = _CompositeTargetDiscoveryCls(_group_config)
        # The delegate is a fresh instance: hand it the rerank grouping and the hint strengths the caller set on this one,
        # or its rerank, group-disjoint holdout and fragility gate run group-blind inside the group.
        for _attr in ("_group_ids_for_rerank", "_hint_strengths_pct"):
            if getattr(self, _attr, None) is not None:
                setattr(_group_discovery, _attr, getattr(self, _attr))
        _g_val_df, _g_val_y = _group_val_rows(val_df, val_y, group_col, group_val)
        try:
            _group_discovery.fit(
                df,
                target_col,
                feature_cols,
                group_row_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                time_ordering=time_ordering,
                val_df=_g_val_df,
                val_y=_g_val_y,
            )
        except Exception as exc:
            log_throttle(
                logger,
                "discovery_per_group_fit_failed",
                logging.WARNING,
                "[CompositeTargetDiscovery.per_group] discovery failed for group=%r (%d rows): %s. "
                "Falling back to the global spec set for this group.",
                group_val, n_rows, exc,
            )
            continue
        specs_by_group[group_val] = list(getattr(_group_discovery, "specs_", []) or [])
        logger.info(
            "[CompositeTargetDiscovery.per_group] group=%r (%d rows) discovered %d spec(s).",
            group_val, n_rows, len(specs_by_group[group_val]),
        )

    return specs_by_group


__all__ = ["run_per_group_discovery"]
