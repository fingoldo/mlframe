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

logger = logging.getLogger(__name__)


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
        try:
            _group_discovery.fit(
                df,
                target_col,
                feature_cols,
                group_row_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                time_ordering=time_ordering,
                val_df=val_df,
                val_y=val_y,
            )
        except Exception as exc:
            logger.warning(
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
