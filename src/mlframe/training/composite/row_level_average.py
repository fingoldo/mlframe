"""``compute_row_level_then_average_predictions``: skip child-table aggregation, average predictions instead.

Source: Home Credit Default Risk 3rd place -- "Models for bureau and prev application were built... without
aggregation by sk_id_curr, using target from main application (same target for all rows of one client)...
predicts were averaged by sk_id_curr after. This approach performed better than aggregating first." For
one-to-many panel/relational data (transactions-to-customer, statements-to-account), the standard pattern is
to hand-aggregate child rows into per-entity summary stats BEFORE modeling; this instead trains directly on
the raw child rows (broadcasting the parent entity's label to every child row) and averages the fitted
model's per-row predictions back to the entity level -- letting the model itself learn a nonlinear
"aggregation" rather than committing to hand-picked summary statistics upfront.

Leakage discipline: entity-broadcast labels mean every row of one entity shares the same target, so a
row-level K-fold CV that splits rows independently of entity would leak (some of an entity's rows in train,
others in val, both carrying the identical label). This reuses :func:`composite_oof_predictions`'s existing
``groups`` support (``GroupKFold``) to guarantee an entity's rows are NEVER split across folds.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import polars as pl

from .ensemble.feature_stacking import composite_oof_predictions

logger = logging.getLogger(__name__)


def compute_row_level_then_average_predictions(
    X_rows: Any,
    y_row_broadcast: np.ndarray,
    entity_ids: np.ndarray,
    model_factory: Callable[[], Any],
    X_query_rows: Optional[Any] = None,
    query_entity_ids: Optional[np.ndarray] = None,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    column_name: str = "row_level_avg_pred",
) -> pl.DataFrame:
    """Fit a model on raw (unaggregated) child rows with an entity-broadcast label, then average its
    per-row predictions back to one prediction per entity.

    Parameters
    ----------
    X_rows
        Child-row feature frame (pandas/polars), ``n_rows`` rows -- NOT pre-aggregated by entity.
    y_row_broadcast
        ``(n_rows,)`` target, the SAME value repeated for every row belonging to one entity (the parent
        label broadcast down to its children).
    entity_ids
        ``(n_rows,)`` parent-entity id per child row. Rows sharing an id are guaranteed to stay together in
        the same CV fold (see module docstring).
    model_factory
        Zero-arg callable returning a fresh unfitted row-level estimator.
    X_query_rows, query_entity_ids
        Mode B: a genuinely held-out child-row set (e.g. real test rows) with its own entity ids. When both
        are None (default), Mode A computes OOF entity-level predictions for ``X_rows``/``entity_ids``
        themselves.
    n_splits, random_state
        Passed to the underlying group-aware OOF CV (Mode A only; Mode B fits once on the full ``X_rows``).

    Returns
    -------
    pl.DataFrame
        Two columns: ``entity_id`` (one row per UNIQUE entity, in first-seen order over the query entity
        ids) and ``column_name`` (that entity's row-level predictions, averaged).
    """
    entity_arr = np.asarray(entity_ids)
    y_arr = np.asarray(y_row_broadcast, dtype=np.float64)

    if X_query_rows is not None:
        if query_entity_ids is None:
            raise ValueError("X_query_rows requires query_entity_ids.")
        model = model_factory()
        model.fit(X_rows, y_arr)
        row_pred = np.asarray(model.predict(X_query_rows), dtype=np.float64)
        target_entities = np.asarray(query_entity_ids)
    else:
        row_pred = composite_oof_predictions(model_factory, X_rows, y_arr, n_splits=n_splits, random_state=random_state, groups=entity_arr)
        target_entities = entity_arr

    df = pd.DataFrame({"entity_id": target_entities, "_pred": row_pred})
    entity_avg = df.groupby("entity_id", sort=False)["_pred"].mean()
    # First-seen order over target_entities (not groupby's own ordering, which sorts when sort=False still
    # follows pandas' internal hash-table insertion order -- pin explicitly for a deterministic contract).
    seen_order = pd.unique(target_entities)
    entity_avg = entity_avg.reindex(seen_order)

    return pl.DataFrame({"entity_id": entity_avg.index.to_numpy(), column_name: entity_avg.to_numpy(dtype=np.float64)})


__all__ = ["compute_row_level_then_average_predictions"]
