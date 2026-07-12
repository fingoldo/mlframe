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

Also covers the Home Credit Default Risk 5th place's "per-sub-table nested row-level model -> multi-stat OOF
aggregation" pattern: propagate an entity's target down to its child/transaction rows (one sub-table at a
time -- installments, bureau, pos_cash, ...), train a CV'd row-level model, and aggregate the OOF row scores
(min/max/mean/std/median) back to entity level as new features. That thread's own reported failure mode --
"AUC boost on CV but not on the leaderboard" when child rows sharing a key got split across folds -- is
exactly the entity-level ``GroupKFold`` leakage discipline documented above; this function already enforces
it structurally (a leak of that kind is not possible to construct through this API), so no separate
``NestedRowLevelOOFEncoder`` class is needed -- ``agg_stats`` below is the one genuinely missing piece
(the original single-stat mean-only signature) that pattern needs.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from .ensemble.feature_stacking import composite_oof_predictions
from .row_level_average_importance import compute_row_level_feature_importance_oof, compute_row_level_feature_importance_single_model

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
    agg_stats: Optional[Sequence[str]] = None,
    flag_low_confidence_quantile: Optional[float] = None,
    return_row_feature_importance: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
    """Fit a model on raw (unaggregated) child rows with an entity-broadcast label, then aggregate its
    per-row predictions back to one row per entity.

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
    agg_stats
        None (default) preserves the original single-column mean-only contract (``column_name``). Pass e.g.
        ``("mean", "min", "max", "std", "median")`` for the Home Credit 5th place's multi-stat nested
        row-level pattern -- returns one column per stat, named ``{column_name}_{stat}``.
    flag_low_confidence_quantile
        None (default) adds nothing -- bit-identical to the original contract. When set to a quantile in
        ``(0, 1)`` (e.g. ``0.75``), an entity's row-level predictions are dispersed (agreement signal): rows
        of one entity that predict very differently from each other mean the group-level average is less
        trustworthy than one where all rows agree, EVEN THOUGH both entities may average to the same mean
        prediction. Adds a boolean ``{column_name}_low_confidence`` column flagging entities whose
        within-group prediction std is above the given quantile of all entities' stds in this call (the top
        ``1 - flag_low_confidence_quantile`` fraction by disagreement). Independent of ``agg_stats`` -- the
        std used for flagging is always computed internally and is not affected by whether ``"std"`` is
        also requested via ``agg_stats``.
    return_row_feature_importance
        ``False`` (default) -- bit-identical to the original contract, returns a single ``pl.DataFrame``. When
        ``True``, additionally fits/extracts the row-level model's feature importance (``feature_importances_``
        or ``abs(coef_)``) and returns ``(entity_df, importance_df)``, surfacing WHICH child-row features drove
        the row-level predictions before they were aggregated -- the aggregate OOF score alone can't tell a
        caller that. Mode B (external query) reuses the single model already fit for prediction at zero extra
        cost; Mode A (OOF) reruns the identical ``GroupKFold`` split used internally to average per-fold
        importances, which duplicates the row-level fit cost -- opt in only when the diagnostic is needed.
        Raises ``AttributeError`` if ``model_factory()`` produces a model exposing neither attribute.

    Returns
    -------
    pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]
        ``entity_id`` (one row per UNIQUE entity, in first-seen order over the query entity ids) plus either
        ``column_name`` (``agg_stats=None``) or one ``{column_name}_{stat}`` column per requested stat, plus
        ``{column_name}_low_confidence`` when ``flag_low_confidence_quantile`` is set. When
        ``return_row_feature_importance=True``, a second ``pl.DataFrame`` with ``feature``/``importance``
        columns (sorted descending) is returned alongside it as a 2-tuple.
    """
    entity_arr = np.asarray(entity_ids)
    y_arr = np.asarray(y_row_broadcast, dtype=np.float64)

    importance_df: Optional[pl.DataFrame] = None
    if X_query_rows is not None:
        if query_entity_ids is None:
            raise ValueError("X_query_rows requires query_entity_ids.")
        model = model_factory()
        model.fit(X_rows, y_arr)
        row_pred = np.asarray(model.predict(X_query_rows), dtype=np.float64)
        target_entities = np.asarray(query_entity_ids)
        if return_row_feature_importance:
            importance_df = compute_row_level_feature_importance_single_model(model, X_rows)
    else:
        row_pred = composite_oof_predictions(model_factory, X_rows, y_arr, n_splits=n_splits, random_state=random_state, groups=entity_arr)
        target_entities = entity_arr
        if return_row_feature_importance:
            importance_df = compute_row_level_feature_importance_oof(model_factory, X_rows, y_arr, entity_arr, n_splits)

    df = pd.DataFrame({"entity_id": target_entities, "_pred": row_pred})
    # First-seen order over target_entities (not groupby's own ordering, which sorts when sort=False still
    # follows pandas' internal hash-table insertion order -- pin explicitly for a deterministic contract).
    seen_order = pd.unique(target_entities)

    if agg_stats is None:
        entity_avg = df.groupby("entity_id", sort=False)["_pred"].mean().reindex(seen_order)
        out_cols: dict[str, np.ndarray] = {"entity_id": entity_avg.index.to_numpy(), column_name: entity_avg.to_numpy(dtype=np.float64)}
    else:
        entity_agg = df.groupby("entity_id", sort=False)["_pred"].agg(list(agg_stats)).reindex(seen_order)
        out_cols = {"entity_id": entity_agg.index.to_numpy()}
        for stat in agg_stats:
            out_cols[f"{column_name}_{stat}"] = entity_agg[stat].to_numpy(dtype=np.float64)

    if flag_low_confidence_quantile is not None:
        # ddof=0: a single-row entity has zero within-group spread (defined agreement, not undefined NaN).
        entity_std = df.groupby("entity_id", sort=False)["_pred"].std(ddof=0).reindex(seen_order).to_numpy(dtype=np.float64)
        threshold = float(np.quantile(entity_std, flag_low_confidence_quantile))
        out_cols[f"{column_name}_low_confidence"] = entity_std > threshold

    entity_df = pl.DataFrame(out_cols)
    if return_row_feature_importance:
        assert importance_df is not None
        return entity_df, importance_df
    return entity_df


__all__ = ["compute_row_level_then_average_predictions"]
