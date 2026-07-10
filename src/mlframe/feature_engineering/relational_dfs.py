"""Cutoff-time-aware relational feature synthesis (Featuretools-DFS-inspired).

Source: bestpractice_automated-feature-engineering-featuretools.md -- "DFS can automatically calculate
features for each training example at the specific time associated with the example via 'cutoff times,'
which is critical for avoiding target leakage in relational/temporal datasets."

Builds on ``as_of_aggregate.leakage_safe_aggregate`` (the existing single-table, per-row-cutoff aggregation
primitive) as the leakage-safety building block, adding the genuinely-missing piece: MULTI-TABLE join +
aggregation across a parent/child relationship, and STACKING (depth-2: aggregating an already-aggregated
child-of-child feature up one more relationship level -- e.g. Featuretools' own "outlet.SUM(bigmart's
grandchild.MEAN(...))" pattern).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from mlframe.feature_engineering.as_of_aggregate import leakage_safe_aggregate


@dataclass
class ChildTableSpec:
    """One parent<-child relationship to aggregate over.

    Attributes
    ----------
    child_df
        The child table -- one row per child event/observation.
    foreign_key_col
        Column in ``child_df`` identifying which parent row it belongs to (matches the parent's id column).
    time_col
        Column in ``child_df`` used for the cutoff comparison (row must be strictly before the parent's
        cutoff to be eligible).
    value_cols
        ``{child_column: [agg_name, ...]}`` -- same shape as ``leakage_safe_aggregate``'s ``agg_funcs``.
    prefix
        Prepended to every generated feature column name (keeps multi-child feature sets disambiguated).
    """

    child_df: pd.DataFrame
    foreign_key_col: str
    time_col: str
    value_cols: Mapping[str, Sequence[str]]
    prefix: str = field(default="")


def compute_relational_features(
    parent_df: pd.DataFrame,
    parent_id_col: str,
    cutoff_col: str,
    child_specs: Sequence[ChildTableSpec],
) -> pd.DataFrame:
    """Aggregate one or more child tables onto ``parent_df``, respecting each parent row's own cutoff time.

    Parameters
    ----------
    parent_df
        One row per prediction target; must contain ``parent_id_col`` and ``cutoff_col``.
    parent_id_col
        Parent's identifying key -- matched against each child spec's ``foreign_key_col``.
    cutoff_col
        Per-parent-row cutoff timestamp; only child rows strictly before this are aggregated (leakage-safe
        by construction -- delegates the actual filtering to ``leakage_safe_aggregate``).
    child_specs
        One ``ChildTableSpec`` per related child table; results from every spec are joined onto
        ``parent_df`` by ``parent_id_col``.

    Returns
    -------
    pd.DataFrame
        ``parent_df`` (all original columns preserved) plus one new column per ``(prefix, agg_column,
        agg_name)`` combination from every child spec.
    """
    as_of = parent_df[[parent_id_col, cutoff_col]].rename(columns={parent_id_col: "__entity__", cutoff_col: "__as_of__"})

    result = parent_df.copy()
    for spec in child_specs:
        agg = leakage_safe_aggregate(
            history_df=spec.child_df.rename(columns={spec.foreign_key_col: "__entity__"}),
            entity_col="__entity__",
            time_col=spec.time_col,
            as_of=as_of,
            agg_funcs=dict(spec.value_cols),
            query_entity_col="__as_of__",
        )
        agg = agg.drop(columns="__entity__")
        agg.columns = [f"{spec.prefix}_{c}" if spec.prefix else c for c in agg.columns]
        agg.index = result.index
        result = pd.concat([result, agg], axis=1)

    return result


def stack_relational_features(
    parent_df: pd.DataFrame,
    parent_id_col: str,
    cutoff_col: str,
    child_df: pd.DataFrame,
    child_id_col: str,
    child_time_col: str,
    child_foreign_key_col: str,
    grandchild_specs: Sequence[ChildTableSpec],
    child_value_cols: Dict[str, Sequence[str]],
    prefix: str = "l2",
) -> pd.DataFrame:
    """Depth-2 stacking: aggregate grandchild table(s) onto ``child_df`` (using each child row's OWN
    timestamp as its cutoff), then aggregate the now-enriched ``child_df`` onto ``parent_df`` -- the DFS
    "aggregate of an aggregate" pattern, staying leakage-safe at BOTH hops (a child row can only see
    grandchild rows before itself, and a parent row can only see child rows -- and their derived
    grandchild-aggregate features -- before its own cutoff).

    Parameters
    ----------
    child_df, child_id_col, child_time_col, child_foreign_key_col
        The intermediate relation table: ``child_id_col`` identifies each child row (the grandchildren's
        cutoff target), ``child_time_col`` is the child's own timestamp (used as the grandchild aggregation
        cutoff), ``child_foreign_key_col`` links each child row to its parent.
    grandchild_specs
        Aggregated onto ``child_df`` first (depth-1 hop), using ``child_id_col``/``child_time_col`` as the
        entity/cutoff pair.
    child_value_cols
        Aggregation spec (``{column: [agg_name, ...]}``) applied to the ENRICHED child table (original
        columns plus the newly-computed grandchild-aggregate columns) when rolling it up to the parent.
    prefix
        Prefix for the final parent-level features (the grandchild-level features already carry their own
        spec-level prefixes from the depth-1 hop).
    """
    enriched_child = compute_relational_features(
        parent_df=child_df,
        parent_id_col=child_id_col,
        cutoff_col=child_time_col,
        child_specs=grandchild_specs,
    )

    new_child_cols = [c for c in enriched_child.columns if c not in child_df.columns]
    # A child with no eligible grandchild history gets NaN from the depth-1 hop; treat that as "zero
    # contribution" for the depth-2 rollup (matching Featuretools' own zero-fill default for no-history
    # aggregates) rather than letting a single NaN row poison the whole parent-level sum via propagation.
    enriched_child[new_child_cols] = enriched_child[new_child_cols].fillna(0.0)
    full_value_cols: Dict[str, List[str]] = {**{k: list(v) for k, v in child_value_cols.items()}, **{c: ["mean", "sum"] for c in new_child_cols}}

    return compute_relational_features(
        parent_df=parent_df,
        parent_id_col=parent_id_col,
        cutoff_col=cutoff_col,
        child_specs=[
            ChildTableSpec(
                child_df=enriched_child,
                foreign_key_col=child_foreign_key_col,
                time_col=child_time_col,
                value_cols=full_value_cols,
                prefix=prefix,
            )
        ],
    )


__all__ = ["ChildTableSpec", "compute_relational_features", "stack_relational_features"]
