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

    This is exactly the depth-2 (one hop) case of ``stack_relational_chain`` -- delegates to it so the two
    code paths are provably identical rather than merely similar.
    """
    return stack_relational_chain(
        parent_df=parent_df,
        parent_id_col=parent_id_col,
        cutoff_col=cutoff_col,
        hops=[
            RelationalHop(
                df=child_df,
                id_col=child_id_col,
                time_col=child_time_col,
                foreign_key_col=child_foreign_key_col,
                value_cols=child_value_cols,
            )
        ],
        leaf_specs=grandchild_specs,
        prefix=prefix,
    )


@dataclass
class RelationalHop:
    """One intermediate table in a depth-N relational chain (see ``stack_relational_chain``).

    Generalizes the ``child_df``/``child_id_col``/``child_time_col``/``child_foreign_key_col`` quadruple
    from ``stack_relational_features`` so it can be chained at arbitrary depth: a sequence of ``RelationalHop``
    ordered from the table closest to ``parent_df`` (index 0) to the table closest to the leaf level.

    Attributes
    ----------
    df
        This hop's own table -- one row per entity at this relational level.
    id_col
        Column in ``df`` identifying each row of this hop (the cutoff target for the NEXT hop in, i.e. the
        hop/leaf level one step deeper in the chain).
    time_col
        This hop's own timestamp column -- used both as its cutoff when receiving aggregates from the next
        hop in, AND as the cutoff comparison when this hop itself is aggregated onto the hop/parent one step
        out.
    foreign_key_col
        Column in ``df`` linking each row of this hop to the hop/parent one step OUT in the chain (matches
        that outer level's ``id_col``, or ``parent_id_col`` for the outermost hop).
    value_cols
        Aggregation spec (``{column: [agg_name, ...]}``) applied to this hop's OWN columns (in addition to
        the newly-derived aggregate columns rolled up from deeper hops) when rolling it up to the next level out.
    prefix
        Prefix for the columns produced when this hop is rolled up onto the next level out.  Defaults to
        ``l{depth}`` (1-indexed from the leaf) if left empty.
    """

    df: pd.DataFrame
    id_col: str
    time_col: str
    foreign_key_col: str
    value_cols: Mapping[str, Sequence[str]] = field(default_factory=dict)
    prefix: str = ""


def stack_relational_chain(
    parent_df: pd.DataFrame,
    parent_id_col: str,
    cutoff_col: str,
    hops: Sequence[RelationalHop],
    leaf_specs: Sequence[ChildTableSpec],
    prefix: str = "l2",
) -> pd.DataFrame:
    """Generic depth-N stacking: recursively aggregate a chain of ``len(hops) + 1`` relational levels
    (parent -> hops[0] -> hops[1] -> ... -> hops[-1] -> leaf_specs) up to ``parent_df`` in one call, staying
    leakage-safe at EVERY hop -- each level can only see rows of the level one step deeper that are strictly
    before its OWN cutoff/timestamp, exactly like ``stack_relational_features`` but generalized from a fixed
    depth-2 chain to an arbitrary-length one, without the caller manually chaining ``compute_relational_features``
    calls hop by hop.

    Parameters
    ----------
    hops
        The chain of intermediate tables, ordered OUTERMOST (closest to ``parent_df``) first.  Must be
        non-empty -- a single-hop chain is exactly ``stack_relational_features``'s depth-2 case.
    leaf_specs
        One or more ``ChildTableSpec`` aggregated onto ``hops[-1]`` first (the innermost/deepest hop),
        using ``hops[-1].id_col``/``hops[-1].time_col`` as the entity/cutoff pair.
    prefix
        Prefix for the final parent-level features produced by rolling ``hops[0]`` up to ``parent_df``.
        Intermediate hops get an ``l{depth}`` prefix (1-indexed from the leaf) unless a hop sets its own.

    Returns
    -------
    pd.DataFrame
        ``parent_df`` plus the rolled-up, multi-hop-aggregate feature columns.
    """
    if not hops:
        raise ValueError("hops must contain at least one RelationalHop")

    # Fold from the deepest hop inward: aggregate leaf_specs onto hops[-1], zero-fill (a hop with no
    # eligible deeper-level history gets NaN, treated as "zero contribution" -- matching Featuretools' own
    # zero-fill default for no-history aggregates), then treat the now-enriched hop as a ChildTableSpec for
    # the next level out. Repeating this at every hop generalizes stack_relational_features's single fold.
    enriched = compute_relational_features(
        parent_df=hops[-1].df,
        parent_id_col=hops[-1].id_col,
        cutoff_col=hops[-1].time_col,
        child_specs=leaf_specs,
    )
    new_cols = [c for c in enriched.columns if c not in hops[-1].df.columns]
    enriched[new_cols] = enriched[new_cols].fillna(0.0)
    value_cols: Dict[str, List[str]] = {**{k: list(v) for k, v in hops[-1].value_cols.items()}, **{c: ["mean", "sum"] for c in new_cols}}

    for depth in range(len(hops) - 1, 0, -1):
        outer = hops[depth - 1]
        inner = hops[depth]
        spec = ChildTableSpec(
            child_df=enriched,
            foreign_key_col=inner.foreign_key_col,
            time_col=inner.time_col,
            value_cols=value_cols,
            prefix=inner.prefix or f"l{len(hops) - depth + 1}",
        )
        rolled = compute_relational_features(parent_df=outer.df, parent_id_col=outer.id_col, cutoff_col=outer.time_col, child_specs=[spec])
        new_cols = [c for c in rolled.columns if c not in outer.df.columns]
        rolled[new_cols] = rolled[new_cols].fillna(0.0)
        enriched = rolled
        value_cols = {**{k: list(v) for k, v in outer.value_cols.items()}, **{c: ["mean", "sum"] for c in new_cols}}

    final_spec = ChildTableSpec(
        child_df=enriched,
        foreign_key_col=hops[0].foreign_key_col,
        time_col=hops[0].time_col,
        value_cols=value_cols,
        prefix=prefix,
    )
    return compute_relational_features(parent_df=parent_df, parent_id_col=parent_id_col, cutoff_col=cutoff_col, child_specs=[final_spec])


__all__ = ["ChildTableSpec", "RelationalHop", "compute_relational_features", "stack_relational_features", "stack_relational_chain"]
