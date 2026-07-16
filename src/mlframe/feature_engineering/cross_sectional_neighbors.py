"""``compute_cross_sectional_neighbor_features``: nearest-neighbor snapshots + a distance-ratio isolation feature.

Source: Optiver Realized Volatility Prediction 3rd place -- for each ``time_id`` (a cross-sectional snapshot
across many stocks), find nearest-neighbor ``time_id``s by similarity of the snapshot's feature vector, then
add features derived from those neighbors (aggregate stats over the neighbor set) plus the ratio of 1st-NN
distance to k-th-NN distance as a confidence/isolation-style feature.

Distinct from :func:`neighbor_aggregate_features.compute_neighbor_aggregate_features` (this session's
earlier addition): that function runs kNN at ROW level (each row is a point in the similarity space, OOF
fold-safe for target aggregation). This function instead first collapses each SNAPSHOT (all rows sharing one
``snapshot_col`` value) into ONE feature vector, runs kNN across SNAPSHOTS, and broadcasts the resulting
neighbor-set aggregates + distance-ratio feature back down to every row of that snapshot -- the right
granularity when "how does this whole cross-section compare to other cross-sections" is the question, not
"how does this row compare to other rows."

No OOF fold-safety is needed here (unlike row-level target aggregation): the snapshot vectors and neighbor
aggregates are built purely from FEATURE columns, never the target, so there is no leakage risk from a
snapshot's own rows contributing to its own neighbor search.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from .transformer._knn_helper import knn_search


def _snapshot_stats_for_k(
    vectors: np.ndarray,
    neighbor_idx_sorted: np.ndarray,
    dists_sorted: np.ndarray,
    k: int,
    feature_cols: Sequence[str],
    agg_stats: Sequence[str],
    column_prefix: str,
) -> dict:
    """Neighbor-aggregate + distance-ratio columns for one ``k``, slicing the top-``k`` columns out of an
    already max-k-sized sorted neighbor table (no re-search)."""
    idx_k = neighbor_idx_sorted[:, :k]
    dists_k = dists_sorted[:, :k]

    out: dict = {}
    for col_j, col in enumerate(feature_cols):
        neighbor_vals = vectors[idx_k, col_j]  # (n_snapshots, k)
        for stat in agg_stats:
            reducer = getattr(np, f"nan{stat}", None) or getattr(np, stat)
            out[f"{column_prefix}_{col}_{stat}"] = reducer(neighbor_vals, axis=1)

    k_eff = dists_k.shape[1]
    first_nn = dists_k[:, 0]
    kth_nn = dists_k[:, k_eff - 1]
    # 1.0 (maximally "isolated/uncertain") for the degenerate all-neighbors-at-distance-0 case.
    distance_ratio = np.divide(first_nn, kth_nn, out=np.ones_like(first_nn), where=kth_nn > 0)
    out[f"{column_prefix}_distance_ratio"] = distance_ratio
    return out


def compute_cross_sectional_neighbor_features(
    df: pd.DataFrame,
    snapshot_col: str,
    feature_cols: Sequence[str],
    k: int = 10,
    agg_stats: Sequence[str] = ("mean", "std"),
    column_prefix: str = "xsnn",
    k_values: Optional[Sequence[int]] = None,
) -> pl.DataFrame:
    """Build one feature vector per snapshot, find its k nearest OTHER snapshots, and broadcast neighbor-set
    aggregates plus a 1st-NN/k-th-NN distance-ratio isolation feature back to every row of that snapshot.

    Parameters
    ----------
    df
        Row-level feature frame containing ``snapshot_col`` and every column in ``feature_cols``.
    snapshot_col
        Column defining cross-sectional snapshots (e.g. ``"time_id"``); all rows sharing a value are
        collapsed into one snapshot vector (mean of ``feature_cols``).
    feature_cols
        Columns to average into each snapshot's vector and to aggregate over its neighbor set.
    k
        Number of nearest OTHER snapshots. Ignored (as the base radius) when ``k_values`` is given --
        use ``k_values`` instead in that case.
    agg_stats
        Aggregate stats computed over the neighbor snapshots' own vectors, per feature column: any pandas
        reduction name (``"mean"``, ``"std"``, ``"min"``, ``"max"``, ``"median"``, ...).
    column_prefix
        Output column-name prefix.
    k_values
        Opt-in multi-k mode: compute neighbor-aggregate + distance-ratio features at EACH k in this
        sequence in one call, columns suffixed ``_k{k}`` (e.g. ``xsnn_k5_f0_mean``, ``xsnn_k20_f0_mean``).
        The neighbor search runs ONCE, extended to ``max(k_values)``, and each k's features are sliced out
        of that single sorted neighbor table -- far cheaper than calling this function once per k value
        (which would repeat the O(n_snapshots log n_snapshots) search each time). When ``None`` (default),
        behavior is unchanged from the original single-k contract (unprefixed-by-k column names).

    Returns
    -------
    pl.DataFrame
        One row per row of ``df`` (same order). Single-k mode (``k_values=None``): ``len(feature_cols) *
        len(agg_stats)`` neighbor-aggregate columns plus ``{column_prefix}_distance_ratio``. Multi-k mode:
        the same columns repeated per ``k`` in ``k_values`` with a ``_k{k}`` suffix inserted after
        ``column_prefix``. The distance-ratio feature is near 0 for a snapshot in a dense, well-matched
        cluster of neighbors; near 1 for an isolated snapshot whose neighbors are all roughly equally (far)
        distant, a local-density/outlier-confidence signal reusable well beyond this specific feature block.
    """
    snapshot_vectors = df.groupby(snapshot_col, sort=False)[list(feature_cols)].mean()
    snapshot_ids = snapshot_vectors.index.to_numpy()
    vectors = snapshot_vectors.to_numpy(dtype=np.float32)
    n_snapshots = vectors.shape[0]

    k_max = max(k_values) if k_values else k
    k_query = min(k_max + 1, n_snapshots)  # +1 to drop the trivial self-match (distance 0) below
    dists, neighbor_idx = knn_search(vectors, vectors, k_query)

    self_mask = neighbor_idx == np.arange(n_snapshots).reshape(-1, 1)
    # A snapshot always matches itself at distance 0; drop that column per row (self may not always be at
    # position 0 if duplicate/zero-distance ties exist, hence the mask rather than a hard-coded slice).
    neighbor_idx_no_self = np.where(self_mask, -1, neighbor_idx)
    dists_no_self = np.where(self_mask, np.inf, dists)
    order = np.argsort(dists_no_self, axis=1)
    neighbor_idx_sorted = np.take_along_axis(neighbor_idx_no_self, order, axis=1)[:, :k_max]
    dists_sorted = np.take_along_axis(dists_no_self, order, axis=1)[:, :k_max]

    if k_values:
        snapshot_out: dict = {}
        for kv in k_values:
            snapshot_out.update(_snapshot_stats_for_k(vectors, neighbor_idx_sorted, dists_sorted, kv, feature_cols, agg_stats, f"{column_prefix}_k{kv}"))
    else:
        snapshot_out = _snapshot_stats_for_k(vectors, neighbor_idx_sorted, dists_sorted, k, feature_cols, agg_stats, column_prefix)

    snapshot_df = pd.DataFrame(snapshot_out, index=snapshot_ids)
    snapshot_df.index.name = snapshot_col

    broadcast = df[[snapshot_col]].join(snapshot_df, on=snapshot_col)
    return pl.from_pandas(broadcast.drop(columns=[snapshot_col]))


__all__ = ["compute_cross_sectional_neighbor_features"]
