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

from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl

from .transformer._knn_helper import knn_search


def compute_cross_sectional_neighbor_features(
    df: pd.DataFrame,
    snapshot_col: str,
    feature_cols: Sequence[str],
    k: int = 10,
    agg_stats: Sequence[str] = ("mean", "std"),
    column_prefix: str = "xsnn",
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
        Number of nearest OTHER snapshots.
    agg_stats
        Aggregate stats computed over the neighbor snapshots' own vectors, per feature column: any pandas
        reduction name (``"mean"``, ``"std"``, ``"min"``, ``"max"``, ``"median"``, ...).
    column_prefix
        Output column-name prefix.

    Returns
    -------
    pl.DataFrame
        One row per row of ``df`` (same order), with ``len(feature_cols) * len(agg_stats)`` neighbor-
        aggregate columns plus ``{column_prefix}_distance_ratio`` (1st-NN distance / k-th-NN distance --
        near 0 for a snapshot in a dense, well-matched cluster of neighbors; near 1 for an isolated
        snapshot whose neighbors are all roughly equally (far) distant, a local-density/outlier-confidence
        signal reusable well beyond this specific feature block).
    """
    snapshot_vectors = df.groupby(snapshot_col, sort=False)[list(feature_cols)].mean()
    snapshot_ids = snapshot_vectors.index.to_numpy()
    vectors = snapshot_vectors.to_numpy(dtype=np.float32)
    n_snapshots = vectors.shape[0]

    k_query = min(k + 1, n_snapshots)  # +1 to drop the trivial self-match (distance 0) below
    dists, neighbor_idx = knn_search(vectors, vectors, k_query)

    self_mask = neighbor_idx == np.arange(n_snapshots).reshape(-1, 1)
    # A snapshot always matches itself at distance 0; drop that column per row (self may not always be at
    # position 0 if duplicate/zero-distance ties exist, hence the mask rather than a hard-coded slice).
    neighbor_idx_no_self = np.where(self_mask, -1, neighbor_idx)
    dists_no_self = np.where(self_mask, np.inf, dists)
    order = np.argsort(dists_no_self, axis=1)
    neighbor_idx_sorted = np.take_along_axis(neighbor_idx_no_self, order, axis=1)[:, :k]
    dists_sorted = np.take_along_axis(dists_no_self, order, axis=1)[:, :k]

    snapshot_out: dict = {}
    for col_j, col in enumerate(feature_cols):
        neighbor_vals = vectors[neighbor_idx_sorted, col_j]  # (n_snapshots, k)
        for stat in agg_stats:
            reducer = getattr(np, f"nan{stat}", None) or getattr(np, stat)
            snapshot_out[f"{column_prefix}_{col}_{stat}"] = reducer(neighbor_vals, axis=1)

    k_eff = dists_sorted.shape[1]
    first_nn = dists_sorted[:, 0]
    kth_nn = dists_sorted[:, k_eff - 1]
    # 1.0 (maximally "isolated/uncertain") for the degenerate all-neighbors-at-distance-0 case.
    distance_ratio = np.divide(first_nn, kth_nn, out=np.ones_like(first_nn), where=kth_nn > 0)
    snapshot_out[f"{column_prefix}_distance_ratio"] = distance_ratio

    snapshot_df = pd.DataFrame(snapshot_out, index=snapshot_ids)
    snapshot_df.index.name = snapshot_col

    broadcast = df[[snapshot_col]].join(snapshot_df, on=snapshot_col)
    return pl.from_pandas(broadcast.drop(columns=[snapshot_col]))


__all__ = ["compute_cross_sectional_neighbor_features"]
