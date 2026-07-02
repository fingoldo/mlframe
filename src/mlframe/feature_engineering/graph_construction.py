"""Build a graph FROM ordinary tabular columns so the graph features apply to plain float/categorical datasets.

`graph_features` needs an edge list, but a normal (X, y) table has none. This module constructs one from the data
itself, covering the three tabular situations the SNA lecture's examples map onto (slide 3 "интернет-магазин —
связь по одинаковым купленным товарам"; slide 20 affiliation networks):

- FLOAT columns -> ``knn_graph_edges``: link each row to its k nearest neighbours in feature space (a similarity
  graph). Feeding this to ``graph_neighbor_aggregate`` gives a label-propagation / graph-smoothed target feature;
  to ``graph_structural_features`` gives local density / outlier-ness (low degree, low clustering = isolated row).
- CATEGORICAL / GROUP column -> ``shared_attribute_edges``: link rows sharing a category (the affiliation graph).
  Neighbour-degree = "how many other rows share my value"; neighbour-label-mean = leakage-safe group target rate.
- TIMESTAMP column -> pass ``timestamp=`` to either constructor: an edge is kept only toward the PAST
  (``t[neighbour] < t[row]``), yielding a directed past-only graph so a temporal feature never peeks at the future.

Both constructors bound memory (a shared category with m rows would otherwise make an O(m^2) clique): pass a
``max_neighbours`` window and/or ``max_group_size`` cap; skipped work is logged, never silently dropped.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["knn_graph_edges", "shared_attribute_edges"]


def knn_graph_edges(
    X: np.ndarray,
    k: int,
    *,
    metric: str = "minkowski",
    timestamp: np.ndarray | None = None,
    return_weights: bool = False,
):
    """k-nearest-neighbour similarity graph on a float feature matrix: edge (i, j) for each of row i's k neighbours.

    Parameters
    ----------
    X : (n, d) float array. k : neighbours per row (excluding self).
    timestamp : optional (n,) times; when given, keep only edges toward earlier rows (``t[j] < t[i]``) -> a directed
        past-only graph for leakage-safe temporal features.
    return_weights : also return ``1/(1+distance)`` per edge (use as ``weights`` in the graph-feature aggregators).

    Returns ``edges`` (E, 2) int64, or ``(edges, weights)`` when ``return_weights``.
    """
    from sklearn.neighbors import NearestNeighbors

    Xf = np.ascontiguousarray(X, dtype=np.float64)
    n = Xf.shape[0]
    if k < 1:
        raise ValueError("knn_graph_edges: k must be >= 1.")
    kq = min(k + 1, n)  # +1 because the first neighbour is the point itself
    nn = NearestNeighbors(n_neighbors=kq, metric=metric).fit(Xf)
    dist, idx = nn.kneighbors(Xf)
    rows = np.repeat(np.arange(n), kq)
    cols = idx.ravel()
    dd = dist.ravel()
    keep = rows != cols  # drop self
    rows, cols, dd = rows[keep], cols[keep], dd[keep]
    if timestamp is not None:
        t = np.ascontiguousarray(timestamp, dtype=np.float64).ravel()
        if t.shape[0] != n:
            raise ValueError("knn_graph_edges: timestamp length must equal n rows.")
        past = t[cols] < t[rows]
        rows, cols, dd = rows[past], cols[past], dd[past]
    edges = np.stack([rows, cols], axis=1).astype(np.int64)
    if return_weights:
        return edges, 1.0 / (1.0 + dd)
    return edges


def shared_attribute_edges(
    codes: np.ndarray,
    *,
    timestamp: np.ndarray | None = None,
    max_neighbours: int | None = 100,
    max_group_size: int | None = None,
):
    """Affiliation graph from a categorical / group column: link rows that share the same code.

    A group of ``m`` rows is a clique with ``m*(m-1)/2`` edges, so large categories are bounded:

    Parameters
    ----------
    codes : (n,) integer-encoded category / group id per row (use ``pd.factorize`` / label-encode first).
    timestamp : optional (n,) times; when given each row links only to EARLIER same-group rows (directed past graph)
        -> leakage-safe "recent same-group peers" features.
    max_neighbours : cap each row to at most this many same-group partners (the most recent when ``timestamp`` is
        given, else the nearest by row order); ``None`` = full clique. Bounds edges to ``n * max_neighbours``.
    max_group_size : skip (with a WARN) any group larger than this; ``None`` = no skip. Use as a hard safety valve.

    Returns ``edges`` (E, 2) int64.
    """
    c = np.ascontiguousarray(codes).ravel()
    n = c.shape[0]
    t = None if timestamp is None else np.ascontiguousarray(timestamp, dtype=np.float64).ravel()
    if t is not None and t.shape[0] != n:
        raise ValueError("shared_attribute_edges: timestamp length must equal n rows.")

    order = np.argsort(c, kind="stable")
    c_sorted = c[order]
    boundaries = np.flatnonzero(np.diff(c_sorted)) + 1
    groups = np.split(order, boundaries)  # each is the original-row-indices of one category

    src_list: list[np.ndarray] = []
    dst_list: list[np.ndarray] = []
    skipped = 0
    for members in groups:
        m = members.shape[0]
        if m < 2:
            continue
        if max_group_size is not None and m > max_group_size:
            skipped += 1
            continue
        mem = members if t is None else members[np.argsort(t[members], kind="stable")]  # ascending time within the group
        for pos in range(m):
            i = mem[pos]
            if t is not None:
                # directed edge i -> each strictly-earlier same-group peer (leakage-safe past); bound the window
                lo = 0 if max_neighbours is None else max(0, pos - max_neighbours)
                partners = mem[lo:pos]
            else:
                # one undirected edge per pair (upper triangle) -> graph_structural_features symmetrizes correctly
                hi = m if max_neighbours is None else min(m, pos + 1 + max_neighbours)
                partners = mem[pos + 1 : hi]
            if partners.size:
                src_list.append(np.full(partners.shape[0], i, dtype=np.int64))
                dst_list.append(partners.astype(np.int64))
    if skipped:
        logger.warning("shared_attribute_edges: skipped %d group(s) larger than max_group_size=%s.", skipped, max_group_size)
    if not src_list:
        return np.empty((0, 2), dtype=np.int64)
    return np.stack([np.concatenate(src_list), np.concatenate(dst_list)], axis=1)
