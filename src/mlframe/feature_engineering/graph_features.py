"""Graph -> tabular per-node features from an edge list (PZAD SNA, feature-generation angle).

The social-network lecture (Дьяконов 2020) stresses repeatedly that graph structure IS a feature source
("Внимание! Это признак" for the clustering coefficient, slide 46; "Это неплохой признак" for local bridges,
slide 30) and that the homophily / social-influence signal — P(a node adopts a label | k of its neighbors
already have it), slides 60-62 (Backstrom, Crandall) — predicts node behavior. This module turns an edge list
plus a per-node value into TABULAR columns for ``train_mlframe_models_suite``:

- ``graph_neighbor_aggregate`` — for each node, aggregate a per-node value (a LABEL or a feature) over its graph
  neighbors (mean/sum/max/min/count/weighted-mean). With a binary label this is the homophily / social-influence
  feature. It is LEAKAGE-SENSITIVE: pass TRAIN-ONLY (or out-of-fold) labels so a node never sees its own target.
  ``networkx`` has no such leakage-safe neighbor-target aggregation — it computes structure, not label spread.
- ``graph_structural_features`` — per-node degree, weighted strength, local clustering coefficient and triangle
  count, computed directly from the edge list (numba two-pointer intersection on sorted CSR adjacency, no
  networkx dependency and no per-node Python-dict overhead), returned as a feature block aligned to node id.

Pass a SIMPLE undirected edge list (each edge once; self-loops are dropped). O(sum deg^2) for clustering, which
is cheap on the sparse graphs SNA targets (avg degree ~10, slide 50).
"""

from __future__ import annotations

import logging

import numpy as np

try:
    import numba

    _HAS_NUMBA = True
except Exception:  # numba is an optional accelerator
    _HAS_NUMBA = False

logger = logging.getLogger(__name__)

__all__ = [
    "graph_neighbor_aggregate",
    "graph_structural_features",
    "link_prediction_features",
    "GRAPH_STRUCT_FEATURES",
    "LINK_PREDICTION_FEATURES",
    "NEIGHBOR_AGGS",
]

LINK_PREDICTION_FEATURES = ("common_neighbors", "jaccard", "adamic_adar", "resource_allocation", "preferential_attachment")

GRAPH_STRUCT_FEATURES = ("degree", "strength", "clustering", "triangles")
NEIGHBOR_AGGS = ("mean", "sum", "max", "min", "count", "wmean")


def _build_csr(n_nodes: int, edges: np.ndarray, weights: np.ndarray | None, directed: bool):
    """Sorted CSR adjacency (indptr, indices, data); neighbours per node are sorted ascending for two-pointer intersection."""
    e = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
    w = np.ones(e.shape[0], dtype=np.float64) if weights is None else np.ascontiguousarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != e.shape[0]:
        raise ValueError("graph features: weights length must match number of edges.")
    src, dst = e[:, 0], e[:, 1]
    if not directed:
        src, dst, w = np.concatenate([src, dst]), np.concatenate([dst, src]), np.concatenate([w, w])
    keep = src != dst  # drop self-loops
    src, dst, w = src[keep], dst[keep], w[keep]
    if src.size and (src.min() < 0 or src.max() >= n_nodes or dst.max() >= n_nodes):
        raise ValueError("graph features: edge endpoint out of range [0, n_nodes).")
    order = np.lexsort((dst, src))  # primary src, secondary dst
    src, indices, data = src[order], dst[order], w[order]
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(src, minlength=n_nodes))
    return indptr, np.ascontiguousarray(indices), np.ascontiguousarray(data)


def _wmean_impl(indptr, indices, data, values, fill):
    """CSR-graph edge-weighted mean of neighbour ``values`` per node; nodes with zero total edge weight (isolated or all-zero-weight neighbours) keep ``fill``."""
    n = indptr.shape[0] - 1
    out = np.full(n, fill, dtype=np.float64)
    for u in range(n):
        s, e = indptr[u], indptr[u + 1]
        wsum = 0.0
        acc = 0.0
        for p in range(s, e):
            wsum += data[p]
            acc += data[p] * values[indices[p]]
        if wsum > 0.0:
            out[u] = acc / wsum
    return out


def _sum_impl(indptr, indices, values, fill, take_max, take_min):
    """CSR-graph neighbour aggregate: sum by default, or max/min of neighbour ``values`` when ``take_max``/``take_min`` is set; isolated nodes (no neighbours) keep ``fill``."""
    n = indptr.shape[0] - 1
    out = np.full(n, fill, dtype=np.float64)
    for u in range(n):
        s, e = indptr[u], indptr[u + 1]
        if e <= s:
            continue
        acc = 0.0
        vmax = -np.inf
        vmin = np.inf
        for p in range(s, e):
            v = values[indices[p]]
            acc += v
            if v > vmax:
                vmax = v
            if v < vmin:
                vmin = v
        if take_max:
            out[u] = vmax
        elif take_min:
            out[u] = vmin
        else:
            out[u] = acc
    return out


def _clustering_impl(indptr, indices):
    """Local clustering coefficient and triangle count per node via sorted-adjacency merge-intersection of each pair of neighbour lists (each node's CSR row is assumed sorted); nodes with degree < 2 get 0 for both."""
    n = indptr.shape[0] - 1
    clus = np.zeros(n, dtype=np.float64)
    tri = np.zeros(n, dtype=np.float64)
    for u in range(n):
        su, eu = indptr[u], indptr[u + 1]
        deg = eu - su
        if deg < 2:
            continue
        common_total = 0
        for p in range(su, eu):
            v = indices[p]
            sv, ev = indptr[v], indptr[v + 1]
            i = su
            j = sv
            while i < eu and j < ev:
                a = indices[i]
                b = indices[j]
                if a == b:
                    common_total += 1
                    i += 1
                    j += 1
                elif a < b:
                    i += 1
                else:
                    j += 1
        tri[u] = common_total / 2.0
        clus[u] = common_total / (deg * (deg - 1.0))
    return clus, tri


def _lp_features_impl(indptr, indices, deg, xs, ys):
    """Link-prediction features for each candidate (x, y) pair: common-neighbour count (via sorted-adjacency merge), Adamic-Adar (sum of 1/log(deg) over shared neighbours), and resource-allocation index (sum of 1/deg over shared neighbours)."""
    m = xs.shape[0]
    cn = np.zeros(m, dtype=np.float64)
    aa = np.zeros(m, dtype=np.float64)
    ra = np.zeros(m, dtype=np.float64)
    for p in range(m):
        x = xs[p]
        y = ys[p]
        i = indptr[x]
        ex = indptr[x + 1]
        j = indptr[y]
        ey = indptr[y + 1]
        c = 0
        a = 0.0
        r = 0.0
        while i < ex and j < ey:
            u = indices[i]
            v = indices[j]
            if u == v:  # a common neighbour z
                c += 1
                d = deg[u]
                if d > 1:  # z is a common neighbour so d>=2; log(d)>0
                    a += 1.0 / np.log(d)
                if d > 0:
                    r += 1.0 / d
                i += 1
                j += 1
            elif u < v:
                i += 1
            else:
                j += 1
        cn[p] = c
        aa[p] = a
        ra[p] = r
    return cn, aa, ra


if _HAS_NUMBA:
    _wmean_impl = numba.njit(cache=True)(_wmean_impl)
    _sum_impl = numba.njit(cache=True)(_sum_impl)
    _clustering_impl = numba.njit(cache=True, nogil=True)(_clustering_impl)
    _lp_features_impl = numba.njit(cache=True, nogil=True)(_lp_features_impl)


def graph_neighbor_aggregate(
    n_nodes: int,
    edges: np.ndarray,
    values: np.ndarray,
    *,
    agg: str = "mean",
    weights: np.ndarray | None = None,
    directed: bool = False,
    fill: float = 0.0,
) -> np.ndarray:
    """For each node, aggregate ``values[neighbour]`` over its graph neighbours -> one feature column (length ``n_nodes``).

    With a binary ``values`` (a label) and ``agg='mean'`` this is the homophily / social-influence feature: the
    fraction of a node's neighbours that carry the label (Backstrom's "k friends already members", slides 60-62).

    LEAKAGE: a node never aggregates its own value (only neighbours), but if ``values`` is the TARGET you must pass
    train-only / out-of-fold labels (isolated nodes / test rows get ``fill``) so the feature stays honest.

    Parameters
    ----------
    agg : {'mean','sum','max','min','count','wmean'}
        ``count`` ignores ``values`` (returns the neighbour count = degree); ``wmean`` is the edge-weight-weighted mean.
    weights : optional per-edge weights (aligned to ``edges``); used only by ``wmean``.
    """
    if agg not in NEIGHBOR_AGGS:
        raise ValueError(f"graph_neighbor_aggregate: agg must be one of {NEIGHBOR_AGGS}, got {agg!r}.")
    vals = np.ascontiguousarray(values, dtype=np.float64).ravel()
    if vals.shape[0] != n_nodes:
        raise ValueError("graph_neighbor_aggregate: values length must equal n_nodes.")
    indptr, indices, data = _build_csr(n_nodes, edges, weights, directed)
    if agg == "count":
        return np.asarray((indptr[1:] - indptr[:-1]).astype(np.float64))
    if agg == "wmean":
        return np.asarray(_wmean_impl(indptr, indices, data, vals, float(fill)))
    if agg == "mean":
        deg = (indptr[1:] - indptr[:-1]).astype(np.float64)
        s = _sum_impl(indptr, indices, vals, float(fill), False, False)
        out = np.full(n_nodes, float(fill), dtype=np.float64)
        nz = deg > 0
        out[nz] = s[nz] / deg[nz]
        return out
    return np.asarray(_sum_impl(indptr, indices, vals, float(fill), agg == "max", agg == "min"))


def graph_structural_features(
    n_nodes: int,
    edges: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    directed: bool = False,
) -> dict:
    """Per-node structural features from the edge list: ``degree``, ``strength`` (weighted degree), ``clustering``
    (local clustering coefficient), ``triangles`` (triangle count). Returns a dict of length-``n_nodes`` arrays,
    each a ready-to-use feature column for a tabular model. Clustering/triangles use the undirected graph."""
    indptr, indices, data = _build_csr(n_nodes, edges, weights, directed)
    degree = (indptr[1:] - indptr[:-1]).astype(np.float64)
    strength = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(strength, np.repeat(np.arange(n_nodes), np.diff(indptr)), data)
    # undirected CSR for clustering (symmetric); if directed input, rebuild symmetric view
    if directed:
        uindptr, uindices, _ = _build_csr(n_nodes, edges, weights, directed=False)
    else:
        uindptr, uindices = indptr, indices
    clustering, triangles = _clustering_impl(uindptr, uindices)
    return {"degree": degree, "strength": strength, "clustering": clustering, "triangles": triangles}


def link_prediction_features(n_nodes: int, edges: np.ndarray, pairs: np.ndarray) -> dict:
    """Pairwise link-prediction similarity features for candidate node pairs (PZAD LPP, slides 27-33).

    For each candidate ``(x, y)`` returns the classic vertex-similarity scores predicting whether an edge will
    appear -- the tabular feature block for link prediction framed as supervised binary classification:

    - ``common_neighbors`` = ``|Γ(x) ∩ Γ(y)|`` (the friend-of-a-friend principle)
    - ``jaccard`` = ``|Γ(x) ∩ Γ(y)| / |Γ(x) ∪ Γ(y)|`` (common-friend fraction)
    - ``adamic_adar`` = ``Σ_{z ∈ Γ(x)∩Γ(y)} 1/log|Γ(z)|`` (rare shared friends count more)
    - ``resource_allocation`` = ``Σ_{z ∈ Γ(x)∩Γ(y)} 1/|Γ(z)|``
    - ``preferential_attachment`` = ``|Γ(x)| · |Γ(y)|`` (sociable nodes link faster)

    Computed with a numba two-pointer intersection over sorted CSR adjacency, batched over all pairs at once (unlike
    ``networkx``'s per-pair Python generators). ``networkx`` remains the tool for the per-NODE importance features
    (PageRank / HITS / Katz / betweenness / closeness / eigenvector centrality) -- feed those columns via the same
    tabular pattern; only these batched pairwise scores are added here.

    Parameters
    ----------
    edges : (E, 2) undirected edge list (each edge once; self-loops dropped).
    pairs : (M, 2) candidate node pairs to score.

    Returns a dict of length-``M`` arrays keyed by :data:`LINK_PREDICTION_FEATURES`.
    """
    indptr, indices, _ = _build_csr(n_nodes, edges, None, directed=False)
    deg = (indptr[1:] - indptr[:-1]).astype(np.float64)
    pr = np.ascontiguousarray(pairs, dtype=np.int64).reshape(-1, 2)
    if pr.size and (pr.min() < 0 or pr.max() >= n_nodes):
        raise ValueError("link_prediction_features: pair endpoint out of range [0, n_nodes).")
    xs = np.ascontiguousarray(pr[:, 0])
    ys = np.ascontiguousarray(pr[:, 1])
    cn, aa, ra = _lp_features_impl(indptr, indices, deg, xs, ys)
    dx = deg[xs]
    dy = deg[ys]
    union = dx + dy - cn
    jaccard = np.divide(cn, union, out=np.zeros_like(cn), where=union > 0)
    return {
        "common_neighbors": cn,
        "jaccard": jaccard,
        "adamic_adar": aa,
        "resource_allocation": ra,
        "preferential_attachment": dx * dy,
    }
