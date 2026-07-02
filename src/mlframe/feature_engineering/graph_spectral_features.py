"""Per-GRAPH spectral descriptor from an edge list (PZAD Spectral Graph Theory, feature-generation angle).

The spectral-graph-theory lecture (Дьяконов 2018) opens with ISOSPECTRALITY: a graph's spectrum does not depend on
the numbering of its vertices (slide 3). That permutation-invariance is exactly what a graph-classification model needs
- a fixed-length fingerprint of a whole graph that is identical however its nodes are ordered. This module turns one
graph (edge list) into a compact tabular row of spectral descriptors for ``train_mlframe_models_suite`` when each SAMPLE
is a graph (a molecule, an ego-network, a session graph, a dependency graph):

- connectivity: ``n_components`` (multiplicity of the zero Laplacian eigenvalue, slide 41) and ``algebraic_connectivity``
  = the Fiedler value λ2 (slide 25), which grows as the graph becomes better-connected.
- the smallest ``k`` non-trivial NORMALIZED-Laplacian eigenvalues (∈ [0, 2], size-comparable across graphs) as a
  low-frequency shape fingerprint, plus ``largest_norm_lap_eig`` (== 2 iff the graph has a bipartite component, slides
  42/49 - a spectral bipartiteness signal).
- adjacency spectrum summaries: ``spectral_radius`` = λ1(A) (bounded by max degree, slide 43), ``graph_energy`` = Σ|λi(A)|,
  and the spectral-moment identities ``n_edges`` = tr(A²)/2 and ``n_triangles`` = tr(A³)/6 (slide 19).
- ``laplacian_energy`` = Σ|μi − 2m/n|, a standard whole-graph spectral invariant.

This complements ``graph_structural_features`` (per-NODE) with a per-GRAPH descriptor. Uses dense eigendecomposition
(``numpy.linalg.eigvalsh``); intended for the small graphs typical of graph classification - guarded at ``max_nodes``.
Spectral embedding / clustering of ONE large affinity graph is `sklearn.manifold.SpectralEmbedding` / `SpectralClustering`
and the Fiedler reorder is `mlframe.core.matrix_seriation`; only this per-graph descriptor is provided here.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["graph_spectral_features", "graph_spectral_feature_names", "GRAPH_SPECTRAL_SCALARS"]

GRAPH_SPECTRAL_SCALARS = (
    "n_nodes",
    "n_edges",
    "mean_degree",
    "n_components",
    "algebraic_connectivity",
    "spectral_radius",
    "graph_energy",
    "laplacian_energy",
    "largest_norm_lap_eig",
    "n_triangles",
)


def _dense_adjacency(n_nodes: int, edges: np.ndarray, weights: np.ndarray | None):
    e = np.ascontiguousarray(edges, dtype=np.int64).reshape(-1, 2)
    if e.size and (e.min() < 0 or e.max() >= n_nodes):
        raise ValueError("graph_spectral_features: edge endpoints out of range [0, n_nodes).")
    w = np.ones(e.shape[0]) if weights is None else np.ascontiguousarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != e.shape[0]:
        raise ValueError("graph_spectral_features: weights length must match number of edges.")
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for idx in range(e.shape[0]):
        i, j = e[idx, 0], e[idx, 1]
        if i == j:
            continue  # drop self-loops
        A[i, j] = w[idx]
        A[j, i] = w[idx]  # undirected
    return A


def graph_spectral_feature_names(k: int = 5) -> list[str]:
    """Ordered feature names produced by :func:`graph_spectral_features` for a given ``k``."""
    return list(GRAPH_SPECTRAL_SCALARS) + [f"norm_lap_eig_{i+1}" for i in range(k)]


def graph_spectral_features(
    n_nodes: int,
    edges,
    *,
    weights=None,
    k: int = 5,
    max_nodes: int = 2000,
    zero_tol: float = 1e-8,
) -> dict:
    """Compact permutation-invariant spectral descriptor of one graph, as a flat ``{name: float}`` row.

    Parameters
    ----------
    n_nodes : number of nodes; node ids are ``0..n_nodes-1``.
    edges : ``(m, 2)`` int array of undirected edges (each edge once; self-loops dropped).
    weights : optional ``(m,)`` edge weights (default 1).
    k : how many of the smallest non-trivial normalized-Laplacian eigenvalues to emit as the shape fingerprint
        (padded with the maximum value 2.0 when the graph has fewer than ``k`` non-trivial eigenvalues).
    max_nodes : guard - dense eigendecomposition is O(n^3); raise above this to avoid a silent blow-up.

    Returns a dict with the :data:`GRAPH_SPECTRAL_SCALARS` plus ``norm_lap_eig_1..k`` (see :func:`graph_spectral_feature_names`).
    """
    if n_nodes < 1:
        raise ValueError("graph_spectral_features: require n_nodes >= 1.")
    if n_nodes > max_nodes:
        raise ValueError(f"graph_spectral_features: n_nodes={n_nodes} exceeds max_nodes={max_nodes} (dense eigdecomp is O(n^3)).")
    if k < 1:
        raise ValueError("graph_spectral_features: require k >= 1.")

    A = _dense_adjacency(n_nodes, np.asarray(edges), None if weights is None else np.asarray(weights))
    deg = A.sum(axis=1)
    n = n_nodes
    m = 0.5 * deg.sum()

    evA = np.linalg.eigvalsh(A)
    spectral_radius = float(evA[-1])
    graph_energy = float(np.abs(evA).sum())
    n_triangles = float(np.round((evA ** 3).sum() / 6.0))  # tr(A^3)/6 (unweighted count; rounded for FP noise)

    L = np.diag(deg) - A
    evL = np.linalg.eigvalsh(L)
    evL[evL < 0] = 0.0  # clip tiny negatives from FP
    n_components = int(np.count_nonzero(evL < zero_tol))
    algebraic_connectivity = float(evL[1]) if n >= 2 else 0.0  # classical Fiedler value λ2 (0 iff disconnected)
    avg_deg = 2.0 * m / n
    laplacian_energy = float(np.abs(evL - avg_deg).sum())

    # normalized Laplacian: symmetric, eigenvalues in [0, 2]; isolated nodes contribute a 0 (own component)
    with np.errstate(divide="ignore"):
        dinv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    Ln = np.eye(n) - (dinv_sqrt[:, None] * A * dinv_sqrt[None, :])
    Ln = 0.5 * (Ln + Ln.T)  # symmetrize against FP drift
    evLn = np.clip(np.linalg.eigvalsh(Ln), 0.0, 2.0)
    largest_norm_lap_eig = float(evLn[-1])  # == 2 iff a bipartite component (spectral bipartiteness signal)
    nontrivial = evLn[evLn > zero_tol]
    fingerprint = np.full(k, 2.0, dtype=np.float64)
    take = min(k, nontrivial.shape[0])
    fingerprint[:take] = nontrivial[:take]

    out = {
        "n_nodes": float(n),
        "n_edges": float(np.round(m)),
        "mean_degree": float(avg_deg),
        "n_components": float(n_components),
        "algebraic_connectivity": algebraic_connectivity,
        "spectral_radius": spectral_radius,
        "graph_energy": graph_energy,
        "laplacian_energy": laplacian_energy,
        "largest_norm_lap_eig": largest_norm_lap_eig,
        "n_triangles": n_triangles,
    }
    for i in range(k):
        out[f"norm_lap_eig_{i+1}"] = float(fingerprint[i])
    return out
