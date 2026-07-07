"""Spectral graph embedding: lay a graph out by its Laplacian Fiedler eigenvectors (PZAD Spectral Graph Theory, visual).

``mlframe.feature_engineering.graph_spectral_features`` fingerprints a whole graph by its Laplacian spectrum. The same
Laplacian's low-frequency EIGENVECTORS give a natural 2-D layout: the 2nd and 3rd smallest-eigenvalue eigenvectors (the
Fiedler vector and the next) place tightly-connected nodes near each other, so a graph with two communities visibly splits
into two clusters. This chart reuses that eigen code path (the combinatorial Laplacian ``diag(deg) - A`` built via the FE
module's ``_dense_adjacency``) and exposes the eigenvector layout the FE descriptor does not.

The eigensolve is the cost: dense ``numpy.linalg.eigh`` is O(n^3), fine for the small graphs typical of graph
classification. For large graphs only the 3 smallest eigenvectors are needed, so ``scipy.sparse.linalg.eigsh(k=3,
which='SA')`` on the sparse Laplacian is used above ``_DENSE_MAX_NODES`` when SciPy is available (k-smallest, not a full
dense decomposition).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from mlframe.reporting.colors import LINE_PALETTE, line_color
from mlframe.reporting.spec import FigureSpec, NetworkPanelSpec

# Above this node count a full dense O(n^3) eigendecomposition is wasteful when only 3 eigenvectors are needed; route to
# the sparse k-smallest ARPACK solver when SciPy is present (falls back to dense otherwise).
_DENSE_MAX_NODES: int = 500


def _laplacian_smallest_eigenvectors(n_nodes: int, edges, k: int = 3):
    """Return ``(eigvals, eigvecs)`` for the ``k`` smallest eigenvalues of the combinatorial Laplacian ``diag(deg) - A``.

    Reuses the FE module's ``_dense_adjacency`` so the graph is built identically to ``graph_spectral_features``.
    """
    from mlframe.feature_engineering.graph_spectral_features import _dense_adjacency

    A = _dense_adjacency(n_nodes, np.asarray(edges), None)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    L = 0.5 * (L + L.T)  # symmetrize against FP drift so eigh/eigsh see an exactly-symmetric operator
    kk = min(k, n_nodes)
    if n_nodes > _DENSE_MAX_NODES:
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh

            vals, vecs = eigsh(csr_matrix(L), k=min(kk, n_nodes - 1), which="SA")  # k smallest-algebraic (L is PSD)
            order = np.argsort(vals)
            return vals[order], vecs[:, order]
        except Exception:  # SciPy absent or ARPACK non-convergence -> dense fallback keeps the layout available  # nosec B110 - best-effort/optional path, no module logger
            pass
    vals, vecs = np.linalg.eigh(L)
    return vals[:kk], vecs[:, :kk]


def spectral_layout(n_nodes: int, edges, *, seed: int = 0) -> np.ndarray:
    """2-D spectral coordinates ``(n_nodes, 2)`` from the 2nd and 3rd smallest Laplacian eigenvectors.

    Handles disconnected / tiny graphs gracefully: missing higher eigenvectors leave a zero coordinate, and a fully
    degenerate layout (all coordinates collapsed, e.g. an edgeless graph) gets a small deterministic jitter so nodes
    remain distinguishable instead of stacking on one point.
    """
    if n_nodes < 1:
        raise ValueError("spectral_layout: require n_nodes >= 1.")
    _, vecs = _laplacian_smallest_eigenvectors(n_nodes, edges, k=3)
    coords = np.zeros((n_nodes, 2), dtype=np.float64)
    if vecs.shape[1] >= 2:
        coords[:, 0] = vecs[:, 1]
    if vecs.shape[1] >= 3:
        coords[:, 1] = vecs[:, 2]
    if float(coords.std()) < 1e-12:  # degenerate layout (edgeless / single-node): jitter so nodes don't stack
        rng = np.random.default_rng(seed)
        coords = rng.standard_normal((n_nodes, 2)) * 1e-3
    return coords


def _resolve_node_colors(node_color, n_nodes: int) -> tuple:
    """Resolve an optional integer community label array to palette color strings; uniform color when None."""
    if node_color is None:
        return tuple(LINE_PALETTE[0] for _ in range(n_nodes))
    labels = np.asarray(node_color).ravel()
    if labels.shape[0] != n_nodes:
        raise ValueError("spectral_embedding_panel: node_color length must equal n_nodes.")
    return tuple(line_color(int(c)) for c in labels)


def spectral_embedding_panel(n_nodes: int, edges, *, node_color=None, node_size=None, seed: int = 0) -> NetworkPanelSpec:
    """``NetworkPanelSpec`` placing nodes at :func:`spectral_layout` coordinates with edges as int src/dst arrays."""
    coords = spectral_layout(n_nodes, edges, seed=seed)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    edge_src = np.ascontiguousarray(e[:, 0])
    edge_dst = np.ascontiguousarray(e[:, 1])
    colors = _resolve_node_colors(node_color, n_nodes)
    if node_size is None:
        sizes = np.full(n_nodes, 120.0, dtype=np.float64)
    else:
        sizes = np.full(n_nodes, float(node_size), dtype=np.float64) if np.isscalar(node_size) else np.asarray(node_size, dtype=np.float64)
    return NetworkPanelSpec(
        node_x=np.ascontiguousarray(coords[:, 0]),
        node_y=np.ascontiguousarray(coords[:, 1]),
        node_size=sizes,
        node_color=colors,
        node_label=tuple(str(i) for i in range(n_nodes)),
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_weight=np.ones(edge_src.shape[0], dtype=np.float64),
        title="",
        xlabel="Fiedler vector",
        ylabel="3rd eigenvector",
    )


def compose_spectral_embedding_figure(
    n_nodes: int, edges, *, node_color=None, node_size=None, seed: int = 0,
    suptitle: str = "Spectral graph embedding",
) -> FigureSpec:
    """One-panel ``FigureSpec`` wrapping :func:`spectral_embedding_panel`."""
    panel = spectral_embedding_panel(n_nodes, edges, node_color=node_color, node_size=node_size, seed=seed)
    return FigureSpec(suptitle=suptitle, panels=((panel,),), figsize=(7.0, 6.0))


__all__ = [
    "spectral_layout",
    "spectral_embedding_panel",
    "compose_spectral_embedding_figure",
]
