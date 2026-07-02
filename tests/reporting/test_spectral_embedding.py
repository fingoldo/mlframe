"""Tests for the spectral graph-embedding chart (charts/spectral_embedding.py).

Covers: layout shape + disconnected-graph no-crash (unit), and biz_value -- a 2-community graph (dense within, sparse
between) separates into two clusters in the Fiedler-vector layout (between/within separation ratio >= 2x), while a single
well-connected community shows no such split (ratio near 1).
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.spectral_embedding import (
    compose_spectral_embedding_figure,
    spectral_embedding_panel,
    spectral_layout,
)
from mlframe.reporting.spec import FigureSpec, NetworkPanelSpec


def _two_community_graph(m=15, p_between=0.02, seed=0):
    """Two dense cliques of ``m`` nodes with only sparse cross edges; returns (n_nodes, edges, labels)."""
    rng = np.random.default_rng(seed)
    edges = []
    for base in (0, m):
        for i in range(base, base + m):
            for j in range(i + 1, base + m):
                edges.append((i, j))
    for i in range(m):
        for j in range(m, 2 * m):
            if rng.random() < p_between:
                edges.append((i, j))
    labels = np.array([0] * m + [1] * m)
    return 2 * m, np.array(edges, dtype=np.int64), labels


def _one_community_graph(n=30, p=0.5, seed=0):
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    return n, np.array(edges, dtype=np.int64)


def _separation_ratio(coords, labels):
    c0 = coords[labels == 0].mean(axis=0)
    c1 = coords[labels == 1].mean(axis=0)
    between = np.linalg.norm(c0 - c1)
    within = 0.0
    for lab, cen in ((0, c0), (1, c1)):
        pts = coords[labels == lab]
        within += np.linalg.norm(pts - cen, axis=1).mean()
    within = within / 2.0
    return between / max(within, 1e-12)


# ----------------------------------------------------------------------------
# Unit
# ----------------------------------------------------------------------------


def test_layout_shape():
    n, edges, _ = _two_community_graph()
    coords = spectral_layout(n, edges)
    assert coords.shape == (n, 2)


def test_disconnected_graph_no_crash():
    # Two isolated components + fully isolated nodes (no edges to some) -- must not raise.
    edges = np.array([(0, 1), (1, 2), (5, 6)], dtype=np.int64)
    coords = spectral_layout(8, edges)
    assert coords.shape == (8, 2)
    assert np.all(np.isfinite(coords))


def test_edgeless_graph_jittered_not_stacked():
    coords = spectral_layout(5, np.empty((0, 2), dtype=np.int64))
    assert coords.shape == (5, 2)
    assert coords.std() > 0  # degenerate layout gets a deterministic jitter so nodes don't stack


def test_tiny_graph():
    coords = spectral_layout(1, np.empty((0, 2), dtype=np.int64))
    assert coords.shape == (1, 2)


def test_panel_and_figure():
    n, edges, labels = _two_community_graph()
    panel = spectral_embedding_panel(n, edges, node_color=labels)
    assert isinstance(panel, NetworkPanelSpec)
    assert panel.node_x.shape == (n,) and panel.node_y.shape == (n,)
    assert len(panel.node_color) == n
    assert panel.edge_src.shape == panel.edge_dst.shape
    fig = compose_spectral_embedding_figure(n, edges, node_color=labels)
    assert isinstance(fig, FigureSpec)


# ----------------------------------------------------------------------------
# biz_value
# ----------------------------------------------------------------------------


def test_biz_two_communities_split_in_layout():
    """A 2-community graph splits into two clusters in the spectral layout: between-centroid distance >= 2x the
    within-community spread. Measured ratio ~5x+; floor 2.0. A regression in the Fiedler-vector layout collapses it."""
    n, edges, labels = _two_community_graph()
    coords = spectral_layout(n, edges)
    ratio = _separation_ratio(coords, labels)
    assert ratio >= 2.0, ratio


def test_biz_single_community_no_split():
    """A single well-connected community shows no 2-cluster split: with an arbitrary label bisection the
    between/within ratio stays near 1 (well below the 2x separation the true 2-community graph reaches)."""
    n, edges = _one_community_graph()
    coords = spectral_layout(n, edges)
    labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
    ratio = _separation_ratio(coords, labels)
    assert ratio < 2.0, ratio
