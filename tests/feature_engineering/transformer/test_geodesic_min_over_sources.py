"""The single virtual-source search equals the minimum of one Dijkstra per source (audit 2026-11-20, finding 09-F9)."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from mlframe.feature_engineering.transformer.geodesic_kgraph import _min_geodesic_from_sources


def test_virtual_source_matches_the_per_source_minimum():
    """Random weighted symmetric graph, several sources: the helper must equal the elementwise minimum over per-source runs."""
    rng = np.random.default_rng(0)
    n = 60
    dense = np.zeros((n, n))
    for i in range(n):
        for j in rng.choice(n, 4, replace=False):
            if i != j:
                w = rng.uniform(0.5, 2.0)
                dense[i, j] = dense[j, i] = w
    graph = csr_matrix(dense)
    sources = np.array([3, 17, 42])
    expected = dijkstra(graph, indices=sources, directed=False).min(axis=0)
    got = _min_geodesic_from_sources(graph, sources)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)
