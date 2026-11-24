"""Cluster 11 regression: a subnormal-width candidate range takes the flat branch, not an overflowed scale (audit 2026-11-20, finding 11-F12).

``NB / (cmax - cmin)`` overflows to infinity for a subnormal width, and the binning then produced NaN. Such a range has no spread, so the flat
branch is taken and every edge is finite.
"""

import numpy as np

from mlframe.feature_selection._benchmarks.fe_operator_factory.kernel_prototypes import offset_kernels as ok


def test_subnormal_width_gives_finite_flat_edges():
    """A width of 1e-310 gives finite edges equal to the minimum candidate (11-F12)."""
    nbins, NB, n = 4, 8, 4
    u = np.zeros(n)
    v = np.array([1e-310, 2e-310, 1e-310, 2e-310])
    t = 0.0
    hist = np.zeros(NB, dtype=np.int64)
    mark = np.zeros(NB, dtype=np.bool_)
    gbuf = np.zeros(n + 1, dtype=np.float64)
    edges = np.zeros(nbins, dtype=np.float64)
    los = np.zeros(nbins, dtype=np.float64)
    his = np.zeros(nbins, dtype=np.float64)
    fracs = np.zeros(nbins, dtype=np.float64)
    ok._edges_hist_refine(u, v, t, nbins, NB, hist, mark, gbuf, edges, los, his, fracs)
    assert np.isfinite(edges).all()
    assert np.allclose(edges[1:], 1e-310)
