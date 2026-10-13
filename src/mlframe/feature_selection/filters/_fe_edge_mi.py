"""Edge-binning plain plug-in MI for the CPU FE-batcher path - the bit-faithful CPU twin of the GPU
Edge plug-in MI.

The orthogonal/basis FE families score candidates by PLAIN plug-in MI(col; y). The legacy CPU kernel
(``_hermite_fe_mi._plugin_mi_classif_njit``) bins x by RANK (argsort -> n/nbins rows per bin); the GPU
kernel (``_hermite_fe_mi._plugin_mi_classif_batch_cuda_resident``) bins by PERCENTILE EDGE
(``_radix_select_interior_edges`` / cp.percentile + searchsorted). On continuous data the two are
bit-identical, but on TIED / low-cardinality columns they disagree (rank splits equal values across a bin
boundary, edge keeps them together) - which, under full GPU residency, would let the two backends select
different forms. This module is the CPU edge-binning twin so the CPU batcher path and the GPU batcher path
score candidates IDENTICALLY on EVERY column, tied or not. See ``_fe_mi_contract`` for the full contract.

Self-contained leaf (numpy + numba only): it reproduces the two canonical, separately-tested algorithms -
the equi-frequency percentile-edge binning of ``_usability_njit_pool._qbin_into`` (np.quantile lerp edges
at virtual index ``q*(n-1)`` + np.unique dedup + ``searchsorted(interior, side='right')``) and the plain
plug-in MI summation of ``_hermite_fe_mi._plugin_mi_from_binned_njit`` - so it carries no import-cycle
risk. The ``test_fe_edge_mi_parity`` suite pins it bit-equal (to ~1e-9) to a numpy reference AND, on a CUDA
host, to the GPU resident edge MI, on continuous AND tied fixtures.
"""
from __future__ import annotations

# The kernels live in ``_numba_utils`` (cross-module @njit invariant); re-exported here so existing importers keep working.
from ._numba_utils import _edge_bin_codes, _plugin_mi_classif_edge_njit, plugin_mi_classif_batch_edge_njit  # noqa: F401
