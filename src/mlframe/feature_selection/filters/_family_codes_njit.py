"""Parallel quantile coding of a block of candidate columns for the family-wise permutation test.

``_family_maxt_gate._quantile_codes`` codes one column with ``np.unique`` / ``np.quantile`` / ``np.searchsorted`` (about 6 ms for 40000 rows, three sorts); a pair or modular family has
thousands of columns, so the loop was 22 s of a 50000-row conjunction fit. Here each column of a block is sorted once, its distinct levels counted from the sorted copy, and the interior
edges taken from the same sorted copy with numpy's ``linear`` quantile arithmetic (virtual index, gamma, ``lerp`` with the ``>= 0.5`` branch) so the codes equal the numpy path's.
"""

from __future__ import annotations


from ._numba_utils import (  # noqa: F401 -- helpers hosted in _numba_utils_cross_module for the cross-module njit invariant
    _edge_value,
    quantile_codes_block,
)
