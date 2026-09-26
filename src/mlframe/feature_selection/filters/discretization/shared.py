"""Cross-package API of ``mlframe.feature_selection.filters.discretization``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection.filters.discretization`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (  # noqa: F401
    _discretize_array_impl as discretize_array_impl,
    _safe_code_dtype as safe_code_dtype,
)
from ._discretization_edges import (  # noqa: F401
    _bayesian_blocks_bin_edges as bayesian_blocks_bin_edges,
    _knuth_bin_edges as knuth_bin_edges,
)

__all__ = [
    "bayesian_blocks_bin_edges",
    "discretize_array_impl",
    "knuth_bin_edges",
    "safe_code_dtype",
]
