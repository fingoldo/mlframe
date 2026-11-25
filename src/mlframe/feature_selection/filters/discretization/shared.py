"""Cross-package API of ``mlframe.feature_selection.filters.discretization``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection.filters.discretization`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (
    _discretize_array_impl as discretize_array_impl,
    _safe_code_dtype as safe_code_dtype,
)
from ._multiselect import (
    multiselect_inplace,
)
from ._discretization_edges import (
    _bayesian_blocks_bin_edges as bayesian_blocks_bin_edges,
    _knuth_bin_edges as knuth_bin_edges,
)

from ._null_binner import PermutedTargetBinner
from ._quantile_radix import sort_xy, sorted_quantiles, unique_with_counts

from ._quantile_radix import quantiles_of_sorted, sorted_unique_with_counts
from ._code_dtype import append_code_columns, narrowest_code_dtype
from ._code_storage_plan import code_alloc_dtype, plan_code_storage
from ._categorical_target_fold import target_classes

__all__ = [
    "PermutedTargetBinner",
    "append_code_columns",
    "bayesian_blocks_bin_edges",
    "code_alloc_dtype",
    "discretize_array_impl",
    "knuth_bin_edges",
    "multiselect_inplace",
    "narrowest_code_dtype",
    "plan_code_storage",
    "quantiles_of_sorted",
    "safe_code_dtype",
    "sort_xy",
    "sorted_quantiles",
    "sorted_unique_with_counts",
    "target_classes",
    "unique_with_counts",
]
