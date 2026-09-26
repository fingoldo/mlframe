"""Cross-package API of ``mlframe.reporting.charts``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.reporting.charts`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from .error_analysis import (  # noqa: F401
    _per_row_error as per_row_error,
)
from .pdp_ice import (  # noqa: F401
    _as_2d as as_2d,
    _resolve_feature_index as resolve_feature_index,
    _subsample_idx as subsample_idx,
)

__all__ = [
    "as_2d",
    "per_row_error",
    "resolve_feature_index",
    "subsample_idx",
]
