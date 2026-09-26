"""Cross-package API of ``mlframe.training.composite.ensemble``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.composite.ensemble`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (
    _is_monotone_nondecreasing as is_monotone_nondecreasing,
    _maybe_pass_sample_weight as maybe_pass_sample_weight,
)
from ._oof_split import (
    _slice_rows as slice_rows,
)

__all__ = [
    "is_monotone_nondecreasing",
    "maybe_pass_sample_weight",
    "slice_rows",
]
