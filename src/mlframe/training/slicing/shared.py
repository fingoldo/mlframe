"""Cross-package API of ``mlframe.training.slicing``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.slicing`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._slice_helpers import (  # noqa: F401
    _is_classification_target as is_classification_target,
)

__all__ = [
    "is_classification_target",
]
