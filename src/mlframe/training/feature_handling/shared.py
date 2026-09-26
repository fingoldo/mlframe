"""Cross-package API of ``mlframe.training.feature_handling``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.feature_handling`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from .polynomial import (  # noqa: F401
    _projected_output_cols as projected_output_cols,
)

__all__ = [
    "projected_output_cols",
]
