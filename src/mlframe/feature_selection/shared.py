"""Cross-package API of ``mlframe.feature_selection``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from .varying_size_top_k_subsets import (  # noqa: F401
    _cluster_anchors as cluster_anchors,
)

__all__ = [
    "cluster_anchors",
]
