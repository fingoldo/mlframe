"""Cross-package API of ``mlframe.feature_engineering.transformer``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_engineering.transformer`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._knn_helper import (
    knn_search,
)
from ._suite_adapter import (
    ShortlistTransformerAdapter,
)

__all__ = [
    "ShortlistTransformerAdapter",
    "knn_search",
]
