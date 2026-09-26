"""Cross-package API of ``mlframe.feature_engineering``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_engineering`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from .basic import (  # noqa: F401
    _DEFAULT_CYCLICAL_PERIODS as DEFAULT_CYCLICAL_PERIODS,
    _DEFAULT_DATE_METHODS as DEFAULT_DATE_METHODS,
)
from .graph_spectral_features import (  # noqa: F401
    _dense_adjacency as dense_adjacency,
)
from .numerical import (  # noqa: F401
    _astropy_histogram as astropy_histogram,
    _resolve_astropy_histogram as resolve_astropy_histogram,
)

__all__ = [
    "DEFAULT_CYCLICAL_PERIODS",
    "DEFAULT_DATE_METHODS",
    "astropy_histogram",
    "dense_adjacency",
    "resolve_astropy_histogram",
]
