"""Cross-package API of ``mlframe.training.callbacks``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.callbacks`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from .monotonic_decline import (  # noqa: F401
    _make_xgb_monotonic_callback as make_xgb_monotonic_callback,
)

__all__ = [
    "make_xgb_monotonic_callback",
]
