"""Cross-package API of ``mlframe.training.neural.base``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.neural.base`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (
    _ensure_numpy as ensure_numpy,
)

__all__ = [
    "ensure_numpy",
]
