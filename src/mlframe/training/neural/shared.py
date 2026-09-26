"""Cross-package API of ``mlframe.training.neural``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.neural`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._recurrent_config import (  # noqa: F401
    RecurrentConfig,
)

__all__ = [
    "RecurrentConfig",
]
