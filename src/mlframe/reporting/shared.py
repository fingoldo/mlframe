"""Cross-package API of ``mlframe.reporting``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.reporting`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from .diagnostics_dispatch import (
    _record_skipped as record_skipped,
)

__all__ = [
    "record_skipped",
]
