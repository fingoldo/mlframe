"""Cross-package API of ``mlframe.utils``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.utils`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from .rng_scope import (
    _fresh_seed as fresh_seed,
)
from .safe_pickle import (
    _sha256_of_file as sha256_of_file,
)

__all__ = [
    "fresh_seed",
    "sha256_of_file",
]
