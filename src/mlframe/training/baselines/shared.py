"""Cross-package API of ``mlframe.training.baselines``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.baselines`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (  # noqa: F401
    _warmup_numba_kernels as warmup_numba_kernels,
)

__all__ = [
    "warmup_numba_kernels",
]
