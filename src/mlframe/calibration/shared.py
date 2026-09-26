"""Cross-package API of ``mlframe.calibration``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.calibration`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from .policy import (  # noqa: F401
    _ece_score as ece_score,
    _ece_score_numba_serial as ece_score_numba_serial,
)

__all__ = [
    "ece_score",
    "ece_score_numba_serial",
]
