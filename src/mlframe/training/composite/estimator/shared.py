"""Cross-package API of ``mlframe.training.composite.estimator``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.composite.estimator`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (  # noqa: F401
    _extract_base as extract_base,
    _extract_base_matrix as extract_base_matrix,
    _extract_groups as extract_groups,
    _to_1d_numpy as to_1d_numpy,
    _y_train_clip_bounds as y_train_clip_bounds,
)
from ._predict import (  # noqa: F401
    _apply_t_clip as apply_t_clip,
)
from ._routing import (  # noqa: F401
    ensure_transforms_registered,
    inner_input,
)
from ._smearing import (  # noqa: F401
    N_SMEAR_QUANTILES,
    SMEARED_TRANSFORMS,
    smeared_inverse,
    smeared_prediction,
)
from ._soft_shrink import (  # noqa: F401
    BASE_FIT_RANGE_KEY,
)

__all__ = [
    "BASE_FIT_RANGE_KEY",
    "N_SMEAR_QUANTILES",
    "SMEARED_TRANSFORMS",
    "apply_t_clip",
    "ensure_transforms_registered",
    "extract_base",
    "extract_base_matrix",
    "extract_groups",
    "inner_input",
    "smeared_inverse",
    "smeared_prediction",
    "to_1d_numpy",
    "y_train_clip_bounds",
]
