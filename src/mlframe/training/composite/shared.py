"""Cross-package API of ``mlframe.training.composite``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.composite`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._estimator_dispatch import (  # noqa: F401
    maybe_inject_distribution_driven_estimator,
    recommend_composite_estimator,
)
from ._frame_ops import (  # noqa: F401
    append_column,
)
from ._hurdle_dispatch import (  # noqa: F401
    maybe_inject_hurdle_for_zero_inflated,
)
from ._row_roles import (  # noqa: F401
    note_rows,
)
from ._synthetic_bases import (  # noqa: F401
    is_resolvable,
    parse_synthetic,
    synthetic_column,
)
from .conformal import (  # noqa: F401
    _fit_sigma_model as fit_sigma_model,
    _sigma_for as sigma_for,
)

__all__ = [
    "append_column",
    "fit_sigma_model",
    "is_resolvable",
    "maybe_inject_distribution_driven_estimator",
    "maybe_inject_hurdle_for_zero_inflated",
    "note_rows",
    "parse_synthetic",
    "recommend_composite_estimator",
    "sigma_for",
    "synthetic_column",
]
