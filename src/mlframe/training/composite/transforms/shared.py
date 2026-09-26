"""Cross-package API of ``mlframe.training.composite.transforms``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.composite.transforms`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._call_gateway import (  # noqa: F401
    call_transform,
)
from .linear import (  # noqa: F401
    _linear_residual_fit as linear_residual_fit,
    _linear_residual_fit_batched as linear_residual_fit_batched,
    _linear_residual_multi_fit as linear_residual_multi_fit,
)
from .nonlinear import (  # noqa: F401
    _make_chain_transform as make_chain_transform,
)
from .registry import (  # noqa: F401
    _TRANSFORMS_REGISTRY as TRANSFORMS_REGISTRY,
)
from .simple import (  # noqa: F401
    _diff_inverse as diff_inverse,
)

__all__ = [
    "TRANSFORMS_REGISTRY",
    "call_transform",
    "diff_inverse",
    "linear_residual_fit",
    "linear_residual_fit_batched",
    "linear_residual_multi_fit",
    "make_chain_transform",
]
