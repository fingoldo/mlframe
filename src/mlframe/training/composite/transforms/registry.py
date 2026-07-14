"""Transforms registry carved out of
``mlframe.training.composite_transforms``.

Builds the ``_TRANSFORMS_REGISTRY`` dict mapping transform name -> :class:`Transform`. The four functional clusters (simple / linear / nonlinear / unary) define the underlying fit/forward/inverse/domain functions; ``_registry_setup.py`` wires them into per-unary adapters, and ``_registry_part1.py`` / ``_registry_part2.py`` hold the two halves of the dict literal itself (split to keep every file under the 1k-line monolith threshold).

Imported by the parent AFTER all four functional siblings load (init-order matters: the registry literal references every per-transform function by binding-resolution at module-import time).

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.composite_transforms import _TRANSFORMS_REGISTRY`` resolves.
"""
from __future__ import annotations

from . import Transform
from ._registry_setup import (
    _bc_domain_a,
    _bc_domain_fitted_a,
    _bc_fit_a,
    _bc_forward_a,
    _bc_inverse_a,
    _cbrt_domain,
    _cbrt_domain_fitted,
    _cbrt_fit,
    _cbrt_forward,
    _cbrt_inverse,
    _centered_ratio_domain_fitted,
    _log_domain_a,
    _log_domain_fitted_a,
    _log_fit_a,
    _log_forward_a,
    _log_inverse_a,
    _make_unary_registry_adapter,
    _qn_domain_a,
    _qn_domain_fitted_a,
    _qn_fit_a,
    _qn_forward_a,
    _qn_inverse_a,
    _sp_domain_a,
    _sp_domain_fitted_a,
    _sp_fit_a,
    _sp_forward_a,
    _sp_inverse_a,
    _yj_domain_a,
    _yj_domain_fitted_a,
    _yj_fit_a,
    _yj_forward_a,
    _yj_inverse_a,
)
from ._registry_part1 import _REGISTRY_PART1
from ._registry_part2 import _REGISTRY_PART2

__all__ = [
    "_TRANSFORMS_REGISTRY",
    "_make_unary_registry_adapter",
    "_cbrt_fit",
    "_cbrt_forward",
    "_cbrt_inverse",
    "_cbrt_domain",
    "_cbrt_domain_fitted",
    "_log_fit_a",
    "_log_forward_a",
    "_log_inverse_a",
    "_log_domain_a",
    "_log_domain_fitted_a",
    "_yj_fit_a",
    "_yj_forward_a",
    "_yj_inverse_a",
    "_yj_domain_a",
    "_yj_domain_fitted_a",
    "_qn_fit_a",
    "_qn_forward_a",
    "_qn_inverse_a",
    "_qn_domain_a",
    "_qn_domain_fitted_a",
    "_sp_fit_a",
    "_sp_forward_a",
    "_sp_inverse_a",
    "_sp_domain_a",
    "_sp_domain_fitted_a",
    "_bc_fit_a",
    "_bc_forward_a",
    "_bc_inverse_a",
    "_bc_domain_a",
    "_bc_domain_fitted_a",
    "_centered_ratio_domain_fitted",
]

_TRANSFORMS_REGISTRY: dict[str, Transform] = {**_REGISTRY_PART1, **_REGISTRY_PART2}
