"""Cross-package API of ``mlframe.feature_selection.filters.engineered_recipes``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection.filters.engineered_recipes`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._orth_basis_recipes import (
    _apply_orth_pre_transform as apply_orth_pre_transform,
    _bspline_basis_values as bspline_basis_values,
    _eval_orth_basis_column as eval_orth_basis_column,
    _fit_spline_knots as fit_spline_knots,
    _freeze_preprocess_params as freeze_preprocess_params,
)
from ._recipe_dispatch import (
    apply_recipe,
)
from ._recipe_extract import (
    _extract_column as extract_column,
    build_category_code_map,
)
from ._recipe_name_simplify import (
    simplified_recipe_names,
    simplify_fe_name,
)

__all__ = [
    "apply_orth_pre_transform",
    "apply_recipe",
    "bspline_basis_values",
    "build_category_code_map",
    "eval_orth_basis_column",
    "extract_column",
    "fit_spline_knots",
    "freeze_preprocess_params",
    "simplified_recipe_names",
    "simplify_fe_name",
]
