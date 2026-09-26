"""Cross-package API of ``mlframe.feature_selection.filters.hermite_fe``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection.filters.hermite_fe`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (
    _BASIS_BUILDERS as BASIS_BUILDERS,
    _CUDA_AVAILABLE as CUDA_AVAILABLE,
    _CUDA_THRESHOLD as CUDA_THRESHOLD,
    _DEFAULT_BIN_FUNCS as DEFAULT_BIN_FUNCS,
    _L2_PENALTY_SATURATION_DEFAULT as L2_PENALTY_SATURATION_DEFAULT,
    _NJIT_FUNCS as NJIT_FUNCS,
    _NJIT_PAR_FUNCS as NJIT_PAR_FUNCS,
    _PAR_THRESHOLD as PAR_THRESHOLD,
    _POLY_BASES as POLY_BASES,
    _chebval_njit as chebval_njit,
    _hermeval_njit as hermeval_njit,
    _lagval_njit as lagval_njit,
    _legval_njit as legval_njit,
    _plugin_mi_classif_batch_cuda_resident as plugin_mi_classif_batch_cuda_resident,
    _plugin_mi_classif_batch_njit as plugin_mi_classif_batch_njit,
    _plugin_mi_classif_batch_rows_njit as plugin_mi_classif_batch_rows_njit,
    _plugin_mi_classif_cuda as plugin_mi_classif_cuda,
    _plugin_mi_classif_njit as plugin_mi_classif_njit,
    _plugin_mi_from_binned_njit as plugin_mi_from_binned_njit,
    _plugin_mi_regression_batch_njit as plugin_mi_regression_batch_njit,
    _plugin_mi_regression_njit as plugin_mi_regression_njit,
    _quantile_bin_njit as quantile_bin_njit,
    _quantile_bin_numpy as quantile_bin_numpy,
)
from ._hermite_prewarp import (
    _canonical_seeds as canonical_seeds,
    _l2_normalize_pair as l2_normalize_pair,
    _l2_penalty_value as l2_penalty_value,
)
from ._hermite_robust import (
    _detect_heavy_tail as detect_heavy_tail,
    _robust_axis_enabled as robust_axis_enabled,
    _robust_lo_hi as robust_lo_hi,
    fit_basis_coef_robust,
    heavy_tail_memo_scope,
)
from . import (
    HermiteResult,
    build_basis_matrix,
    warm_start_als_seed,
)

__all__ = [
    "HermiteResult",
    "build_basis_matrix",
    "warm_start_als_seed",
    "BASIS_BUILDERS",
    "CUDA_AVAILABLE",
    "CUDA_THRESHOLD",
    "DEFAULT_BIN_FUNCS",
    "L2_PENALTY_SATURATION_DEFAULT",
    "NJIT_FUNCS",
    "NJIT_PAR_FUNCS",
    "PAR_THRESHOLD",
    "POLY_BASES",
    "canonical_seeds",
    "chebval_njit",
    "detect_heavy_tail",
    "fit_basis_coef_robust",
    "heavy_tail_memo_scope",
    "hermeval_njit",
    "l2_normalize_pair",
    "l2_penalty_value",
    "lagval_njit",
    "legval_njit",
    "plugin_mi_classif_batch_cuda_resident",
    "plugin_mi_classif_batch_njit",
    "plugin_mi_classif_batch_rows_njit",
    "plugin_mi_classif_cuda",
    "plugin_mi_classif_njit",
    "plugin_mi_from_binned_njit",
    "plugin_mi_regression_batch_njit",
    "plugin_mi_regression_njit",
    "quantile_bin_njit",
    "quantile_bin_numpy",
    "robust_axis_enabled",
    "robust_lo_hi",
]
