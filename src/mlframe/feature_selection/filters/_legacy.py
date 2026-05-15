"""Legacy BC shim. Re-exports every public name from the (now-decomposed) monolith so ``from mlframe.feature_selection.filters._legacy import X`` keeps working."""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
warnings.filterwarnings("ignore", module=".*_discretization")
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

from ._internals import (
    ENSURE_ARROW_DF_SUPPORT,
    GPU_MAX_BLOCK_SIZE,
    LARGE_CONST,
    MAX_CONFIRMATION_CAND_NBINS,
    MAX_ITERATIONS_TO_TRACK,
    MAX_JOBLIB_NBYTES,
    NMAX_NONPARALLEL_ITERS,
    njit_functions_dict,
    sanitize,
    smart_log,
)
from ._numba_utils import arr2str, count_cand_nbins, unpack_and_sort

from .discretization import (
    _discretize_array_impl,
    categorize_1d_array,
    categorize_dataset,
    create_redundant_continuous_factor,
    digitize,
    discretize_2d_array,
    discretize_array,
    discretize_sklearn,
    discretize_uniform,
    edges,
    get_binning_edges,
    quantize_dig,
    quantize_search,
)
from .info_theory import (
    compute_mi_from_classes,
    conditional_mi,
    entropy,
    merge_vars,
    mi,
)
from .permutation import (
    distribute_permutations,
    mi_direct,
    parallel_mi,
    shuffle_arr,
)
from .gpu import init_kernels, mi_direct_gpu

from .evaluation import (
    evaluate_candidate,
    evaluate_candidates,
    evaluate_gain,
    find_best_partial_gain,
    get_candidate_name,
    handle_best_candidate,
    should_skip_candidate,
)
from .fleuret import (
    get_fleuret_criteria_confidence,
    get_fleuret_criteria_confidence_parallel,
    parallel_fleuret,
)
from .feature_engineering import (
    check_prospective_fe_pairs,
    compute_pairs_mis,
    create_binary_transformations,
    create_unary_transformations,
    get_existing_feature_name,
    get_new_feature_name,
)
from .screen import postprocess_candidates, screen_predictors
from .mrmr import MRMR

# Explicit re-export surface so static analysers (ruff F401) see them as intended public exports rather than unused imports.
__all__ = [
    "ENSURE_ARROW_DF_SUPPORT", "GPU_MAX_BLOCK_SIZE", "LARGE_CONST", "MAX_CONFIRMATION_CAND_NBINS", "MAX_ITERATIONS_TO_TRACK", "MAX_JOBLIB_NBYTES",
    "NMAX_NONPARALLEL_ITERS", "njit_functions_dict", "sanitize", "smart_log",
    "arr2str", "count_cand_nbins", "unpack_and_sort",
    "_discretize_array_impl", "categorize_1d_array", "categorize_dataset", "create_redundant_continuous_factor", "digitize", "discretize_2d_array",
    "discretize_array", "discretize_sklearn", "discretize_uniform", "edges", "get_binning_edges", "quantize_dig", "quantize_search",
    "compute_mi_from_classes", "conditional_mi", "entropy", "merge_vars", "mi",
    "distribute_permutations", "mi_direct", "parallel_mi", "shuffle_arr",
    "init_kernels", "mi_direct_gpu",
    "evaluate_candidate", "evaluate_candidates", "evaluate_gain", "find_best_partial_gain", "get_candidate_name", "handle_best_candidate", "should_skip_candidate",
    "get_fleuret_criteria_confidence", "get_fleuret_criteria_confidence_parallel", "parallel_fleuret",
    "check_prospective_fe_pairs", "compute_pairs_mis", "create_binary_transformations", "create_unary_transformations", "get_existing_feature_name", "get_new_feature_name",
    "postprocess_candidates", "screen_predictors",
    "MRMR",
]
