"""Cross-package API of ``mlframe.feature_selection.filters.mrmr``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.feature_selection.filters.mrmr`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (
    _MRMR_BATCH_PRECOMPUTE_MIN_PAIRS as MRMR_BATCH_PRECOMPUTE_MIN_PAIRS,
    _content_array_signature as content_array_signature,
    _full_x_content_hash as full_x_content_hash,
    _full_y_content_hash as full_y_content_hash,
    _hashable_params_signature as hashable_params_signature,
    _lazy_chunks as lazy_chunks,
    _replay_fitted_state as replay_fitted_state,
    _target_name_signature as target_name_signature,
    _target_to_numpy_values as target_to_numpy_values,
)
from ._mrmr_param_constants import (
    _VALID_FE_HYBRID_ORTH_BASES as VALID_FE_HYBRID_ORTH_BASES,
    _VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS as VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS,
    _VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS as VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS,
    _VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS as VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS,
    _VALID_FE_HYBRID_ORTH_HSIC_KERNELS as VALID_FE_HYBRID_ORTH_HSIC_KERNELS,
    _VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS as VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS,
)

__all__ = [
    "MRMR_BATCH_PRECOMPUTE_MIN_PAIRS",
    "VALID_FE_HYBRID_ORTH_BASES",
    "VALID_FE_HYBRID_ORTH_CLUSTER_BASIS_AGGREGATORS",
    "VALID_FE_HYBRID_ORTH_ENSEMBLE_AGGREGATORS",
    "VALID_FE_HYBRID_ORTH_ENSEMBLE_SCORERS",
    "VALID_FE_HYBRID_ORTH_HSIC_KERNELS",
    "VALID_FE_HYBRID_ORTH_META_FORCE_SCORERS",
    "content_array_signature",
    "full_x_content_hash",
    "full_y_content_hash",
    "hashable_params_signature",
    "lazy_chunks",
    "replay_fitted_state",
    "target_name_signature",
    "target_to_numpy_values",
]
