"""Cross-package API of ``mlframe.training.composite.discovery``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.composite.discovery`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from ._algo_version import (
    DISCOVERY_ALGO_VERSION,
)
from ._auto_chain import (
    reregister_auto_chain_transforms,
)
from ._base_engineering import (
    engineer_temporal_bases,
)
from ._causal_lag import (
    CAUSAL_LAG_SUFFIXES,
    causal_lag_predict_rmse,
    detect_causal_lag_column,
)
from ._leakage import (
    detect_base_target_leakage,
)
from ._point_mass_gate import (
    MIN_ROWS_FOR_POINT_MASS_CHECK,
    POINT_MASS_FRACTION_THRESHOLD,
)
from ._score import (
    Score,
    rank_specs,
)
from ._screening_tiny import (
    _build_tiny_model as build_tiny_model,
)
from ._splitter import (
    make_discovery_splitter,
)
from ._stability import (
    stability_select_specs,
)
from ._structural_hints import (
    structural_affinity_scores,
)
from ._t_equivalence import (
    DEFAULT_R2_TOL,
    find_equivalent_composite_specs,
    t_train_envelope,
)
from .screening import (
    _extract_column_array as extract_column_array,
    _is_numeric_column as is_numeric_column,
)

__all__ = [
    "CAUSAL_LAG_SUFFIXES",
    "DEFAULT_R2_TOL",
    "DISCOVERY_ALGO_VERSION",
    "MIN_ROWS_FOR_POINT_MASS_CHECK",
    "POINT_MASS_FRACTION_THRESHOLD",
    "Score",
    "build_tiny_model",
    "causal_lag_predict_rmse",
    "detect_base_target_leakage",
    "detect_causal_lag_column",
    "engineer_temporal_bases",
    "extract_column_array",
    "find_equivalent_composite_specs",
    "is_numeric_column",
    "make_discovery_splitter",
    "rank_specs",
    "reregister_auto_chain_transforms",
    "stability_select_specs",
    "structural_affinity_scores",
    "t_train_envelope",
]
