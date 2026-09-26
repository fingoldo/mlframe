"""Cross-package API of ``mlframe.training.cb``: the helpers other packages may import, under public names.

The implementations stay in their (private) home modules; code outside ``mlframe.training.cb`` imports them from here, so a
helper can move inside the package without breaking its users, and the private-import gate stays meaningful.
"""

from . import (  # noqa: F401
    _CB_POOL_CACHE as CB_POOL_CACHE,
)
from ._cb_eval_weights import (  # noqa: F401
    CB_EVAL_WEIGHTS_KEY,
    apply_cb_eval_sample_weights,
)
from ._cb_gpu_budget import (  # noqa: F401
    fit_with_cb_gpu_guard,
)
from ._cb_polars_text import (  # noqa: F401
    cb_text_features_as_strings,
)
from ._cb_pool import (  # noqa: F401
    _cached_gpu_info as cached_gpu_info,
    _cb_gpu_usable as cb_gpu_usable,
    _maybe_rewrite_eval_set_as_cb_pool as maybe_rewrite_eval_set_as_cb_pool,
    _polars_df_has_null_in_categorical as polars_df_has_null_in_categorical,
    _polars_fill_null_in_categorical as polars_fill_null_in_categorical,
    _polars_nullable_categorical_cols as polars_nullable_categorical_cols,
    _polars_schema_diagnostic as polars_schema_diagnostic,
    _predict_with_fallback as predict_with_fallback,
    _recover_cb_feature_names as recover_cb_feature_names,
)
from ._cb_pool_build import (  # noqa: F401
    _maybe_get_or_build_cb_pool as maybe_get_or_build_cb_pool,
)

__all__ = [
    "CB_EVAL_WEIGHTS_KEY",
    "CB_POOL_CACHE",
    "apply_cb_eval_sample_weights",
    "cached_gpu_info",
    "cb_gpu_usable",
    "cb_text_features_as_strings",
    "fit_with_cb_gpu_guard",
    "maybe_get_or_build_cb_pool",
    "maybe_rewrite_eval_set_as_cb_pool",
    "polars_df_has_null_in_categorical",
    "polars_fill_null_in_categorical",
    "polars_nullable_categorical_cols",
    "polars_schema_diagnostic",
    "predict_with_fallback",
    "recover_cb_feature_names",
]
