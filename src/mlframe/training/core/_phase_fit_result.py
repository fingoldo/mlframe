"""Return shape of ``_phase_fit_pipeline``, kept in a leaf module.

Leaf module: ``_phase_helpers_fit_pipeline`` builds this type and ``_phase_helpers_fit_split`` re-exports it,
so neither imports the other for it.
"""

from typing import Any, NamedTuple


class FitPipelineResult(NamedTuple):
    """Return shape for ``_phase_fit_pipeline`` (see comment on
    ``TrainValTestSplitResult``)."""

    train_df: Any
    val_df: Any
    test_df: Any
    pipeline: Any
    extensions_pipeline: Any
    cat_features: Any
    cat_features_polars: Any
    was_polars_input: Any
    all_models_polars_native: Any
    polars_pipeline_applied: Any
    train_df_polars_pre: Any
    val_df_polars_pre: Any
    test_df_polars_pre: Any
    pipeline_config: Any
    preprocessing_extensions: Any
    train_df_pandas_pre_meta: Any
