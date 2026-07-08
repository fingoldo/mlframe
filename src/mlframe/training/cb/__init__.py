"""CatBoost Pool / GPU helpers subsystem.

Groups the CatBoost-specific training utilities carved out of
``trainer.py``: Pool building + caching, GPU usability probing, Polars
nullable-categorical diagnostics / filling, and the predict-with-fallback
path.

- ``_cb_pool`` -- the bulk of the helpers + the train/val Pool caches.
- ``_cb_pool_build`` -- ``_maybe_get_or_build_cb_pool`` (cached Pool
  construction), re-exported from ``_cb_pool``'s bottom.

The surface external code imports is re-exported here so importers resolve
from the documented package path instead of a deep ``_cb_pool`` private
module.
"""
from __future__ import annotations

from ._cb_pool import (
    _predict_with_fallback,
    _maybe_get_or_build_cb_pool,
    _maybe_rewrite_eval_set_as_cb_pool,
    _cached_gpu_info,
    _cb_gpu_usable,
    _polars_df_has_null_in_categorical,
    _polars_fill_null_in_categorical,
    _polars_nullable_categorical_cols,
    _polars_schema_diagnostic,
    _CB_POOL_CACHE,
)

__all__ = [
    "_predict_with_fallback",
    "_maybe_get_or_build_cb_pool",
    "_maybe_rewrite_eval_set_as_cb_pool",
    "_cached_gpu_info",
    "_cb_gpu_usable",
    "_CB_POOL_CACHE",
]
