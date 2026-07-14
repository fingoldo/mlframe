"""Cache-key and input-fingerprint helpers used by ``_train_one_target``.

Carved out of ``_phase_train_one_target_body.py`` to keep it under the repo's
1k-LOC-per-file limit. Both helpers are pure with respect to their explicit
arguments, aside from mutating the caller-supplied ``ctx`` cache dicts.
"""

from __future__ import annotations

from typing import Any, Callable


def compute_model_pipeline_cache_key(
    strategy,
    pre_pipeline_name: str,
    cat_features,
    text_features,
    embedding_features,
    train_df_polars,
    cur_target_name,
    current_train_target,
    _compute_pipeline_cache_key: Callable[..., str],
) -> str:
    """Build the per-(strategy, pre_pipeline, tier, kind, features) pipeline cache key.

    Uses a CONTENT-based key derived from the preprocessing-requirements tuple
    instead of ``strategy.cache_key`` (name-based), so two strategies that
    consume identical ``imp+scaler`` pipelines share the same cache slot even
    when their ``cache_key`` names differ (e.g. Linear vs MLP on an all-numeric
    frame). The encoding bit is the EFFECTIVE one (``requires_encoding`` AND
    there are cats to encode), since a strategy that only target-encodes
    differs from a learnable-cat-embedding strategy exclusively when cat
    columns actually exist.
    """
    _effective_enc = bool(getattr(strategy, "requires_encoding", False)) and bool(cat_features)
    _content_key = f"imp{int(getattr(strategy, 'requires_imputation', False))}" f"_scale{int(getattr(strategy, 'requires_scaling', False))}" f"_enc{int(_effective_enc)}"
    # feature_tier = (supports_text, supports_embedding) segments the trimmed frame per
    # text/embedding-support level. When the data carries NO text/embedding columns there is
    # nothing to trim, so every tier yields the IDENTICAL frame -- collapse the tier to a neutral
    # value so two strategies with matching imp+scale+enc share the slot instead of each
    # re-running the pre_pipeline.
    _effective_tier = strategy.feature_tier() if (text_features or embedding_features) else (False, False)
    _cache_key_train_df = train_df_polars if strategy.supports_polars else None
    return str(
        _compute_pipeline_cache_key(
            _content_key,
            pre_pipeline_name,
            _effective_tier,
            strategy.supports_polars,
            cat_features,
            text_features,
            embedding_features,
            train_df=_cache_key_train_df,
            target_name=cur_target_name,
            train_target=current_train_target,
        )
    )


def compute_cached_model_input_fingerprint(
    ctx,
    polars_fastpath_active: bool,
    prepared_train,
    tier_pandas: dict[str, Any],
    strategy,
    pre_pipeline_name: str,
    cat_features,
    text_features,
    embedding_features,
    compute_model_input_fingerprint,
):
    """Return ``(schema_hash, input_schema)`` for the current (strategy, tier, kind, pp_name) combo.

    Cached on ``ctx._model_input_fingerprint_cache`` per (model, pre_pipeline) so
    it's computed once outside the weight loop, where only ``sample_weight``
    changes across iterations. The key folds ``id(train_df)`` (safe only because
    the frame is strong-ref-pinned at this point) plus column count; on a hit,
    the cached schema's column names are re-checked against the live frame so a
    GC-recycled ``id`` collision can never silently replay a stale fingerprint.
    """
    _fp_train_df_pre = prepared_train if polars_fastpath_active else tier_pandas["train_df"]
    _fp_train_df_id = id(_fp_train_df_pre) if _fp_train_df_pre is not None else 0
    _fp_train_df_ncols = len(_fp_train_df_pre.columns) if _fp_train_df_pre is not None and hasattr(_fp_train_df_pre, "columns") else 0
    _fp_cache_key = (
        id(strategy),
        strategy.feature_tier(),
        strategy.supports_polars,
        pre_pipeline_name,
        _fp_train_df_id,
        _fp_train_df_ncols,
    )
    _cs_fp = ctx._cache_stats.setdefault("fingerprint_cache", {"hits": 0, "misses": 0})
    _fp_cached = ctx._model_input_fingerprint_cache.get(_fp_cache_key)
    if _fp_cached is not None and _fp_train_df_pre is not None and hasattr(_fp_train_df_pre, "columns"):
        _live_cols = list(_fp_train_df_pre.columns)
        _cached_cols = [rec.get("name") for rec in _fp_cached[1]] if _fp_cached[1] else []
        if sorted(str(c) for c in _cached_cols) != sorted(str(c) for c in _live_cols):
            _fp_cached = None
    if _fp_cached is not None:
        _cs_fp["hits"] += 1
        _schema_hash, _input_schema = _fp_cached
    else:
        _cs_fp["misses"] += 1
        _schema_hash, _input_schema = compute_model_input_fingerprint(
            _fp_train_df_pre,
            cat_features=cat_features,
            text_features=text_features,
            embedding_features=embedding_features,
        )
        ctx._model_input_fingerprint_cache[_fp_cache_key] = (_schema_hash, _input_schema)
    return _schema_hash, _input_schema
