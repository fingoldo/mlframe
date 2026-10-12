"""Cache-key and input-fingerprint helpers used by ``_train_one_target``.

Carved out of ``_phase_train_one_target_body.py`` to keep it under the repo's
1k-LOC-per-file limit. Both helpers are pure with respect to their explicit
arguments, aside from mutating the caller-supplied ``ctx`` cache dicts.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


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
    train_df_pd=None,
    pre_pipeline_digest: str = "",
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
    # LIVE BUG (found+fixed together): for a non-polars strategy this used to pass ``train_df=None``
    # unconditionally ("pandas frames don't reach this branch typed-distinct enough to need the
    # suffix, handled upstream in split_features" -- that upstream claim doesn't hold for the plain
    # no-pre_pipeline case). With no pre_pipeline (pre_pipeline_name=None) the target discriminator
    # inside _compute_pipeline_cache_key is ALSO gated off (by design, to preserve legitimate
    # same-X-multi-target cache sharing -- see test_pipeline_cache_key_target_discrimination.py), so
    # the resulting key had ZERO content discriminator at all for a plain pandas tree-model fit:
    # confirmed live, two ENTIRELY UNRELATED train_mlframe_models_suite calls (different X shape --
    # 11 cols vs 3 cols, different targets) both produced the identical key
    # "lgb_tier(False, False)_kindpd_feats69dda5e0bd93b362", and PipelineCache served the first
    # call's fitted/cached artefacts to the second, producing a Booster trained on the WRONG X that
    # crashed at predict time with LightGBMError: "number of features in data (3) is not the same as
    # it was in training data (11)" (test_registered_composite_model_keys_suite.py::
    # test_lgb_string_key_dispatch_unaffected_by_gated_outlier_registration). Falling back to the
    # pandas train_df (already available on ctx as train_df_pd) when the strategy isn't polars-native
    # lets _compute_pipeline_cache_key's existing, already backend-agnostic ``_dtype_suffix`` logic
    # (_canonical_dtype_pairs_compute handles pandas via a generic hasattr(columns) path) fold a real
    # schema fingerprint in for pandas paths too -- closing the gap without touching the
    # pre_pipeline_name-gated target-suffix logic at all.
    _cache_key_train_df = train_df_polars if strategy.supports_polars else train_df_pd
    # The key holds no row identity: a target trained on its labelled rows only would be served the pipeline fitted on
    # every row. Its rows' signature goes into the key; a fully labelled target's key is unchanged.
    from ._target_row_scope import active_rows

    rows = active_rows()
    rows_suffix = f"_rows{rows.signature}" if rows is not None else ""
    key = str(
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
            pre_pipeline_digest=pre_pipeline_digest,
        )
    )
    return key + rows_suffix


def pre_pipeline_params_digest(pre_pipeline: Any) -> str:
    """Digest of every step's ``get_params(deep=False)`` in a fitted or unfitted pre-pipeline; "" for no pre-pipeline or any failure.

    A pre-pipeline's name does not say which parameters it was built with, so two pipelines of the same name and different
    parameters (an MRMR with the target mix on and off) would otherwise share a cache slot. ``repr`` of the sorted items is
    stable for the primitive parameters the selectors use; a failure leaves the key unchanged rather than blocking training."""
    if pre_pipeline is None:
        return ""
    try:
        steps = getattr(pre_pipeline, "steps", None)
        items = [(name, est) for name, est in steps] if steps is not None else [("", pre_pipeline)]
        parts = []
        for name, est in items:
            params = est.get_params(deep=False) if hasattr(est, "get_params") else {}
            parts.append((str(name), type(est).__name__, sorted(params.items(), key=lambda kv: kv[0])))
        return hashlib.blake2b(repr(parts).encode("utf-8", errors="replace"), digest_size=8).hexdigest()
    except Exception as exc:
        # A failed introspection must not share one constant slot across every failing pipeline: key on the object itself, so
        # two different un-introspectable pipelines never reuse each other's cache entry (an object repr carries its identity).
        logger.debug("pre_pipeline_params_digest: introspection failed, keying on the object instead: %s", exc)
        return hashlib.blake2b(repr(pre_pipeline).encode("utf-8", errors="replace"), digest_size=8).hexdigest()


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
