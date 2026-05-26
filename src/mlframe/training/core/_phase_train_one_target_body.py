"""``_train_one_target`` body carved out of ``mlframe.training.core._phase_train_one_target``.

Holds only the main per-target training entry point. Helpers used by the
function stay in the parent module and are lazily imported inside the
function body so the parent-bottom re-export doesn't create a hard import
cycle (``test_no_import_cycles`` walks top-level imports only).

Re-imported at the parent's module bottom so historical
``from ._phase_train_one_target import _train_one_target`` resolves
transparently.
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any

import numpy as np

try:
    import psutil as _ps_module  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _ps_module = None  # type: ignore[assignment]

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from sklearn.base import clone

from pyutilz.system import tqdmu_lazy_start

from ..configs import TargetTypes as _TargetTypes
from ..phases import phase
from ..models import is_neural_model
from ..strategies import PipelineCache
from ..train_eval import process_model
from ..utils import compute_model_input_fingerprint, filter_existing
from ._misc_helpers import (
    _compute_neural_max_time, _elapsed_str,
    _filter_polars_cat_features_by_dtype, _maybe_clear_shim_cache,
)
from ._setup_helpers import (
    _build_process_model_kwargs,
    _should_skip_catboost_metamodel,
)
from ._phase_train_one_target_ensembling import _finalize_per_target_ensembling
from ._phase_train_one_target_polars_fastpath import _prepare_strategy_inputs
from ._phase_train_one_target_pre_screen import _maybe_run_unsupervised_pre_screen
from ._phase_train_one_target_model_setup import _setup_per_target_mlframe_models
from ._phase_train_one_target_schema import (
    _build_and_record_model_schema,
    _clone_model_with_sticky_flags,
    _resolve_weight_schemas_and_warn_val_placement,
)
from ._phase_train_one_target_mlp_helpers import (
    _apply_mlp_extreme_ar_output_activation,
    _apply_mlp_extreme_ar_weight_decay_bump,
    _drop_columns_for_mlp,
    _identify_per_group_columns,
)

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _train_one_target(ctx, target_type, targets, cur_target_name, cur_target_values):
    """Train all models for one (target_type, target_name) pair."""
    # Lazy import: ``._phase_train_one_target`` re-imports this sibling at
    # its module bottom for re-export. Top-level ``from ._phase_train_one_target
    # import ...`` would create a hard import cycle. Python module cache makes
    # repeat imports cheap; the dict lookup is sub-microsecond per call.
    from ._phase_train_one_target import (
        _apply_loss_recommendation_in_place,
        _build_feature_selection_report,
        _cached_init_params,
        _capture_dataset_reuse_cache,
        _compute_pipeline_cache_key,
        _ensure_feature_side_cache,
        _forward_dataset_reuse_cache,
        _invalidate_polars_feature_side_cache,
        _is_regression_target_type,
        _maybe_run_feature_handling_apply,
        _release_ctx_polars_frames,
        _restore_dataset_reuse_cache,
        _selector_params_hash,
        _unwrap_selector,
    )
    from ._phase_train_one_target import slugify
    _maybe_run_unsupervised_pre_screen(ctx, targets)
    split_config = ctx.split_config
    behavior_config = ctx.behavior_config
    feature_selection_config = ctx.feature_selection_config
    verbose = ctx.verbose
    use_ordinary_models = ctx.use_ordinary_models
    use_mlframe_ensembles = ctx.use_mlframe_ensembles
    metadata = ctx.metadata
    sample_weights = ctx.sample_weights
    baseline_rss_mb = ctx.baseline_rss_mb
    df_size_mb = ctx.df_size_mb
    pipeline = ctx.pipeline
    polars_pipeline_applied = ctx.polars_pipeline_applied
    cat_features = ctx.cat_features
    text_features = ctx.text_features
    embedding_features = ctx.embedding_features
    train_df_pd = ctx.train_df_pd
    val_df_pd = ctx.val_df_pd
    test_df_pd = ctx.test_df_pd
    train_df_polars = ctx.train_df_polars
    val_df_polars = ctx.val_df_polars
    test_df_polars = ctx.test_df_polars
    filtered_train_df = ctx.filtered_train_df
    filtered_val_df = ctx.filtered_val_df
    category_encoder = ctx.category_encoder
    imputer = ctx.imputer
    scaler = ctx.scaler
    trainset_features_stats = ctx.trainset_features_stats
    defer_pandas_conv = ctx.defer_pandas_conv
    train_df_size_bytes_cached = ctx.train_df_size_bytes_cached
    val_df_size_bytes_cached = ctx.val_df_size_bytes_cached
    _non_neural_train_times = ctx._non_neural_train_times
    models = ctx.models
    slug_to_original_target_name = ctx.slug_to_original_target_name
    _setup_out = _setup_per_target_mlframe_models(
        ctx=ctx,
        target_type=target_type,
        cur_target_name=cur_target_name,
        cur_target_values=cur_target_values,
        metadata=metadata,
        slug_to_original_target_name=slug_to_original_target_name,
    )
    plot_file = _setup_out["plot_file"]
    model_file = _setup_out["model_file"]
    _train_idx = _setup_out["_train_idx"]
    current_train_target = _setup_out["current_train_target"]
    current_val_target = _setup_out["current_val_target"]
    current_test_target = _setup_out["current_test_target"]
    metadata = _setup_out["metadata"]
    common_params = _setup_out["common_params"]
    models_params = _setup_out["models_params"]
    rfecv_models_params = _setup_out["rfecv_models_params"]
    cpu_configs = _setup_out["cpu_configs"]
    gpu_configs = _setup_out["gpu_configs"]
    pre_pipelines = _setup_out["pre_pipelines"]
    pre_pipeline_names = _setup_out["pre_pipeline_names"]


    # Custom transformers run AFTER preprocessing, so the preprocessing output is shared across
    # pre_pipelines of the same model-type bucket; one cache instance covers the whole sweep. Hoist
    # to ctx (PIPECACHE-PER-TGT) so multi-target suites share one cache across targets -- selector /
    # encoder fits done for target 1 are reusable for target 2 when the cache_key matches (only
    # changes when the feature set / strategy / kind / pp_name changes).
    if ctx._pipeline_cache is None:
        ctx._pipeline_cache = PipelineCache()
    pipeline_cache = ctx._pipeline_cache

    # Suite-scoped cache observability. ``finalize_suite`` aggregates these into
    # ``metadata["cache_stats"]``. Initialise once per call rather than per pre_pipeline so the inner
    # loop's HIT / MISS bumps accumulate across the whole target's training, and use ``setdefault``
    # at ctx level so cross-target calls (multi-target suites) keep counters monotonic across calls.
    if not hasattr(ctx, "_cache_stats") or ctx._cache_stats is None:
        ctx._cache_stats = {}

    # bench-attempt-rejected (2026-05-24): dropping the two outer ``tqdmu_lazy_start`` bars (keeping only the innermost weight-schema bar) saved ~1.1ms
    # of the 2.6ms per outer iteration in a synthetic 2x3x4 nested loop. Reverted because the outer bars give users visible per-pre_pipeline + per-model
    # progress on long suites; the small per-iter saving does not offset the diagnostic loss. ``tqdmu_lazy_start`` already suppresses single-item bars.
    for pre_pipeline, pre_pipeline_name in tqdmu_lazy_start(zip(pre_pipelines, pre_pipeline_names), desc="pre_pipeline", total=len(pre_pipelines)):
        # CatBoost + RFECV metamodel_func combination breaks sklearn.clone().
        if _should_skip_catboost_metamodel(pre_pipeline_name.strip(), target_type, behavior_config):
            continue

        # Skip identity-equivalent pre_pipelines: marker survives across targets, so a selector
        # that was a no-op on a prior target gets skipped here before any model trains.
        # Honour ``feature_selection_config.skip_identity_equivalent_pre_pipelines``: when False
        # the caller asked to retrain even on identity-equivalent pre_pipelines (e.g. for
        # ensembling-diversity-via-RNG-seed scenarios), so this early-exit must not fire.
        _pp_name_stripped = pre_pipeline_name.strip()
        if (
            _pp_name_stripped
            and feature_selection_config.skip_identity_equivalent_pre_pipelines
            and getattr(pre_pipeline, "_mlframe_identity_equivalent", False)
        ):
            logger.info(
                "[Dedup] Skipping pre_pipeline '%s' -- "
                "identity-equivalent to ordinary (cached from "
                "prior target/iteration); models already covered.",
                _pp_name_stripped,
            )
            continue
        ens_models = [] if use_mlframe_ensembles else None
        orig_pre_pipeline = pre_pipeline

        weight_schemas = _resolve_weight_schemas_and_warn_val_placement(
            sample_weights=sample_weights,
            split_config=split_config,
            ctx=ctx,
        )

        # Models sorted by feature tier (richest first) so text/embedding columns are dropped once per tier.
        # Strategy lookup keyed by id() because estimators / tuples are not hashable, and identity-distinct
        # instances must stay distinct in the map. Pre-computed once per suite by setup_configuration;
        # reading off ctx here avoids the O(targets * pre_pipelines * models) re-evaluation that used to
        # rebuild this map per inner-loop iteration.
        strategy_by_model = ctx.strategy_by_model
        sorted_models = ctx.sorted_mlframe_models
        # Suite-scoped feature-side cache: tier_dfs / pl.Enum map / prepared polars frames carry
        # ACROSS targets (target-independent transforms) so only y / sample_weight differ inside
        # the inner loop. Both inner caches are scoped to the current ``pre_pipeline_name`` since
        # different pre_pipelines may keep different columns (MRMR / RFECV vs ordinary), and the
        # tier-DFs / Enum maps depend on the column set after pre-pipeline column trimming.
        _suite_feature_cache = _ensure_feature_side_cache(ctx)
        _per_pp_cache = _suite_feature_cache.setdefault(pre_pipeline_name, {})
        tier_dfs_cache: dict[tuple, dict[str, Any]] = _per_pp_cache.setdefault("tier_dfs", {})
        # Leak-free pl.Enum map built from train+val UNION only (test EXCLUDED to avoid label-time leakage).
        # Depends only on (feature_tier, strategy class) - target-independent so it carries cross-target.
        tier_enum_map_cache: dict[tuple, dict[str, Any] | None] = _per_pp_cache.setdefault("tier_enum_map", {})
        # Prepared polars frames + xgb_category_map per (tier, supports_polars, strategy_class).
        # Target-independent because _prep_polars_df / build_polars_enum_map do not touch y; the
        # text-features fill_null pass below is also target-independent. Carry cross-target.
        prepared_frames_cache: dict[tuple, dict[str, Any]] = _per_pp_cache.setdefault("prepared_frames", {})
        prev_tier = None

        # Neural max_time defaults to P95 of non-neural train times so MLP can't run 2h while boosters take 5min.
        # CODE-LOW-4: per-target reset is INTENTIONAL -- each target's neural budget is computed only from the
        # same target's non-neural runs, so an unusually fast/slow earlier target cannot widen or starve the
        # current target's neural budget. We rebind both the local AND ctx._non_neural_train_times to the
        # SAME fresh list so the writeback at end-of-function is a no-op (the dict the caller sees is the
        # one we just mutated) and downstream readers of ctx._non_neural_train_times observe the per-target
        # contents in-flight, not the previous target's tail.
        _non_neural_train_times = []
        ctx._non_neural_train_times = _non_neural_train_times

        _total_models_in_run = len(sorted_models)
        _model_idx_in_run = 0
        _break_model_loop = False
        for mlframe_model_name in tqdmu_lazy_start(sorted_models, desc="mlframe model"):
            if _should_skip_catboost_metamodel(mlframe_model_name, target_type, behavior_config):
                continue
            # Extreme-AR + group-aware MLP skip (mirrors the composite-
            # discovery extreme_ar_group_aware_skip from round 5.3).
            # On AR(1)-dominated targets with a group-aware split, MLP
            # cannot learn a transferable residual: the target is fully
            # explained by the lag, and the MLP's nearly-linear decision
            # surface extrapolates catastrophically on unseen-group test
            # rows (observed in prod: very small pred_std vs target_std,
            # strongly negative R2, predictions near-constant vs target_mean).
            # The ensemble's quality gate catches it, but the wasted
            # train time + multi-MB save dump is pure cost. Skip MLP
            # in this regime; lag_predict + Ridge carry the AR signal.
            # Extreme-AR + group-aware MLP trigger predicate (shared by
            # 3 protections: skip / drop per-group aggregate cols /
            # bump weight_decay 100x). Computed once per (target, model)
            # so the three protections agree on whether to fire.
            _mlp_extreme_ar_fired = False
            _mlp_ea_lag1 = None
            _mlp_ea_thr = 0.99
            if mlframe_model_name == "mlp":
                _mlp_ea_thr = float(getattr(
                    behavior_config, "mlp_extreme_ar_threshold", 0.99,
                ))
                _td_report = metadata.get("target_distribution_report", {}) or {}
                _td_diag = _td_report.get("diagnostics", {}) or {}
                _td_knobs = _td_report.get("knob_overrides", {}) or {}
                _mlp_ea_lag1 = _td_diag.get("lag1_autocorr_per_group")
                _split_overrides = _td_knobs.get("split_config", {}) or {}
                _group_aware = bool(_split_overrides.get("prefer_group_aware", False))
                _mlp_extreme_ar_fired = bool(
                    _group_aware
                    and _mlp_ea_lag1 is not None
                    and float(_mlp_ea_lag1) >= _mlp_ea_thr
                )

                # Protection 1 (defensive opt-in): skip MLP entirely when
                # the operator has flipped ``mlp_extreme_ar_group_aware_skip=True``.
                # Default OFF; user explicitly wants MLP to train and
                # rely on protections 2/3 + the TTR predict-clip to bound
                # damage.
                if _mlp_extreme_ar_fired and bool(getattr(
                    behavior_config, "mlp_extreme_ar_group_aware_skip", False,
                )):
                    logger.warning(
                        "Skipping MLP training for target='%s' (model %d/%d): "
                        "extreme-AR + group-aware skip fired "
                        "(lag1_autocorr_per_group=%.4f >= %.2f). Disable via "
                        "TrainingBehaviorConfig(mlp_extreme_ar_group_aware_skip=False).",
                        cur_target_name, _model_idx_in_run + 1,
                        _total_models_in_run, float(_mlp_ea_lag1), _mlp_ea_thr,
                    )
                    continue
            _model_idx_in_run += 1
            if verbose:
                # Per-model RSS sample is intentional: localising OOM-blame to a specific
                # model+target+pre_pipeline tuple in the verbose-suite log saves hours of post-mortem
                # log-correlation. The ~3ms/call Windows cost is dwarfed by per-model fit times.
                # PSUTIL-IMPORT-HOT: ``psutil`` is now imported at module level (``_ps_module``);
                # the prior in-loop import paid ImportError lookup costs on every iter.
                try:
                    _ram_gb_now = (
                        _ps_module.Process().memory_info().rss / (1024 ** 3)
                        if _ps_module is not None else 0.0
                    )
                except Exception:
                    _ram_gb_now = 0.0
                logger.info(
                    "  process_model(%s) START -- model %d/%d, RAM=%.1fGB",
                    mlframe_model_name,
                    _model_idx_in_run, _total_models_in_run,
                    _ram_gb_now,
                )

            if mlframe_model_name not in models_params:
                logger.warning(f"mlframe model {mlframe_model_name} not known, skipping...")
                continue

            # Cross-target dataset reuse: restore the prior target's _DATASET_REUSE_CACHE_ATTRS
            # snapshot onto the freshly-built model template BEFORE the weight loop's clone()
            # forward-transfer reads them. select_target() rebuilds models_params per target so
            # the cache attributes are absent on a virgin template - without this restore the
            # XGB/LGB shims would rebuild the binned dataset on target 2. The shim's
            # signature_of(X) check then matches against the same ctx-pinned train_df pointer
            # and triggers set_label / set_weight in place rather than a fresh build.
            _restore_dataset_reuse_cache(
                ctx, mlframe_model_name, models_params[mlframe_model_name]["model"],
                pp_name=pre_pipeline_name,
            )

            strategy = strategy_by_model[id(mlframe_model_name)]

            # Drop pre-pipeline Polars originals as soon as we hit the first non-Polars strategy. The
            # post-iteration release fires only on tier transitions, but same-tier siblings (e.g. XGB and
            # LGB share tier=(False,False)) would keep Polars frames alive into a lazy pandas conversion,
            # doubling peak RAM. Releasing upfront halves peak in mixed suites.
            if (
                not strategy.supports_polars
                and train_df_polars is not None
            ):
                # Drop locals AND ctx attributes -- ctx still pins the strong ref to the same frames
                # assigned via ctx.*_df_polars at function entry, so a bare ``del`` of the locals would
                # leave maybe_clean_ram_and_gpu with nothing to reclaim and turn the log line into a lie.
                del train_df_polars, val_df_polars, test_df_polars
                train_df_polars = val_df_polars = test_df_polars = None
                # Drop polars-tier entries only - pandas-tier entries hang on the SAME cache dicts
                # (these locals now reference suite-scoped dicts in _per_pp_cache) and must survive
                # the release. ``_invalidate_polars_feature_side_cache(ctx)`` runs further down the
                # _release_ctx_polars_frames path and does the same for the prepared_frames sub-
                # cache; the tier_dfs / tier_enum_map dicts that pre-date this hoist are scrubbed
                # here so a same-target pandas-tier sibling reads a clean enum-map slot.
                for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
                    tier_dfs_cache.pop(_pl_only_key, None)
                tier_enum_map_cache.clear()  # All entries are polars-only (populated only on the polars fastpath).
                baseline_rss_mb = _release_ctx_polars_frames(
                    ctx,
                    baseline_rss_mb,
                    df_size_mb,
                    verbose=verbose,
                    reason="non-polars-native strategy entry",
                )
                if verbose:
                    logger.info(
                        "  Released pre-pipeline Polars originals before %s (non-polars-native strategy).",
                        mlframe_model_name,
                    )

            # Clone the base_pipeline per model so each iteration gets a fresh, un-fitted selector. Sharing a
            # fitted MRMR/RFECV across strategies caused `_is_fitted` to misreport True for a partially-fit
            # pipeline (selector fitted but encoder/imputer/scaler not), tripping imputer.transform on a
            # feature-names mismatch.
            _base_for_strategy = orig_pre_pipeline
            if _base_for_strategy is not None:
                try:
                    _base_for_strategy = clone(_base_for_strategy)
                except Exception as _clone_e:
                    # Custom non-BaseEstimator pipelines can't be sklearn-cloned;
                    # falling back to the original reference is correct IF the
                    # pipeline is genuinely stateless OR if it carries its own
                    # per-call reset. WARN-log so operators see when this
                    # fallback fires -- a partially-fit selector reused across
                    # strategies trips `imputer.transform` on a feature-names
                    # mismatch (the exact bug the docstring 4 lines above
                    # describes).
                    logger.warning(
                        "  sklearn.clone failed for base_pipeline (%s); reusing "
                        "original reference. If %s is a stateful selector with "
                        "no per-call reset, downstream `pre_pipeline.fit` may "
                        "see stale state from a prior model in the suite.",
                        _clone_e, type(_base_for_strategy).__name__,
                    )
            pre_pipeline = strategy.build_pipeline(
                base_pipeline=_base_for_strategy,
                cat_features=cat_features,
                category_encoder=category_encoder if cat_features else None,
                imputer=imputer,
                scaler=scaler,
            )
            # Cache key = strategy.cache_key + pre_pipeline_name + feature_tier + container kind + feature-list digest.
            # feature_tier is required because CB/LGB/XGB all share cache_key="tree" but have different
            # tiers; without it, CB's text/embedding-bearing frame would be served to LGB/XGB.
            # Kind suffix prevents Polars-native (XGB) and pandas-only (LGB) consumers from sharing entries
            # within a tier, which would otherwise undo the lazy pandas conversion downstream.
            # See _compute_pipeline_cache_key for the features-digest contract (frozenset, order-invariant).
            # Pass the polars train frame (if present) so dtype changes between targets / runs
            # invalidate the cache; pandas frames don't reach this branch typed-distinct enough to
            # need the suffix (handled upstream in split_features), so it's safe to skip there.
            _cache_key_train_df = train_df_polars if strategy.supports_polars else None
            # Use a CONTENT-based cache key derived from the preprocessing-
            # requirements tuple instead of strategy.cache_key (name-based).
            # Two strategies that consume IDENTICAL ``imp+scaler`` pipelines
            # MUST hit the same cache slot; name-keyed lookups
            # (LinearStrategy.cache_key="linear" vs NeuralStrategy.cache_key=
            # "neural") miss-on-name and re-do the 17s pre_pipeline transform
            # for the second tier (e.g. MLP after Ridge) on the same 4M rows.
            # The content key folds (requires_imputation, requires_scaling,
            # requires_encoding) into a stable string so any two strategies
            # with matching requirements share the cache slot.
            _content_key = (
                f"imp{int(getattr(strategy, 'requires_imputation', False))}"
                f"_scale{int(getattr(strategy, 'requires_scaling', False))}"
                f"_enc{int(getattr(strategy, 'requires_encoding', False))}"
            )
            cache_key = _compute_pipeline_cache_key(
                _content_key,
                pre_pipeline_name,
                strategy.feature_tier(),
                strategy.supports_polars,
                cat_features,
                text_features,
                embedding_features,
                train_df=_cache_key_train_df,
            )

            # Polars fastpath substitutes original Polars DataFrames for natively-Polars consumers
            # (CatBoost >= 1.2.7, HGB). Polars DFs are prepared once per model (outside the weight loop)
            # because prepare_polars_dataframe() allocates via .with_columns().
            polars_fastpath_active = train_df_polars is not None and strategy.supports_polars

            _prep_out = _prepare_strategy_inputs(
                polars_fastpath_active=polars_fastpath_active,
                mlframe_model_name=mlframe_model_name,
                strategy=strategy,
                cat_features=cat_features,
                text_features=text_features,
                embedding_features=embedding_features,
                train_df_polars=train_df_polars,
                val_df_polars=val_df_polars,
                test_df_polars=test_df_polars,
                prepared_frames_cache=prepared_frames_cache,
                tier_dfs_cache=tier_dfs_cache,
                tier_enum_map_cache=tier_enum_map_cache,
                common_params=common_params,
                pre_pipeline_name=pre_pipeline_name,
                ctx=ctx,
                verbose=verbose,
            )
            prepared_train = _prep_out["prepared_train"]
            prepared_val = _prep_out["prepared_val"]
            prepared_test = _prep_out["prepared_test"]
            _xgb_category_map = _prep_out["xgb_category_map"]
            _cat_features = _prep_out["cat_features"]
            tier_pandas = _prep_out["tier_pandas"]

            # CODE-P1-10: compute input-schema fingerprint ONCE per (model, pre_pipeline) outside the
            # weight loop. The fingerprinted train_df is the same across all weight schemas (only
            # sample_weight changes inside the weight loop), so the previous per-iteration call was
            # pure waste. Cache key is purely feature-side (strategy+tier+kind+pp_name) - dropping
            # ``target_type`` / ``cur_target_name`` from the key was the per-target hoist: the
            # schema hash depends on column names/dtypes, NOT on y, so target N reuses target 1's
            # fingerprint without recomputation. Audit-checked vs compute_model_input_fingerprint:
            # signature takes train_df + cat/text/embedding_features only, no target.
            _fp_train_df_pre = prepared_train if polars_fastpath_active else tier_pandas["train_df"]
            # FP-KEY-OMITS-CONTENT: original key excluded the train_df identity, so when the same
            # strategy / tier / kind / pp_name combination was hit by two different per-target
            # frames (filtered_train_df rebuilt across targets), the cache would return target 1's
            # schema hash for target 2. Fold ``id(train_df)`` (strong-ref-pinned at this point) and
            # the schema column-count to disambiguate; full schema hash isn't needed here because a
            # mismatch in id alone forces recompute.
            _fp_train_df_id = id(_fp_train_df_pre) if _fp_train_df_pre is not None else 0
            _fp_train_df_ncols = (
                len(_fp_train_df_pre.columns)
                if _fp_train_df_pre is not None and hasattr(_fp_train_df_pre, "columns")
                else 0
            )
            _fp_cache_key = (
                id(strategy),
                strategy.feature_tier(),
                strategy.supports_polars,
                pre_pipeline_name,
                _fp_train_df_id,
                _fp_train_df_ncols,
            )
            # Fingerprint cache stats: HIT when the per-(strategy, tier, kind, pp_name) key is already
            # cached, MISS when we have to compute. Same proxy-counter pattern as the pandas-view
            # cache above (the underlying cache is a plain dict on ctx with no counters of its own).
            _cs_fp = ctx._cache_stats.setdefault("fingerprint_cache", {"hits": 0, "misses": 0})
            if _fp_cache_key in ctx._model_input_fingerprint_cache:
                _cs_fp["hits"] += 1
                _schema_hash, _input_schema = ctx._model_input_fingerprint_cache[_fp_cache_key]
            else:
                _cs_fp["misses"] += 1
                _schema_hash, _input_schema = compute_model_input_fingerprint(
                    _fp_train_df_pre,
                    cat_features=cat_features,
                    text_features=text_features,
                    embedding_features=embedding_features,
                )
                ctx._model_input_fingerprint_cache[_fp_cache_key] = (_schema_hash, _input_schema)

            # Per-PP NGBoost-fallback invariant: when ``clone(original_model)`` raises ``TypeError`` (NGBoost: ``get_params`` exposes non-constructor
            # attrs), the fallback path re-pays ``original_model.get_params(deep=False)`` + ``{k:v for k in sig}`` once per weight iteration. The snapshot
            # is invariant across weights (only ``sample_weight`` differs at fit-time), so cache it once outside the loop. Lazy: compute on first use to
            # avoid paying for models that don't hit the TypeError path.
            _ngb_fallback_snapshot: dict | None = None

            # Per-PP CB extras invariants: ``_filter_polars_cat_features_by_dtype`` + ``filter_existing(text/embedding)`` depend only on ``prepared_train`` +
            # the (cat/text/embedding) feature lists -- all invariant across the weight loop. Compute once here so the inner loop only stitches the result
            # into ``current_model_params["fit_params"]`` (avoids paying the dtype filter + ``filter_existing`` 3 scans per weight iteration).
            _cb_extra_fit_invariant: dict[str, Any] | None = None
            if polars_fastpath_active and mlframe_model_name == "cb":
                _cb_extra_fit_invariant = {}
                if _cat_features:
                    _valid_cat_inv = _filter_polars_cat_features_by_dtype(prepared_train, _cat_features)
                    if _valid_cat_inv:
                        _cb_extra_fit_invariant["cat_features"] = _valid_cat_inv
                if text_features:
                    _cb_text_inv = filter_existing(prepared_train, text_features)
                    if _cb_text_inv:
                        _cb_extra_fit_invariant["text_features"] = _cb_text_inv
                if embedding_features:
                    _cb_emb_inv = filter_existing(prepared_train, embedding_features)
                    if _cb_emb_inv:
                        _cb_extra_fit_invariant["embedding_features"] = _cb_emb_inv

            for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items(), desc="weighting schema"):
                model_name_with_weight = common_params["model_name"]
                model_file_name=f"{mlframe_model_name}"
                if weight_name != "uniform":
                    model_name_with_weight += f" w={weight_name}"
                    model_file_name +=f"_{weight_name}"

                # Isolation copy: per-(model, weight) inner mutations (sample_weight, plot_file
                # decoration, lazy pandas conversion, fastpath frame swap) must not bleed into
                # the outer ``common_params`` template that the next iteration consumes. The
                # 4-deep nesting (target_type x target x pre_pipeline x model x weight) has been
                # verified across the suite -- removing the copy regresses the cross-weight
                # contamination tests. Do NOT inline.
                current_common_params = common_params.copy()
                current_common_params["sample_weight"] = weight_values

                if polars_fastpath_active:
                    current_common_params["train_df"] = prepared_train
                    if prepared_val is not None:
                        current_common_params["val_df"] = prepared_val
                    if prepared_test is not None:
                        current_common_params["test_df"] = prepared_test
                else:
                    current_common_params["train_df"] = tier_pandas["train_df"]
                    if tier_pandas.get("val_df") is not None:
                        current_common_params["val_df"] = tier_pandas["val_df"]
                    if tier_pandas.get("test_df") is not None:
                        current_common_params["test_df"] = tier_pandas["test_df"]

                # Drop per-group aggregate columns from the MLP's view of X.
                # Pattern matches ``group_*_(mean|std|min|max)`` by default.
                # Only the MLP sees the trimmed feature set; tree models in
                # the suite get the original columns. Gated on the knob +
                # the extreme-AR + group-aware trigger predicate (computed
                # above for the model-level skip). Drop applies to train /
                # val / test consistently so the predict path doesn't see
                # extra columns the network wasn't trained on.
                if (
                    mlframe_model_name == "mlp"
                    and bool(getattr(behavior_config, "mlp_drop_per_group_constants", False))
                ):
                    _drop_pattern = str(getattr(
                        behavior_config,
                        "mlp_drop_per_group_constants_pattern",
                        r"^group_.*_(mean|std|min|max)$",
                    ))
                    _train_df_now = current_common_params.get("train_df")
                    _cols_now = list(getattr(_train_df_now, "columns", []) or []) if _train_df_now is not None else []
                    _per_group_cols = _identify_per_group_columns(_cols_now, _drop_pattern)
                    if _per_group_cols:
                        logger.info(
                            "MLP per-group-aggregate column drop fired for target='%s': "
                            "dropping %d columns matching %r (e.g. %s). Tree models "
                            "still see them; only MLP gets the trimmed feature set.",
                            cur_target_name, len(_per_group_cols), _drop_pattern,
                            _per_group_cols[:3],
                        )
                        current_common_params["train_df"] = _drop_columns_for_mlp(
                            current_common_params.get("train_df"), _per_group_cols,
                        )
                        if current_common_params.get("val_df") is not None:
                            current_common_params["val_df"] = _drop_columns_for_mlp(
                                current_common_params.get("val_df"), _per_group_cols,
                            )
                        if current_common_params.get("test_df") is not None:
                            current_common_params["test_df"] = _drop_columns_for_mlp(
                                current_common_params.get("test_df"), _per_group_cols,
                            )
                if getattr(behavior_config, "model_file_hash_suffix", True):
                    model_file_name += f"__sch_{_schema_hash}"

                if weight_name != "uniform" and current_common_params.get("plot_file"):
                    current_common_params["plot_file"] = current_common_params["plot_file"] + weight_name + "_"

                cached_dfs = pipeline_cache.get(cache_key)

                # INTENTIONAL: clone() lives INSIDE the weight loop. Each weight schema produces a
                # different trained model stored separately in models[type][target]; without per-iteration
                # cloning all in-memory entries would alias to the same last-trained sklearn object and
                # only the .dump snapshots would be correct. Do NOT move clone() outside the loop.
                original_model = models_params[mlframe_model_name]["model"]
                cloned_model, _ngb_fallback_snapshot = _clone_model_with_sticky_flags(
                    original_model=original_model,
                    _cached_init_params=_cached_init_params,
                    _ngb_fallback_snapshot=_ngb_fallback_snapshot,
                    _forward_dataset_reuse_cache=_forward_dataset_reuse_cache,
                    logger_obj=logger,
                )
                # Isolation copy: each weight iteration installs its own cloned_model and may
                # patch fit_params (CatBoost text/embedding fastpath); without copying we would
                # mutate the suite-level models_params template and the next target would inherit
                # this iteration's overrides.
                current_model_params = models_params[mlframe_model_name].copy()
                current_model_params["model"] = cloned_model

                # CatBoost is the only Polars-native consumer that accepts cat_features / text_features / embedding_features at fit time; XGB and HGB
                # auto-detect via enable_categorical=True. Hoisted invariant ``_cb_extra_fit_invariant`` carries the filtered cat/text/embedding lists
                # (invariant across weights); stitch them into the per-weight fit_params here.
                if _cb_extra_fit_invariant and "fit_params" in current_model_params:
                    if _cb_extra_fit_invariant:
                        current_model_params["fit_params"] = {**current_model_params["fit_params"], **_cb_extra_fit_invariant}

                # MLP extreme-AR + group-aware protections (Fix 1 + Fix 3,
                # 2026-05-26). Trigger predicate ``_mlp_extreme_ar_fired``
                # is set above (per target, per model). Both modifications
                # land on the per-weight CLONED model, so other weight
                # schemas of this target see the same overrides; the
                # cross-target template is untouched because we mutate
                # ``current_model_params["model"]`` not ``models_params``.
                if mlframe_model_name == "mlp" and _mlp_extreme_ar_fired:
                    # Fix 1: bounded output activation (tanh -> hard cap).
                    _apply_mlp_extreme_ar_output_activation(cloned_model)
                    # Fix 3: L2 weight_decay bump by factor (default 100x).
                    _wd_factor = float(getattr(
                        behavior_config, "mlp_extreme_ar_weight_decay_factor", 100.0,
                    ))
                    _wd_base = float(getattr(
                        behavior_config, "mlp_extreme_ar_weight_decay_base", 1e-4,
                    ))
                    _apply_mlp_extreme_ar_weight_decay_bump(
                        cloned_model, factor=_wd_factor, base_weight_decay=_wd_base,
                    )

                # Build process_model kwargs using helper
                process_model_kwargs = _build_process_model_kwargs(
                    model_file=model_file,
                    model_name_with_weight=model_name_with_weight,
                    model_file_name=model_file_name,
                    target_type=target_type,
                    pre_pipeline=pre_pipeline,
                    pre_pipeline_name=pre_pipeline_name,
                    cur_target_name=cur_target_name,
                    models=models,
                    model_params=current_model_params,
                    common_params=current_common_params,
                    ens_models=ens_models,
                    trainset_features_stats=trainset_features_stats,
                    verbose=verbose,
                    cached_dfs=cached_dfs,
                    # Per-strategy decision on whether preprocessing for this strategy is already done.
                    # Two sufficient conditions:
                    #   (1) the suite-level polars-ds pipeline ran AND this strategy consumes polars natively;
                    #   (2) the polars fastpath is active for this strategy (its frame is the polars native
                    #       one, so sklearn encoder/scaler/imputer would be redundant and crash anyway).
                    # Note: requires_encoding=True is NOT a re-run trigger (HGB declares it for pandas-fallback
                    # only; on the polars fastpath HGB consumes pl.Categorical natively). Only non-Polars
                    # strategies fall through to their own pre_pipeline run in trainer.py.
                    polars_pipeline_applied=(
                        (polars_pipeline_applied and strategy.supports_polars)
                        or polars_fastpath_active
                    ),
                    mlframe_model_name=mlframe_model_name,
                    metadata_columns=metadata.get("columns"),
                )

                _is_neural = is_neural_model(mlframe_model_name)
                _timeout = _compute_neural_max_time(_non_neural_train_times) if _is_neural else None
                if _timeout is not None:
                    _max_time_dict, _p95, _n = _timeout
                    # Reach into Pipeline(StandardScaler, TTR(PytorchLightningRegressor(...))) to find trainer_params.
                    _neural_model = current_model_params.get("model")
                    if _neural_model is not None:
                        _inner = getattr(_neural_model, "regressor", None)
                        if _inner is None and hasattr(_neural_model, "named_steps"):
                            for _step in _neural_model.named_steps.values():
                                if hasattr(_step, "regressor"):
                                    _inner = _step.regressor
                                    break
                        if _inner is not None and hasattr(_inner, "trainer_params"):
                            _inner.trainer_params["max_time"] = _max_time_dict
                            if verbose:
                                logger.info(
                                    "  [NeuralTimeout] %s max_time=%dh%02dm%02ds "
                                    "(P95 of %d prior non-neural train times: %.0fs)",
                                    mlframe_model_name,
                                    _max_time_dict["hours"], _max_time_dict["minutes"], _max_time_dict["seconds"],
                                    _n, _p95,
                                )

                t0_model = timer()
                try:
                    with phase("process_model", model=mlframe_model_name, weight=weight_name):
                        trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                            **process_model_kwargs
                        )
                except Exception as model_err:
                    # Skip-and-continue is opt-in. KeyboardInterrupt is intentionally not caught here;
                    # native SIGSEGV that kills the process won't be caught either.
                    if not behavior_config.continue_on_model_failure:
                        raise
                    logger.error(
                        f"  process_model({mlframe_model_name}, w={weight_name}) FAILED after "
                        f"{_elapsed_str(t0_model)} -- {type(model_err).__name__}: {model_err}. "
                        f"continue_on_model_failure=True -> skipping and moving on.",
                        exc_info=True,
                    )
                    metadata.setdefault("failed_models", []).append({
                        "model": mlframe_model_name,
                        "weighting": weight_name,
                        "error_type": type(model_err).__name__,
                        "error_message": str(model_err),
                    })
                    continue  # next weight_name in the inner loop
                if verbose:
                    logger.info("  process_model(%s, w=%s) done -- %s", mlframe_model_name, weight_name, _elapsed_str(t0_model))
                if not _is_neural and t0_model is not None:
                    _non_neural_train_times.append(timer() - t0_model)

                # After the first model trains, if the pre_pipeline is identity-equivalent (kept all
                # columns) AND the ordinary branch is in the suite, the remaining models would see
                # identical data - skip them.
                if (
                    _model_idx_in_run == 1
                    and _pp_name_stripped
                    and use_ordinary_models
                    and feature_selection_config.skip_identity_equivalent_pre_pipelines
                    and getattr(
                        pre_pipeline, "_mlframe_identity_equivalent", False
                    )
                ):
                    _skip_remaining = _total_models_in_run - 1
                    if _skip_remaining > 0:
                        logger.info(
                            "[Dedup] pre_pipeline '%s' is "
                            "identity-equivalent to ordinary (kept "
                            "all %d columns); skipping remaining "
                            "%d model(s) for this target.",
                            _pp_name_stripped,
                            train_df_transformed.shape[1]
                            if train_df_transformed is not None
                            else 0,
                            _skip_remaining,
                        )
                    _break_model_loop = True
                    break  # exit weight_schema loop

                # Hand the dataset-reuse cache from cloned_model back to the template so the next
                # weight-schema iteration's clone() carries it forward (symmetric to the forward-transfer
                # block above). Without this the cache would be born and die in a single iteration.
                _forward_dataset_reuse_cache(cloned_model, original_model, skip_none=True)

                _build_and_record_model_schema(
                    ctx=ctx,
                    metadata=metadata,
                    model_file_name=model_file_name,
                    mlframe_model_name=mlframe_model_name,
                    weight_name=weight_name,
                    target_type=target_type,
                    strategy=strategy,
                    cur_target_name=cur_target_name,
                    cur_target_values=cur_target_values,
                    _train_idx=_train_idx,
                    pre_pipeline=pre_pipeline,
                    pre_pipeline_name=pre_pipeline_name,
                    train_df_transformed=train_df_transformed,
                    _schema_hash=_schema_hash,
                    _input_schema=_input_schema,
                    _build_feature_selection_report=_build_feature_selection_report,
                    _selector_params_hash=_selector_params_hash,
                    _unwrap_selector=_unwrap_selector,
                )

                if cached_dfs is None:
                    pipeline_cache.set(cache_key, train_df_transformed, val_df_transformed, test_df_transformed)

            # Preserve a fitted feature-selector across same-bucket tree iterations. Tree strategies return
            # just the base_pipeline from build_pipeline(); non-tree strategies wrap it in a full Pipeline
            # (encoder/imputer/scaler), which we do NOT want to reuse as the base for other model types.
            if cache_key.startswith("tree"):
                orig_pre_pipeline = pre_pipeline

            if _break_model_loop:
                break

            # Release dataset-reuse caches at strategy-iter end. Both shims park the heavy binned dataset
            # on ``_cached_train_*`` / ``_cached_val_*`` as a weight-schema-loop scratchpad; nothing
            # downstream reads them (.predict goes through _Booster, ensemble uses pre-computed probs,
            # save strips via __getstate__). Releasing here frees ~30% of peak RAM between strategies.
            # Capture the binned-dataset references off the template BEFORE clearing so the next
            # target's _restore_dataset_reuse_cache can re-attach them. Without this snapshot the
            # clear below frees the dataset and the cross-target hoist degrades to a no-op (same
            # behaviour as before the hoist). Storing references only - the binned dataset is
            # shared with whatever held it before; the clear merely drops the template's pointer.
            _capture_dataset_reuse_cache(ctx, mlframe_model_name, original_model, pp_name=pre_pipeline_name)
            _maybe_clear_shim_cache(original_model)
            # ens_models snapshots may also hold the cache by reference (forward-transfer at clone() copied
            # the reference rather than moving it); release on each so the binned dataset can be freed.
            if ens_models:
                for _ens_ns in ens_models:
                    _maybe_clear_shim_cache(getattr(_ens_ns, "model", None))

            # On a tier transition into a non-Polars strategy, release the pre-pipeline Polars originals.
            cur_tier = strategy.feature_tier()
            if prev_tier is not None and cur_tier != prev_tier and not strategy.supports_polars:
                if train_df_polars is not None:
                    # Same rationale as the entry-site release: locals AND ctx attributes must both drop their refs.
                    del train_df_polars, val_df_polars, test_df_polars
                    train_df_polars = val_df_polars = test_df_polars = None
                    # Selective drop: see same-shape comment at the non-polars-native entry site
                    # above for rationale. These cache references are suite-scoped now, so a blanket
                    # .clear() would also wipe pandas-tier entries that survived the polars release.
                    for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
                        tier_dfs_cache.pop(_pl_only_key, None)
                    tier_enum_map_cache.clear()
                    baseline_rss_mb = _release_ctx_polars_frames(ctx, baseline_rss_mb, df_size_mb, verbose=verbose, reason="tier transition")
                    if verbose:
                        logger.info("  Released pre-pipeline Polars originals (tier transition)")
            prev_tier = cur_tier

        _finalize_per_target_ensembling(
            ens_models=ens_models,
            # ``train_df_transformed`` is set inside the inner weight-schema
            # loop after a successful ``process_model`` call; if every model
            # in the suite errored out (unknown model name, infinity-row
            # ShapeError, etc.) the loop never executes the assignment.
            # Use locals().get(...) so the post-loop ensembling block degrades
            # gracefully to ``None`` instead of UnboundLocalError; the
            # ensembling helper already skips when its inputs are unusable.
            train_df_transformed=locals().get("train_df_transformed"),
            behavior_config=behavior_config,
            ctx=ctx,
            cur_target_name=cur_target_name,
            current_common_params=locals().get("current_common_params"),
            common_params=common_params,
            pre_pipeline_name=pre_pipeline_name,
            models=models,
            target_type=target_type,
            metadata=metadata,
            verbose=verbose,
        )

    ctx.models = models
    ctx.metadata = metadata
    ctx.trainset_features_stats = trainset_features_stats
    # Merge ``pipeline_cache`` HIT / MISS counters into the per-suite cache_stats accumulator.
    # PipelineCache itself is local to this function (one instance per pre_pipeline sweep) so the
    # only handoff to finalize is this stash; later targets create fresh PipelineCaches whose hits
    # accumulate via ``+=`` into the suite-wide running totals.
    try:
        _cs_pc = ctx._cache_stats.setdefault("pipeline_cache", {"hits": 0, "misses": 0})
        _cs_pc["hits"] += int(getattr(pipeline_cache, "n_hits", 0))
        _cs_pc["misses"] += int(getattr(pipeline_cache, "n_misses", 0))
    except Exception as _pc_stats_err:
        logger.debug("pipeline_cache stats merge failed: %s", _pc_stats_err)
    # CODE-LOW-2 + CODE-LOW-4: slug_to_original_target_{type,name} and _non_neural_train_times
    # are mutable containers we already rebound on ctx (the slugs are bound by reference at the top
    # of this function and mutated in place; _non_neural_train_times is rebound to a fresh list each
    # target with a matching ``ctx._non_neural_train_times = _non_neural_train_times`` at that point).
    # The earlier writeback of these three was a no-op.
    ctx.train_df_polars = train_df_polars
    ctx.val_df_polars = val_df_polars
    ctx.test_df_polars = test_df_polars
    ctx.train_df_pd = train_df_pd
    ctx.val_df_pd = val_df_pd
    ctx.test_df_pd = test_df_pd
    ctx.filtered_train_df = filtered_train_df
    ctx.filtered_val_df = filtered_val_df
    ctx.pipeline = pipeline
    ctx.defer_pandas_conv = defer_pandas_conv
    ctx.baseline_rss_mb = baseline_rss_mb
    ctx.train_df_size_bytes_cached = train_df_size_bytes_cached
    ctx.val_df_size_bytes_cached = val_df_size_bytes_cached
