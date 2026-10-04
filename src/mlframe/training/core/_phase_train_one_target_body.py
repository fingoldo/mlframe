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

from typing import Any

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]


from pyutilz.system import tqdmu_lazy_start

from ..strategies import PipelineCache
from ._setup_helpers import (
    _should_skip_catboost_metamodel,
)
from ._phase_train_one_target_ensembling import _finalize_per_target_ensembling
from ._phase_train_one_target_pre_screen import _maybe_run_unsupervised_pre_screen
from ._phase_train_one_target_model_setup import _setup_per_target_mlframe_models
from ._phase_train_one_target_schema import (
    _resolve_weight_schemas_and_warn_val_placement,
)
from types import SimpleNamespace as _SimpleNamespace

from ._phase_train_one_target_steps import (  # noqa: F401  -- carved helpers
    logger,
    _train_one_target_ste_step1_model_entry_tqdmu,
    _train_one_target_ste_step1_model_name_weight,
    _train_one_target_ste_step2_supports_native_multi,
    _train_one_target_ste_step3_timeout_none,
    _train_one_target_ste_step1_doubling_peak_ram,
    _train_one_target_ste_step2_into_current_model,
    _train_one_target_ste_step3_strategy_but_cannot,
    _train_one_target_ste_step4_tier_transition_into,
)
# Names the moved stage helpers no longer need here, kept importable from this module (tests and callers reach them through it).
import logging  # noqa: F401
from timeit import default_timer as timer  # noqa: F401
from sklearn.base import clone  # noqa: F401
from ..phases import phase  # noqa: F401
from ..models import is_neural_model  # noqa: F401
from ..train_eval import process_model  # noqa: F401
from ..utils import compute_model_input_fingerprint, filter_existing  # noqa: F401
from ._misc_helpers import _compute_neural_max_time, _elapsed_str, _filter_polars_cat_features_by_dtype, _maybe_clear_shim_cache  # noqa: F401
from ._setup_helpers import _build_process_model_kwargs, _entry_not_for_target  # noqa: F401
from ._phase_train_one_target_polars_fastpath import _prepare_strategy_inputs  # noqa: F401
from ._phase_train_one_target_schema import _build_and_record_model_schema, _clone_model_with_sticky_flags  # noqa: F401
from ._phase_train_one_target_mlp_helpers import _apply_mlp_extreme_ar_output_activation, _apply_mlp_extreme_ar_weight_decay_bump, _drop_columns_for_mlp, _identify_per_group_columns  # noqa: F401
from ._phase_train_one_target_post import _evaluate_mlp_extreme_ar_gate, _forward_selector_sticky_attrs, _run_per_model_post_train_tail  # noqa: F401
from ._phase_train_one_target_cache_helpers import compute_cached_model_input_fingerprint, compute_model_pipeline_cache_key  # noqa: F401
from mlframe.utils.log_throttle import log_throttle  # noqa: F401
from ._phase_train_one_target_steps import _clone_base_pipeline_for_strategy  # noqa: F401  -- defined with the stage helpers that call it


def _train_one_target(ctx, target_type, targets, cur_target_name, cur_target_values):
    """Train all models for one (target_type, target_name) pair."""
    # Lazy import: ``._phase_train_one_target`` re-imports this sibling at
    # its module bottom for re-export. Top-level ``from ._phase_train_one_target
    # import ...`` would create a hard import cycle. Python module cache makes
    # repeat imports cheap; the dict lookup is sub-microsecond per call.
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)

    # Normally-initialized (not read back via locals().get(...) later) so a rename/reorder/
    # extraction-into-a-helper fails loudly (NameError) instead of silently degrading to None.
    # Both are set inside the inner weight-schema loop below and read after it finishes; if every
    # model in the suite errored out the loop body never runs, and these stay None -- the
    # post-loop ensembling helper already handles that degraded case.
    st.train_df_transformed = None
    st.current_common_params = None

    _maybe_run_unsupervised_pre_screen(ctx, targets)
    st.split_config = ctx.split_config
    st.behavior_config = ctx.behavior_config
    st.feature_selection_config = ctx.feature_selection_config
    st.verbose = ctx.verbose
    st.use_ordinary_models = ctx.use_ordinary_models
    st.use_mlframe_ensembles = ctx.use_mlframe_ensembles
    st.metadata = ctx.metadata
    st.sample_weights = ctx.sample_weights
    st.baseline_rss_mb = ctx.baseline_rss_mb
    st.df_size_mb = ctx.df_size_mb
    st.pipeline = ctx.pipeline
    st.polars_pipeline_applied = ctx.polars_pipeline_applied
    st.cat_features = ctx.cat_features
    st.text_features = ctx.text_features
    st.embedding_features = ctx.embedding_features
    st.train_df_pd = ctx.train_df_pd
    st.val_df_pd = ctx.val_df_pd
    st.test_df_pd = ctx.test_df_pd
    st.train_df_polars = ctx.train_df_polars
    st.val_df_polars = ctx.val_df_polars
    st.test_df_polars = ctx.test_df_polars
    st.filtered_train_df = ctx.filtered_train_df
    st.filtered_val_df = ctx.filtered_val_df
    st.category_encoder = ctx.category_encoder
    st.imputer = ctx.imputer
    st.scaler = ctx.scaler
    st.trainset_features_stats = ctx.trainset_features_stats
    st.defer_pandas_conv = ctx.defer_pandas_conv
    st.train_df_size_bytes_cached = ctx.train_df_size_bytes_cached
    st.val_df_size_bytes_cached = ctx.val_df_size_bytes_cached
    st._non_neural_train_times = ctx._non_neural_train_times
    st.models = ctx.models
    st.slug_to_original_target_name = ctx.slug_to_original_target_name
    st._setup_out = _setup_per_target_mlframe_models(
        ctx=ctx,
        target_type=target_type,
        cur_target_name=cur_target_name,
        cur_target_values=cur_target_values,
        metadata=st.metadata,
        slug_to_original_target_name=st.slug_to_original_target_name,
    )
    st.model_file = st._setup_out["model_file"]
    st._train_idx = st._setup_out["_train_idx"]
    st.current_train_target = st._setup_out["current_train_target"]
    st.current_val_target = st._setup_out["current_val_target"]
    st.current_test_target = st._setup_out["current_test_target"]
    st.metadata = st._setup_out["metadata"]
    st.common_params = st._setup_out["common_params"]
    st.models_params = st._setup_out["models_params"]
    st.pre_pipelines = st._setup_out["pre_pipelines"]
    st.pre_pipeline_names = st._setup_out["pre_pipeline_names"]

    # Custom transformers run AFTER preprocessing, so the preprocessing output is shared across
    # pre_pipelines of the same model-type bucket; one cache instance covers the whole sweep. Hoist
    # to ctx (PIPECACHE-PER-TGT) so multi-target suites share one cache across targets -- selector /
    # encoder fits done for target 1 are reusable for target 2 when the cache_key matches (only
    # changes when the feature set / strategy / kind / pp_name changes).
    if ctx._pipeline_cache is None:
        ctx._pipeline_cache = PipelineCache(ram_budget_fraction=getattr(getattr(ctx, "behavior_config", None), "pipeline_cache_ram_budget_fraction", None))
    st.pipeline_cache = ctx._pipeline_cache

    # Suite-scoped cache observability. ``finalize_suite`` aggregates these into
    # ``metadata["cache_stats"]``. Initialise once per call rather than per pre_pipeline so the inner
    # loop's HIT / MISS bumps accumulate across the whole target's training, and use ``setdefault``
    # at ctx level so cross-target calls (multi-target suites) keep counters monotonic across calls.
    if not hasattr(ctx, "_cache_stats") or ctx._cache_stats is None:
        ctx._cache_stats = {}

    # bench-attempt-rejected: dropping the two outer ``tqdmu_lazy_start`` bars (keeping only the innermost weight-schema bar) saved ~1.1ms
    # of the 2.6ms per outer iteration in a synthetic 2x3x4 nested loop. Reverted because the outer bars give users visible per-pre_pipeline + per-model
    # progress on long suites; the small per-iter saving does not offset the diagnostic loss. ``tqdmu_lazy_start`` already suppresses single-item bars.
    _train_one_target_step1_progress_long_suites(st, target_type, ctx, cur_target_name, cur_target_values)

    ctx.models = st.models
    ctx.metadata = st.metadata
    ctx.trainset_features_stats = st.trainset_features_stats
    # Merge ``pipeline_cache`` HIT / MISS counters into the per-suite cache_stats accumulator.
    # PipelineCache itself is local to this function (one instance per pre_pipeline sweep) so the
    # only handoff to finalize is this stash; later targets create fresh PipelineCaches whose hits
    # accumulate via ``+=`` into the suite-wide running totals.
    try:
        _cs_pc = ctx._cache_stats.setdefault("pipeline_cache", {"hits": 0, "misses": 0})
        _cs_pc["hits"] += int(getattr(st.pipeline_cache, "n_hits", 0))
        _cs_pc["misses"] += int(getattr(st.pipeline_cache, "n_misses", 0))
    except Exception as _pc_stats_err:
        logger.debug("pipeline_cache stats merge failed: %s", _pc_stats_err)
    # CODE-LOW-2 + CODE-LOW-4: slug_to_original_target_{type,name} and _non_neural_train_times
    # are mutable containers we already rebound on ctx (the slugs are bound by reference at the top
    # of this function and mutated in place; _non_neural_train_times is rebound to a fresh list each
    # target with a matching ``ctx._non_neural_train_times = _non_neural_train_times`` at that point).
    # The earlier writeback of these three was a no-op.
    ctx.train_df_polars = st.train_df_polars
    ctx.val_df_polars = st.val_df_polars
    ctx.test_df_polars = st.test_df_polars
    ctx.train_df_pd = st.train_df_pd
    ctx.val_df_pd = st.val_df_pd
    ctx.test_df_pd = st.test_df_pd
    ctx.filtered_train_df = st.filtered_train_df
    ctx.filtered_val_df = st.filtered_val_df
    ctx.pipeline = st.pipeline
    ctx.defer_pandas_conv = st.defer_pandas_conv
    ctx.baseline_rss_mb = st.baseline_rss_mb
    ctx.train_df_size_bytes_cached = st.train_df_size_bytes_cached
    ctx.val_df_size_bytes_cached = st.val_df_size_bytes_cached


def _train_one_target_step1_progress_long_suites(st, target_type, ctx, cur_target_name, cur_target_values):
    """Step 1 of _train_one_target: lines starting at ``for pre_pipeline, pre_pipeline_name in tqdmu_lazy_start(zip(st.pre_pip``."""
    from mlframe.training.core._phase_train_one_target import (
        _ensure_feature_side_cache,
    )

    for pre_pipeline, pre_pipeline_name in tqdmu_lazy_start(zip(st.pre_pipelines, st.pre_pipeline_names), desc="pre_pipeline", total=len(st.pre_pipelines)):
        # CatBoost + RFECV metamodel_func combination breaks sklearn.clone().
        if _should_skip_catboost_metamodel(pre_pipeline_name.strip(), target_type, st.behavior_config):
            continue

        # Skip identity-equivalent pre_pipelines: marker survives across targets, so a selector
        # that was a no-op on a prior target gets skipped here before any model trains.
        # Honour ``feature_selection_config.skip_identity_equivalent_pre_pipelines``: when False
        # the caller asked to retrain even on identity-equivalent pre_pipelines (e.g. for
        # ensembling-diversity-via-RNG-seed scenarios), so this early-exit must not fire.
        _pp_name_stripped = pre_pipeline_name.strip()
        if (
            _pp_name_stripped
            and st.feature_selection_config.skip_identity_equivalent_pre_pipelines
            and getattr(pre_pipeline, "_mlframe_identity_equivalent", False)
        ):
            logger.info(
                "[Dedup] Skipping pre_pipeline '%s' -- " "identity-equivalent to ordinary (cached from " "prior target/iteration); models already covered.",
                _pp_name_stripped,
            )
            continue
        ens_models: list | None = [] if st.use_mlframe_ensembles else None
        orig_pre_pipeline = pre_pipeline

        weight_schemas = _resolve_weight_schemas_and_warn_val_placement(
            sample_weights=st.sample_weights,
            split_config=st.split_config,
            ctx=ctx,
        )

        # Models sorted by feature tier (richest first) so text/embedding columns are dropped once per tier.
        # Strategy lookup keyed by id() because estimators / tuples are not hashable, and identity-distinct
        # instances must stay distinct in the map. Pre-computed once per suite by setup_configuration;
        # reading off ctx here avoids the O(targets * pre_pipelines * models) re-evaluation that used to
        # rebuild this map per inner-loop iteration.
        strategy_by_model = ctx.strategy_by_model
        sorted_models = ctx.sorted_mlframe_models
        # ``sorted_mlframe_models`` is a suite-level constant computed once at setup time from the
        # (possibly still-unresolved) ``mlframe_models`` argument, before any per-target data exists --
        # it can never contain a model key that ``configure_training_params`` only decides to register
        # per-target from data it inspects at fit time (e.g. ``gated_outlier``, auto-added to
        # ``models_params`` only when the CURRENT target's train split shows a genuine point mass).
        # Without this, ``models_params["gated_outlier"]`` would be built and then silently never
        # visited by the loop below (``if _model_entry not in models_params: skip``), since the loop
        # iterates ``sorted_models``, not ``models_params.keys()``. Extend rather than replace: this
        # keeps the original tier-sort order for every statically-known model and only appends genuinely
        # new, dynamically-discovered keys.
        _dynamic_extra_models = [m for m in st.models_params.keys() if m not in sorted_models]
        if _dynamic_extra_models:
            sorted_models = list(sorted_models) + _dynamic_extra_models
            # ``strategy_by_model`` is looked up by ``id(_model_entry)`` further down (``strategy_by_model[id(_model_entry)]``)
            # and was built only from the original static ``mlframe_models`` list -- a dynamically-appended key has no
            # entry there and would KeyError the first time the loop reaches it. Compute + register a strategy for each
            # new key the same way ``_phase_config_setup.py`` did for the static list, and copy the dict so the suite-level
            # ``ctx.strategy_by_model`` (shared across pre_pipelines/targets) is never mutated by a single target's discovery.
            from mlframe.training.strategies import get_strategy as _get_strategy_dynamic

            strategy_by_model = dict(strategy_by_model)
            for _extra_model in _dynamic_extra_models:
                strategy_by_model[id(_extra_model)] = _get_strategy_dynamic(_extra_model)
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
        st._non_neural_train_times = []
        ctx._non_neural_train_times = st._non_neural_train_times

        _total_models_in_run = len(sorted_models)
        _model_idx_in_run = 0
        pre_pipeline = _train_one_target_ste_step1_model_entry_tqdmu(sorted_models, target_type, st, cur_target_name, _model_idx_in_run, _total_models_in_run, ctx, pre_pipeline_name, strategy_by_model, tier_dfs_cache, tier_enum_map_cache, orig_pre_pipeline, prepared_frames_cache, weight_schemas, ens_models, cur_target_values, _pp_name_stripped, prev_tier, pre_pipeline)

        _finalize_per_target_ensembling(
            ens_models=ens_models,
            # ``train_df_transformed`` is set inside the inner weight-schema loop after a
            # successful ``process_model`` call; if every model in the suite errored out (unknown
            # model name, infinity-row ShapeError, etc.) the loop never executes the assignment
            # and this stays at its function-top None -- the ensembling helper already skips when
            # its inputs are unusable.
            train_df_transformed=st.train_df_transformed,
            behavior_config=st.behavior_config,
            ctx=ctx,
            cur_target_name=cur_target_name,
            current_common_params=st.current_common_params,
            common_params=st.common_params,
            current_val_target=st.current_val_target,
            pre_pipeline_name=pre_pipeline_name,
            models=st.models,
            target_type=target_type,
            metadata=st.metadata,
            verbose=st.verbose,
        )
