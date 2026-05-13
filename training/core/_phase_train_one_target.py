"""
_train_one_target — per-target training entry point.

Receives TrainingContext + loop variables, unpacks ctx fields as locals so the
body stays byte-for-byte identical to the inline orchestrator code.
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer

import numpy as np

from pyutilz.strings import slugify

from ..phases import phase
from ..utils import log_ram_usage
from ._misc_helpers import _elapsed_str
from ._setup_helpers import _setup_model_directories
from ..strategies import PipelineCache

logger = logging.getLogger(__name__)


def _train_one_target(ctx, target_type, targets, cur_target_name, cur_target_values):
    """Train all models for one (target_type, target_name) pair.

    Unpacks ctx fields as local variables, then runs the full per-target loop body
    (diagnostics → select_target → model loop → ensemble scoring).
    """
    # ── Unpack ctx to local variables (mirrors orchestrator scope) ──
    model_name = ctx.model_name
    target_name = ctx.target_name
    preprocessing_config = ctx.preprocessing_config
    pipeline_config = ctx.pipeline_config
    feature_types_config = ctx.feature_types_config
    split_config = ctx.split_config
    hyperparams_config = ctx.hyperparams_config
    behavior_config = ctx.behavior_config
    reporting_config = ctx.reporting_config
    output_config = ctx.output_config
    outlier_detection_config = ctx.outlier_detection_config
    feature_selection_config = ctx.feature_selection_config
    confidence_analysis_config = ctx.confidence_analysis_config
    baseline_diagnostics_config = ctx.baseline_diagnostics_config
    dummy_baselines_config = ctx.dummy_baselines_config
    quantile_regression_config = ctx.quantile_regression_config
    composite_target_discovery_config = ctx.composite_target_discovery_config
    data_dir = ctx.data_dir
    models_dir = ctx.models_dir
    save_charts = ctx.save_charts
    outlier_detector = ctx.outlier_detector
    od_val_set = ctx.od_val_set
    use_mrmr_fs = ctx.use_mrmr_fs
    mrmr_kwargs = ctx.mrmr_kwargs
    rfecv_models = ctx.rfecv_models
    custom_pre_pipelines = ctx.custom_pre_pipelines
    common_params_dict = ctx.common_params_dict
    mlframe_models = ctx.mlframe_models
    metadata = ctx.metadata
    target_by_type = ctx.target_by_type
    group_ids = ctx.group_ids
    timestamps = ctx.timestamps
    sample_weights = ctx.sample_weights
    baseline_rss_mb = ctx.baseline_rss_mb
    df_size_mb = ctx.df_size_mb
    sequences = ctx.sequences
    train_idx = ctx.train_idx
    val_idx = ctx.val_idx
    test_idx = ctx.test_idx
    train_details = ctx.train_details
    val_details = ctx.val_details
    test_details = ctx.test_details
    fairness_subgroups = ctx.fairness_subgroups
    fairness_features = ctx.fairness_features
    train_sequences = ctx.train_sequences
    val_sequences = ctx.val_sequences
    test_sequences = ctx.test_sequences
    train_df = ctx.train_df
    val_df = ctx.val_df
    test_df = ctx.test_df
    pipeline = ctx.pipeline
    extensions_pipeline = ctx.extensions_pipeline
    was_polars_input = ctx.was_polars_input
    all_models_polars_native = ctx.all_models_polars_native
    polars_pipeline_applied = ctx.polars_pipeline_applied
    train_df_polars_pre = ctx.train_df_polars_pre
    val_df_polars_pre = ctx.val_df_polars_pre
    test_df_polars_pre = ctx.test_df_polars_pre
    preprocessing_extensions = ctx.preprocessing_extensions
    cat_features = ctx.cat_features
    cat_features_polars = ctx.cat_features_polars
    text_features = ctx.text_features
    embedding_features = ctx.embedding_features
    text_emb_set = ctx.text_emb_set
    _dropped_high_card_data = ctx._dropped_high_card_data
    train_df_pd = ctx.train_df_pd
    val_df_pd = ctx.val_df_pd
    test_df_pd = ctx.test_df_pd
    train_df_polars = ctx.train_df_polars
    val_df_polars = ctx.val_df_polars
    test_df_polars = ctx.test_df_polars
    filtered_train_df = ctx.filtered_train_df
    filtered_val_df = ctx.filtered_val_df
    filtered_train_idx = ctx.filtered_train_idx
    filtered_val_idx = ctx.filtered_val_idx
    train_od_idx = ctx.train_od_idx
    val_od_idx = ctx.val_od_idx
    outlier_detection_result = ctx.outlier_detection_result
    category_encoder = ctx.category_encoder
    imputer = ctx.imputer
    scaler = ctx.scaler
    trainset_features_stats = ctx.trainset_features_stats
    can_skip_pandas_conv = ctx.can_skip_pandas_conv
    train_df_size_bytes_cached = ctx.train_df_size_bytes_cached
    val_df_size_bytes_cached = ctx.val_df_size_bytes_cached
    _all_target_audits = ctx._all_target_audits
    _non_neural_train_times = ctx._non_neural_train_times
    models = ctx.models
    slug_to_original_target_type = ctx.slug_to_original_target_type
    slug_to_original_target_name = ctx.slug_to_original_target_name
    models_dir = ctx.models_dir
    # Store original cur_target_name mapping
    slug_to_original_target_name[slugify(cur_target_name)] = cur_target_name
    # Initialize rfecv_models_params before conditional to avoid NameError if mlframe_models is empty
    rfecv_models_params = {}
    if mlframe_models:
        # Set up directories for charts and models
        plot_file, model_file = _setup_model_directories(
            target_name=target_name,
            model_name=model_name,
            target_type=target_type,
            cur_target_name=cur_target_name,
            data_dir=data_dir,
            models_dir=models_dir,
            save_charts=save_charts,
        )

        # Subset targets using pre-filtered indices (OD already applied globally)
        current_train_target = (
            cur_target_values[filtered_train_idx]
            if isinstance(cur_target_values, (np.ndarray, pl.Series))
            else cur_target_values.iloc[filtered_train_idx]
        )
        current_val_target = None
        if filtered_val_idx is not None:
            current_val_target = (
                cur_target_values[filtered_val_idx]
                if isinstance(cur_target_values, (np.ndarray, pl.Series))
                else cur_target_values.iloc[filtered_val_idx]
            )
        # Test target extraction for the drift report. test_idx is
        # NOT subset by the outlier-detector (test never gets
        # OD-filtered) so we use the raw test_idx here.
        current_test_target = None
        if test_idx is not None:
            current_test_target = (
                cur_target_values[test_idx]
                if isinstance(cur_target_values, (np.ndarray, pl.Series))
                else cur_target_values.iloc[test_idx]
            )
        # Drift report + baseline diagnostics -- extracted helper
        metadata = run_per_target_diagnostics(
            target_type=target_type,
            cur_target_name=cur_target_name,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            current_test_target=current_test_target,
            filtered_train_df=filtered_train_df,
            baseline_diagnostics_config=baseline_diagnostics_config,
            cat_features=cat_features,
            metadata=metadata,
        )

        # Dummy baselines diagnostic -- extracted helper
        metadata = run_dummy_baselines(
            target_type=target_type,
            cur_target_name=cur_target_name,
            target_name=target_name,
            model_name=model_name,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            current_test_target=current_test_target,
            filtered_train_df=filtered_train_df,
            filtered_val_df=filtered_val_df,
            test_df_pd=test_df_pd,
            filtered_train_idx=filtered_train_idx,
            filtered_val_idx=filtered_val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            cat_features=cat_features,
            dummy_baselines_config=dummy_baselines_config,
            quantile_regression_config=quantile_regression_config,
            reporting_config=reporting_config,
            _dropped_high_card_data=_dropped_high_card_data,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            plot_file=plot_file,
            metadata=metadata,
            target_by_type=target_by_type,
            _split_preds_probs=_split_preds_probs,
        )

        # Look up the precomputed temporal audit (built ONCE
        # for all targets above the loop via the batch API).
        _audit = _all_target_audits.get(target_type, {}).get(cur_target_name)
        if _audit is not None:
            try:
                logger.info(_format_temporal_audit_report(_audit))
                if (getattr(behavior_config, "target_temporal_audit_save_plot", True)
                        and plot_file):
                    _plot_path = f"{plot_file}_target_temporal_audit.png"
                    _plot_target_over_time(_audit, save_path=_plot_path)
                metadata.setdefault("target_temporal_audit", {}) \
                    .setdefault(str(target_type), {})[cur_target_name] = _audit.to_dict()
            except Exception as _audit_err:
                logger.warning(
                    "target_temporal_audit (per-target render) failed for "
                    "target='%s': %s. Training continues.",
                    cur_target_name, _audit_err,
                )

        if verbose:
            logger.info(f"select_target...")

        t0_select_target = timer()
        # Build common_params and behavior_config for select_target
        od_common_params, current_behavior_config = _build_common_params_for_target(
            common_params_dict=common_params_dict,
            trainset_features_stats=trainset_features_stats,
            plot_file=plot_file,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            outlier_detector=outlier_detector,
            behavior_config=behavior_config,
            fairness_subgroups=fairness_subgroups,
        )

        common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs = select_target(
            model_name=f"{target_name} {model_name} {cur_target_name}",
            target=cur_target_values,  # Full target (for test_target extraction)
            target_type=target_type,
            df=None,
            train_df=filtered_train_df,  # Use pre-filtered DataFrame
            val_df=filtered_val_df,  # Use pre-filtered DataFrame
            test_df=test_df_pd,  # Test set is not filtered by outlier detector
            train_idx=filtered_train_idx,  # Use pre-filtered indices
            val_idx=filtered_val_idx,  # Use pre-filtered indices
            test_idx=test_idx,
            train_details=train_details,
            val_details=val_details,
            test_details=test_details,
            group_ids=group_ids,
            cat_features=cat_features,
            text_features=text_features,
            embedding_features=embedding_features,
            hyperparams_config=hyperparams_config,
            behavior_config=current_behavior_config,
            common_params=od_common_params,
            mlframe_models=mlframe_models,
            linear_model_config=linear_model_config,
            # Fix 3B: forward pre-conversion Polars-side size so
            # configure_training_params skips the 3-min pandas
            # memory_usage(deep=...) scan on frames with high-
            # cardinality object columns. Slight approximation
            # acceptable (only feeds GPU-RAM-fit heuristic);
            # outlier-filter shrinkage is small vs total size.
            train_df_size_bytes=train_df_size_bytes_cached,
            val_df_size_bytes=val_df_size_bytes_cached,
            multilabel_dispatch_config=multilabel_dispatch_config,
        )

    if verbose:
        logger.info("  select_target done in %s", _elapsed_str(t0_select_target))
        log_ram_usage()

    # Build list of pre-pipelines (feature selection methods) to try
    pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
        use_ordinary_models=use_ordinary_models,
        rfecv_models=rfecv_models,
        rfecv_models_params=rfecv_models_params,
        use_mrmr_fs=use_mrmr_fs,
        mrmr_kwargs=mrmr_kwargs,
        custom_pre_pipelines=custom_pre_pipelines,
    )

    # Initialize pipeline cache ONCE - preprocessing output is reused across pre_pipelines.
    # Since custom transformers run AFTER preprocessing, the preprocessing output is the same
    # for ordinary models and all custom pipelines with the same model type (linear, tree, etc).
    pipeline_cache = PipelineCache()

    for pre_pipeline, pre_pipeline_name in tqdmu_lazy_start(zip(pre_pipelines, pre_pipeline_names), desc="pre_pipeline", total=len(pre_pipelines)):
        # Skip CatBoost RFECV pipeline with metamodel_func due to sklearn clone issue
        if _should_skip_catboost_metamodel(pre_pipeline_name.strip(), target_type, behavior_config):
            continue

        # 2026-05-13 (user request): skip identity-equivalent
        # pre_pipelines when an ordinary (no-pipeline) branch or
        # another identity-equivalent selector already covered this
        # target. The marker is set by _apply_pre_pipeline_transforms
        # after the first fit-transform.  If the same MRMR/RFECV
        # instance was identity on a previous target, the marker
        # survives → skip before any model trains.
        _pp_name_stripped = pre_pipeline_name.strip()
        if _pp_name_stripped and getattr(
            pre_pipeline, "_mlframe_identity_equivalent", False
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

        # Build weight schemas from extractor output
        if sample_weights:
            weight_schemas = sample_weights
            if "uniform" in sample_weights:
                logger.info("Using %d weighting schema(s) from extractor: %s", len(weight_schemas), list(weight_schemas.keys()))
            else:
                logger.info("Using %d weighting schema(s) from extractor: %s. Note: uniform weighting not included.", len(weight_schemas), list(weight_schemas.keys()))
        else:
            weight_schemas = {"uniform": None}
            logger.info("No weighting schemas from extractor, defaulting to uniform weighting.")

        # Conflict check: backward val placement + non-uniform weighting.
        # Backward split puts val BEFORE train on the timeline; recency
        # weighting makes train bias toward the NEWEST rows. The two
        # together optimise "fit the newest rows and validate on the
        # oldest ones" -- the opposite of the concept-drift proxy that
        # motivated choosing backward in the first place. Warn at suite
        # entry (rather than silently let the suite run) so the user
        # either disables recency on the extractor or reverts to
        # forward placement deliberately.
        _val_placement = getattr(split_config, "val_placement", "forward")
        if _val_placement == "backward":
            _non_uniform = [k for k in weight_schemas.keys() if k != "uniform"]
            if _non_uniform:
                logger.warning(
                    "  val_placement='backward' is combined with %d non-"
                    "uniform weighting schema(s) %s. Backward val is "
                    "designed to approximate DEPLOYMENT error under "
                    "drift by mirroring the val->train gap against the "
                    "train->prod gap, while recency-style weights bias "
                    "training toward the newest rows. Together they "
                    "optimise 'fit newest, validate on oldest' -- which "
                    "contradicts the drift-proxy intent of backward. "
                    "Consider disabling use_recency_weighting on the "
                    "extractor (runs will fall back to uniform only) "
                    "or switching back to val_placement='forward'.",
                    len(_non_uniform), _non_uniform,
                )

        # -----------------------------------------------------------------------
        # MODEL LOOP: Train each model type with all weight variations
        # Models sorted by feature tier (most features first) so that
        # text/embedding columns are dropped once per tier, not per model.
        # -----------------------------------------------------------------------
        # Resolve strategies once and reuse -- avoids re-calling get_strategy() inside
        # sort key, main loop, and tier-transition check (was ~3x redundant calls).
        # Keyed by id(m) so unhashable estimator instances / tuples don't break lookup,
        # and two identical-by-value-but-distinct-by-identity entries stay distinct.
        strategy_by_model = {id(m): get_strategy(m) for m in mlframe_models}
        sorted_models = sorted(
            mlframe_models,
            key=lambda m: (
                is_neural_model(m),  # push neural to end
                tuple(
                    -int(t) for t in strategy_by_model[id(m)].feature_tier()
                ),  # rich-feature models first
            ),
        )
        tier_dfs_cache = {}  # feature_tier -> {train_df, val_df, test_df}
        # 2026-04-28: Enum-map cache parallel to tier_dfs_cache.
        # XGBoostStrategy.prepare_polars_dataframe needs a leak-free
        # pl.Enum dict built from the train+val UNION (test EXCLUDED
        # to avoid label-time leakage). The map only depends on
        # (feature_tier, strategy class) -- invariant across the
        # weight-schema loop and across multiple models that share
        # a tier. Cache here so we compute it at most once per
        # (target, tier, strategy) instead of once per model.
        tier_enum_map_cache: Dict[tuple, Optional[Dict[str, Any]]] = {}
        prev_tier = None

        # 2026-05-13 (user request): neural max_time defaults to
        # the 95th percentile of non-neural model train times so
        # MLP doesn't run 2h while boosting models finish in 5min.
        # Accumulated as each non-neural model completes.
        _non_neural_train_times: List[float] = []

        _total_models_in_run = len(list(sorted_models))
        _model_idx_in_run = 0
        _break_model_loop = False
        for mlframe_model_name in tqdmu_lazy_start(sorted_models, desc="mlframe model"):
            # Skip CatBoost model with metamodel_func due to sklearn clone issue
            if _should_skip_catboost_metamodel(mlframe_model_name, target_type, behavior_config):
                continue
            _model_idx_in_run += 1
            # 2026-05-10 (rec a): mid-run phase progress so the
            # operator sees fractional progress AND per-model RAM
            # snapshot in real time, instead of waiting for the
            # phase summary at suite end. Logged once at MODEL
            # entry (before the weight-schemas inner loop); the
            # existing per-(model, weight) DONE line at the
            # bottom of the inner loop already carries the
            # post-fit duration + RAM. Together they bracket
            # each model so the operator can see "model 2/4
            # cb starting at RAM=7.2 GB" in the live tail.
            if verbose:
                try:
                    import psutil as _ps
                    _ram_gb_now = _ps.Process().memory_info().rss / (1024 ** 3)
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

            # Use strategy pattern to determine pipeline and cache key
            strategy = strategy_by_model[id(mlframe_model_name)]

            # B5 release (upfront, 2026-04-23 fix): drop the
            # pre-pipeline Polars originals as soon as we hit the
            # FIRST strategy that can't consume them, regardless of
            # ``feature_tier`` change. The original release block
            # below (post-iteration) only fires on a tier transition
            # -- but XGB and LGB share ``tier=(False,False)``, so a
            # ``cb -> xgb -> lgb`` suite kept ``train_df_polars`` /
            # ``val_df_polars`` / ``test_df_polars`` alive through
            # the LGB iteration. Right after that LGB triggered the
            # lazy polars->pandas conversion, holding **both** the
            # 29 GB polars frames and the 57 GB pandas copies
            # simultaneously -- peak 86 GB on the 2026-04-23 prod
            # run, instead of ~57 GB. Releasing here, before the
            # lazy conversion, halves peak RAM in mixed suites.
            if (
                not strategy.supports_polars
                and train_df_polars is not None
            ):
                del train_df_polars, val_df_polars, test_df_polars
                train_df_polars = val_df_polars = test_df_polars = None
                tier_dfs_cache.clear()
                tier_enum_map_cache.clear()
                baseline_rss_mb = maybe_clean_ram_and_gpu(
                    baseline_rss_mb, df_size_mb,
                    verbose=verbose,
                    reason="non-polars-native strategy entry",
                )
                if verbose:
                    logger.info(
                        "  Released pre-pipeline Polars originals before "
                        "%s (non-polars-native strategy) -- frees ~30 %% "
                        "of peak RAM before lazy pandas conversion.",
                        mlframe_model_name,
                    )

            # Clone base_pipeline per model so each iteration gets a
            # fresh un-fitted selector (MRMR / RFECV). Previously, a
            # single fitted ``orig_pre_pipeline`` was shared across
            # all models in the suite -- if CB fit it on train_df
            # and selected 3 of 4 cols, the following Linear iter
            # saw a pipeline where MRMR was already fitted but
            # encoder/imputer/scaler were not. ``_is_fitted`` mis-
            # reported True -> code took the transform-only branch
            # -> imputer.transform raised "feature names should
            # match those passed during fit". Cloning isolates each
            # strategy's fit state. sklearn.base.clone handles
            # feature selectors (they're BaseEstimator-compatible)
            # without copying fitted data -- resets parameters only.
            _base_for_strategy = orig_pre_pipeline
            if _base_for_strategy is not None:
                try:
                    _base_for_strategy = clone(_base_for_strategy)
                except Exception:
                    # Some custom base pipelines (non-BaseEstimator)
                    # don't clone; keep the original reference.
                    pass
            pre_pipeline = strategy.build_pipeline(
                base_pipeline=_base_for_strategy,
                cat_features=cat_features,
                category_encoder=category_encoder if cat_features else None,
                imputer=imputer,
                scaler=scaler,
            )
            # Cache key composition (2026-04-19 round-9 verify):
            #   1. strategy.cache_key (e.g. "tree", "hgb", "neural")
            #   2. pre_pipeline_name (MRMR vs RFECV vs ordinary)
            #   3. feature_tier() -- CRITICAL: CB/LGB/XGB all inherit
            #      cache_key="tree", but CB has tier=(True,True) while
            #      LGB/XGB have tier=(False,False). Without tier in the
            #      key, CB (running first by tier-desc sort) cached its
            #      polars DF with text/embedding cols under "tree";
            #      LGB/XGB then retrieved that cache and received cols
            #      they can't handle (either polars when they need
            #      pandas, or text cols they don't support). Adding
            #      feature_tier() partitions the cache so same-tier
            #      models still share, different-tier models don't
            #      collide.
            _tier_suffix = f"_tier{strategy.feature_tier()}"
            # Container-kind suffix (2026-04-23 fix): CB, XGB, and LGB
            # all inherit ``cache_key="tree"`` and XGB/LGB share the
            # same ``feature_tier``. Without the kind qualifier, XGB
            # (Polars-native) stored the *polars* post-pipeline frame
            # under ``tree_..._tier(False,False)`` and LGB pulled that
            # polars frame out on its next iteration -- undoing the
            # strategy-loop lazy pandas conversion a few lines later
            # and triggering a duplicate 224 s conversion in trainer's
            # self-heal (2026-04-23 prod log, ~38 min wasted). Mirror
            # the kind-tagging already used by ``_build_tier_dfs`` so
            # pandas-consumers and polars-consumers never collide on a
            # shared tier key.
            _kind_suffix = f"_kind{'pl' if strategy.supports_polars else 'pd'}"
            cache_key = (
                f"{strategy.cache_key}_{pre_pipeline_name}{_tier_suffix}{_kind_suffix}"
                if pre_pipeline_name else
                f"{strategy.cache_key}{_tier_suffix}{_kind_suffix}"
            )

            # Polars fastpath: substitute original Polars DataFrames for models
            # that support native Polars input (e.g. CatBoost >= 1.2.7, HGB).
            polars_fastpath_active = train_df_polars is not None and strategy.supports_polars

            # B3: Prepare Polars DFs once per model (outside weight loop).
            # prepare_polars_dataframe() calls .with_columns() which allocates --
            # doing it per weight schema wastes memory for 100GB+ DataFrames.
            if polars_fastpath_active:
                if verbose:
                    logger.info("  Polars fastpath active for %s (strategy=%s)", mlframe_model_name, type(strategy).__name__)
                # Use ``cat_features`` (the post-promotion, deduplicated
                # list from the Phase 3.5 reassignment at line ~1526) --
                # NOT the stale ``cat_features_polars`` captured at the
                # start of Phase 3 before auto-detection ran.
                # Production 2026-04-19 bug: the old
                #   _cat_features = cat_features_polars or cat_features or []
                # short-circuit picked the stale 13-column list even
                # after text-promotion removed 4 of them. Those 4 were
                # later cast Categorical->String for CatBoost's text
                # path, so CB saw cat_features=[..., 'skills_text'] with
                # skills_text being pl.String -- its fused-cpdef
                # ``_set_features_order_data_polars_categorical_column``
                # has no String overload and raised "No matching
                # signature found", burning 22 s + a 150 s pandas
                # fallback on every run.
                _cat_features = list(cat_features or [])

                # Build tier-specific DFs with text/embedding columns dropped for non-supporting models
                tier_base = {
                    "train_df": train_df_polars,
                    "val_df": val_df_polars,
                    "test_df": test_df_polars,
                }
                tier_polars = _build_tier_dfs(
                    tier_base, strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                )

                # 2026-04-28: leak-free pl.Enum map from train+val
                # UNION of unique categorical values (test EXCLUDED).
                # Strategy-class+tier-keyed cache so the map is built
                # once per (target, tier, strategy) and reused across
                # the weight-schema loop and any sibling-tier models.
                _enum_cache_key = (strategy.feature_tier(), type(strategy).__name__)
                if _enum_cache_key in tier_enum_map_cache:
                    _xgb_category_map = tier_enum_map_cache[_enum_cache_key]
                elif hasattr(strategy, "build_polars_enum_map"):
                    try:
                        _xgb_category_map = strategy.build_polars_enum_map(
                            tier_polars["train_df"],
                            tier_polars.get("val_df"),
                            _cat_features,
                        )
                    except Exception as _emb_exc:
                        logger.warning(
                            "build_polars_enum_map failed for %s; "
                            "falling back to per-DF Enum cast: %s",
                            type(strategy).__name__, _emb_exc,
                        )
                        _xgb_category_map = None
                    tier_enum_map_cache[_enum_cache_key] = _xgb_category_map
                else:
                    _xgb_category_map = None
                    tier_enum_map_cache[_enum_cache_key] = None

                prepared_train = _prep_polars_df(tier_polars["train_df"], strategy, _cat_features, _xgb_category_map)
                prepared_val = _prep_polars_df(tier_polars.get("val_df"), strategy, _cat_features, _xgb_category_map)
                prepared_test = _prep_polars_df(tier_polars.get("test_df"), strategy, _cat_features, _xgb_category_map)

                # Null-fill text features for CatBoost (requires no nulls in text columns).
                # Also cast Polars Categorical/Enum to pl.String: CatBoost's Polars
                # fastpath for text_features (`_set_features_order_data_polars_text_column`)
                # rejects Categorical with
                #   "Unsupported data type Categorical for a text feature column".
                # This happens whenever a column was auto-promoted from cat_features to
                # text_features (done in `_auto_detect_feature_types`) but its backing
                # dtype was left as Categorical -- CB's text path requires plain string.
                if text_features and mlframe_model_name == "cb":
                    text_cols_present = filter_existing(prepared_train, text_features)
                    if text_cols_present:
                        # Determine which of the text columns need a dtype cast.
                        needs_cast = [
                            c for c in text_cols_present
                            if prepared_train.schema[c] == pl.Categorical
                            or isinstance(prepared_train.schema[c], pl.Enum)
                        ]
                        prep_exprs = []
                        for c in text_cols_present:
                            expr = pl.col(c)
                            if c in needs_cast:
                                expr = expr.cast(pl.String)
                            prep_exprs.append(expr.fill_null(""))
                        prepared_train = prepared_train.with_columns(prep_exprs)
                        if prepared_val is not None:
                            prepared_val = prepared_val.with_columns(prep_exprs)
                        if prepared_test is not None:
                            prepared_test = prepared_test.with_columns(prep_exprs)
                        if needs_cast and verbose:
                            logger.info(
                                "  Cast %d text feature(s) from Polars Categorical to String "
                                "for CatBoost: %s",
                                len(needs_cast), needs_cast,
                            )

                # Null-in-Categorical fix: applied UPSTREAM once
                # at pre-loop level on train_df_polars/val/test
                # (see the ``_polars_nullable_categorical_cols`` +
                # ``_polars_fill_null_in_categorical`` block near
                # OD-filter, keyword ``__MISSING__``). Every
                # polars-capable strategy -- CB, XGB, HGB -- now
                # operates on the same pre-filled frame. No
                # per-model fill needed in the loop.

                polars_fastpath_skip_preprocessing = strategy.requires_encoding
            else:
                polars_fastpath_skip_preprocessing = False

                # Lazy pandas conversion for non-Polars-native strategies.
                # The "deferred" half of the 2026-04-21 design: when all
                # blockers for the Polars fastpath are non-native
                # strategies, the upfront ``_convert_dfs_to_pandas`` is
                # skipped and per-strategy conversion happens here,
                # preserving RAM when CB/XGB can run natively on polars.
                # In mixed suites (cb+xgb+lgb) this fires once when
                # lgb's iteration is reached.
                # B3 fix (2026-05-11): the previous message blamed the strategy ("non-Polars-native strategy CatBoostStrategy") but the strategy may BE polars-native; the actual trigger is that the polars frames were released earlier in the run (e.g. between targets). Two distinct cases now reported differently:
                #   (a) ``strategy.supports_polars`` is FALSE — really non-native (lgb, linear, etc.).
                #   (b) ``strategy.supports_polars`` is TRUE but ``train_df_polars`` was released — falling back to pandas because the polars originals are gone.
                _logged_lazy_conv = False
                for df_key in ("train_df", "val_df", "test_df"):
                    df_ = common_params.get(df_key)
                    if isinstance(df_, pl.DataFrame):
                        if not _logged_lazy_conv and verbose:
                            if strategy.supports_polars:
                                _reason = (
                                    "Polars originals released "
                                    "(common_params still carries "
                                    "polars frames; converting to "
                                    "pandas for inner predict path)"
                                )
                            else:
                                _reason = (
                                    f"non-Polars-native strategy "
                                    f"{type(strategy).__name__}"
                                )
                            logger.info(
                                "  Lazy pandas conversion for %s -- %s",
                                mlframe_model_name, _reason,
                            )
                            _logged_lazy_conv = True
                        common_params[df_key] = get_pandas_view_of_polars_df(df_)

                # Defense-in-depth (2026-04-23): immediately after the
                # lazy conversion ran above, EVERY ``common_params``
                # DF key must be non-polars. If a Polars frame is
                # still sitting here it means either (a) the loop
                # above silently missed a key (bug in this function)
                # or (b) some later step replaced the converted
                # frame with a polars original between iterations
                # -- which is exactly the 2026-04-23 prod regression
                # that pipeline_cache's kind-suffix fix addresses.
                # The trainer-level hard-raise also catches this,
                # but failing HERE surfaces the bug one function up
                # the stack, closer to the cause (cheaper to debug
                # -- you see ``strategy.__class__.__name__`` and the
                # live ``common_params`` state, not just a model
                # type at fit time).
                for df_key in ("train_df", "val_df", "test_df"):
                    df_ = common_params.get(df_key)
                    if isinstance(df_, pl.DataFrame):
                        raise RuntimeError(
                            f"Lazy pandas conversion produced incomplete "
                            f"state for non-Polars-native strategy "
                            f"{type(strategy).__name__} ({mlframe_model_name}): "
                            f"common_params[{df_key!r}] is still pl.DataFrame "
                            f"(shape={df_.shape}, id={id(df_)}). The lazy-"
                            f"conversion hook iterated over train/val/test but "
                            f"this key escaped. Likely cause: a ``common_params`` "
                            f"override between lazy-conversion and here, or "
                            f"pipeline_cache cross-stream leakage (see core.py "
                            f"kind-suffix in cache_key)."
                        )

                # For non-Polars models, build tier DFs from pandas common_params
                tier_pandas = _build_tier_dfs(
                    {"train_df": common_params.get("train_df"), "val_df": common_params.get("val_df"), "test_df": common_params.get("test_df")},
                    strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                )

            # --- WEIGHT SCHEMA LOOP: Train with each sample weighting ---
            for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items(), desc="weighting schema"):
                # Create model name with weight suffix
                model_name_with_weight = common_params["model_name"]
                model_file_name=f"{mlframe_model_name}"
                if weight_name != "uniform":
                    model_name_with_weight += f" w={weight_name}"
                    model_file_name +=f"_{weight_name}"

                # Shallow copy common_params - only sample_weight changes per iteration
                current_common_params = common_params.copy()
                current_common_params["sample_weight"] = weight_values

                # Apply tier DFs (text/embedding columns dropped for non-supporting models)
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

                # Fix 8: compute per-model input-schema fingerprint and
                # append to model_file_name so two runs with different
                # feature-type configs don't silently overwrite each
                # other. Hashes realised schema (sorted cols + canonical
                # dtypes + roles) of the DF just assigned above -- the
                # actual DF the strategy feeds to model.fit(). Uses the
                # outer cat/text/embedding lists because the per-model
                # fit_params are built later (see extra_fit near the CB
                # branch) and the roles at the DF level are stable across
                # strategies for a given DataFrame.
                _schema_hash, _input_schema = compute_model_input_fingerprint(
                    current_common_params.get("train_df"),
                    cat_features=cat_features,
                    text_features=text_features,
                    embedding_features=embedding_features,
                )
                if getattr(behavior_config, "model_file_hash_suffix", True):
                    model_file_name += f"__sch_{_schema_hash}"

                # Append weight_name to plot_file for non-uniform weights
                if weight_name != "uniform" and current_common_params.get("plot_file"):
                    current_common_params["plot_file"] = current_common_params["plot_file"] + weight_name + "_"

                # Check if we have cached transformed DataFrames for this pipeline type
                cached_dfs = pipeline_cache.get(cache_key)

                # ============================================================================
                # INTENTIONAL: Clone model for EACH weight schema iteration.
                # DO NOT "OPTIMIZE" BY MOVING CLONE OUTSIDE THE LOOP!
                # ============================================================================
                # Each weight schema produces a DIFFERENT trained model that gets stored
                # separately in models[type][target]. Without cloning per iteration:
                #   - All entries would point to the SAME sklearn object (last-trained state)
                #   - In-memory model.model.predict() would give WRONG results
                #   - Only saved .dump files would work correctly (they capture snapshots)
                # The clone() cost is negligible compared to training time.
                # ============================================================================
                original_model = models_params[mlframe_model_name]["model"]
                try:
                    cloned_model = clone(original_model)
                except RuntimeError:
                    # CatBoost wraps custom eval_metric objects internally, causing sklearn's
                    # identity check (param1 is not param2) to fail. Fall back to direct
                    # constructor call with the same params, which produces an equivalent
                    # fresh unfitted model without the verification step.
                    cloned_model = type(original_model)(**original_model.get_params())
                except TypeError:
                    # NGBoost: get_params() exposes attributes (validation_fraction,
                    # early_stopping_rounds) that __init__ doesn't accept. Filter get_params
                    # to those actually in the signature.
                    import inspect as _inspect
                    _cls = type(original_model)
                    _sig_params = set(_inspect.signature(_cls.__init__).parameters) - {"self"}
                    _raw = original_model.get_params(deep=False)
                    cloned_model = _cls(**{k: v for k, v in _raw.items() if k in _sig_params})
                # Preserve the mlframe posthoc calibration tag across clone.
                # sklearn.clone() strips non-param attributes, which would
                # drop the calibration directive and silently revert to an
                # uncalibrated model. Fix 2026-04-15.
                if getattr(original_model, "_mlframe_posthoc_calibrate", False):
                    try:
                        cloned_model._mlframe_posthoc_calibrate = True
                    except Exception:
                        pass
                # Same problem class for the polars-fastpath sticky flag
                # (set defensively in trainer.configure_training_params
                # for every freshly constructed CatBoost instance -- see
                # the 2026-04-24 prod log analysis). sklearn.clone()
                # would otherwise blank it on every weight-schema iter,
                # forcing the dispatch-miss + retry on the FIRST predict
                # of CB recency / CB uniform alike. Re-assert here.
                if getattr(original_model, "_mlframe_polars_fastpath_broken", False):
                    try:
                        cloned_model._mlframe_polars_fastpath_broken = True
                    except Exception:
                        pass
                # XGB DMatrix-reuse shim cache (2026-04-24) AND
                # LGB Dataset-reuse shim cache (2026-05-08). Both
                # shims cache the heavy binned dataset on instance
                # attributes (``_cached_train_dmatrix`` /
                # ``_cached_train_dataset`` and their val/key
                # siblings); sklearn.clone() blanks them. Without
                # this hand-off, the weight-schema loop (uniform ->
                # recency on the same train_df) would rebuild the
                # dataset from scratch on every iteration -- defeating
                # the whole point of the shim. Hand the cache forward
                # so reused dataset sees consecutive ``set_label`` /
                # ``set_weight`` swaps in place.
                for _attr in (
                    "_cached_train_dmatrix",
                    "_cached_train_key",
                    "_cached_val_dmatrix",
                    "_cached_val_key",
                    "_cached_train_dataset",
                    "_cached_val_dataset",
                ):
                    if hasattr(original_model, _attr):
                        try:
                            setattr(cloned_model, _attr, getattr(original_model, _attr))
                        except Exception:
                            pass
                current_model_params = models_params[mlframe_model_name].copy()
                current_model_params["model"] = cloned_model

                # Polars fastpath: update cat_features/text_features/embedding_features
                # in fit_params for CatBoost only.
                # XGBoost/HGB auto-detect pl.Categorical via enable_categorical=True
                # and do NOT accept cat_features/text_features/embedding_features as fit() params.
                if polars_fastpath_active and mlframe_model_name == "cb" and "fit_params" in current_model_params:
                    extra_fit = {}
                    if _cat_features:
                        _valid_cat = _filter_polars_cat_features_by_dtype(
                            prepared_train, _cat_features
                        )
                        if _valid_cat:
                            extra_fit["cat_features"] = _valid_cat
                    if text_features:
                        cb_text = filter_existing(prepared_train, text_features)
                        if cb_text:
                            extra_fit["text_features"] = cb_text
                    if embedding_features:
                        cb_emb = filter_existing(prepared_train, embedding_features)
                        if cb_emb:
                            extra_fit["embedding_features"] = cb_emb
                    if extra_fit:
                        current_model_params["fit_params"] = {**current_model_params["fit_params"], **extra_fit}

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
                    # Fix 11 (2026-04-22): per-strategy compute of
                    # whether the Polars-ds pipeline already did the
                    # preprocessing this strategy would have needed.
                    #
                    # Previously ``polars_pipeline_applied or
                    # polars_fastpath_skip_preprocessing`` accumulated
                    # globally across the suite loop. The initial
                    # value at core.py:1995 is True when the input is
                    # Polars and the polars-ds pipeline exists -- so
                    # every later iteration (including Linear, which
                    # really DOES need the encoder/scaler/imputer)
                    # inherited True and skipped its pre_pipeline
                    # via train_eval.py:675 -> trainer.py:775
                    # ``elif skip_preprocessing: feature_selector
                    # only`` branch. LogReg then received raw
                    # pd.Categorical and crashed on 'HOURLY'.
                    #
                    # Correct semantics: the preprocessing is
                    # already done for THIS strategy iff
                    #   (a) the initial polars-ds pipeline ran at
                    #       the suite level (``polars_pipeline_applied``
                    #       as seeded before the loop), AND
                    #   (b) this strategy itself can consume
                    #       Polars frames (``supports_polars``).
                    #
                    # requires_encoding=True is NOT a sufficient
                    # trigger to re-run preprocessing: HGB declares
                    # requires_encoding=True for its pandas-fallback
                    # path only, but on the Polars fastpath HGB
                    # consumes pl.Categorical natively (no encoder
                    # needed). Gating on supports_polars alone is
                    # correct -- only the non-Polars-native
                    # strategies (Linear, Neural, sklearn-generic,
                    # LGB-via-bridge) fall through to their own
                    # pre_pipeline run in trainer.py.
                    #
                    # 2026-04-23 extension (fuzz c0003 / HGB +
                    # polars_enum + prefer_polarsds=False):
                    # when the shared polars-ds pipeline does NOT
                    # run, Fix 11's left-hand side collapses to
                    # False -- the gate was then False for HGB too,
                    # forcing pre_pipeline (with a sklearn
                    # category_encoders encoder) to fit on a
                    # pl.DataFrame and crash at convert_inputs.
                    # The polars fastpath being ACTIVE for this
                    # strategy is itself a sufficient reason to
                    # skip sklearn preprocessing -- the strategy
                    # consumes the Polars frame natively, so the
                    # encoder/scaler/imputer would both be
                    # redundant and crash on Polars input.
                    # ``polars_fastpath_active`` already encodes
                    # (train_df_polars is not None AND
                    # strategy.supports_polars), so this disjunct
                    # keeps the supports_polars-required invariant.
                    polars_pipeline_applied=(
                        (polars_pipeline_applied and strategy.supports_polars)
                        or polars_fastpath_active
                    ),
                    mlframe_model_name=mlframe_model_name,
                    metadata_columns=metadata.get("columns"),
                )

                # 2026-05-13 (user request): neural-model max_time
                # defaults to the P95 of all prior non-neural model
                # train times (seconds). If no non-neural model has
                # trained yet, keep the config-level max_time (2h).
                _is_neural = is_neural_model(mlframe_model_name)
                if _is_neural and _non_neural_train_times:
                    _p95 = float(np.percentile(_non_neural_train_times, 95))
                    # Floor: 5 min (300 s) so LightGBM/CB don't
                    # produce a sub-minute P95 that rounds to
                    # 0h0m → Lightning stops immediately.
                    _max_sec = max(int(round(_p95)), 300)
                    _dd = _max_sec // 86400
                    _hh = (_max_sec % 86400) // 3600
                    _mm = (_max_sec % 3600) // 60
                    _ss = _max_sec % 60
                    _max_time_dict = {"days": _dd, "hours": _hh, "minutes": _mm, "seconds": _ss}
                    # MLP is Pipeline(StandardScaler, TTR(PytorchLightningRegressor(...)))
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
                                    mlframe_model_name, _hh, _mm, _ss,
                                    len(_non_neural_train_times), _p95,
                                )

                t0_model = timer()
                try:
                    with phase("process_model", model=mlframe_model_name, weight=weight_name):
                        trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                            **process_model_kwargs
                        )
                except Exception as model_err:
                    # Skip-and-continue only when the caller explicitly opted in.
                    # KeyboardInterrupt is intentionally NOT caught -- Ctrl-C must
                    # still abort the run. A native SIGSEGV that kills the process
                    # also won't be caught here; only Python-level exceptions
                    # (XGBoostError, CatBoostError, MemoryError, ...) are.
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

                # 2026-05-13 (user request): after the FIRST model
                # under this pre_pipeline completes, check whether
                # the pre_pipeline is identity-equivalent (kept all
                # columns, created none).  If it is AND ordinary
                # models are in the suite, every remaining model
                # would train on identical data → skip.
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

                # XGB DMatrix-reuse shim cache (2026-04-24, second
                # half). The cache lives on the cloned_model that
                # ``process_model`` actually fit. Hand it back to the
                # template (``original_model`` / ``models_params[...]
                # ["model"]``) so the NEXT weight-schema iteration's
                # ``clone()`` carries the cache forward (via the
                # forward-transfer block above). Without this, the
                # cache would be born and die inside one weight-
                # schema iteration -- wasted because the SAME train
                # frame is consumed by the next iteration with only
                # a different sample_weight. Symmetric counterpart
                # to the forward-transfer block in this loop.
                for _attr in (
                    "_cached_train_dmatrix",
                    "_cached_train_key",
                    "_cached_val_dmatrix",
                    "_cached_val_key",
                    "_cached_train_dataset",
                    "_cached_val_dataset",
                ):
                    if hasattr(cloned_model, _attr):
                        _val = getattr(cloned_model, _attr)
                        if _val is not None:
                            try:
                                setattr(original_model, _attr, _val)
                            except Exception:
                                pass

                # Fix 8: record this model's input-schema fingerprint
                # in metadata so load-time can verify or at least diff
                # against the serving data. Keyed by the final
                # model_file_name (already includes weight + hash), so
                # repeat weights don't collide.
                # 2026-04-24: extend with target_type / n_classes /
                # multilabel_strategy + schema_version to support
                # multi-output target inference at load time. Old
                # artifacts without these fields fall through to
                # legacy binary inference (load_mlframe_suite).
                _record = {
                    "schema_hash": _schema_hash,
                    "input_schema": _input_schema,
                    "mlframe_model": mlframe_model_name,
                    "weight_name": weight_name,
                    # Multi-output extensions:
                    "target_type": str(target_type) if target_type is not None else None,
                    "schema_version": 2,  # 1=legacy, 2=multi-output-aware
                }
                try:
                    from ..configs import TargetTypes as _TT
                    if target_type == _TT.MULTILABEL_CLASSIFICATION:
                        # Number of label outputs and dispatch strategy
                        _record["n_classes"] = (
                            int(train_y.shape[1])
                            if hasattr(train_y, "shape") and train_y.ndim == 2
                            else None
                        )
                        _record["multilabel_strategy"] = "native" if (
                            hasattr(strategy, "supports_native_multilabel") and strategy.supports_native_multilabel
                        ) else "wrapper"
                    elif target_type == _TT.MULTICLASS_CLASSIFICATION:
                        _record["n_classes"] = (
                            int(len(np.unique(np.asarray(train_y))))
                            if hasattr(train_y, "shape") else None
                        )
                        _record["multilabel_strategy"] = None
                    else:
                        _record["n_classes"] = None
                        _record["multilabel_strategy"] = None
                except Exception:
                    # Defensive -- never fail the metadata write because
                    # of an introspection problem on multi-output fields.
                    pass
                metadata.setdefault("model_schemas", {})[model_file_name] = _record

                # Cache the transformed DataFrames if not already cached
                if cached_dfs is None:
                    pipeline_cache.set(cache_key, train_df_transformed, val_df_transformed, test_df_transformed)

            # Update orig_pre_pipeline for tree models only.
            # Tree models return just the base_pipeline (feature selector) from build_pipeline(),
            # so after process_model() fits it, we preserve the fitted version for subsequent models.
            # Non-tree models wrap base_pipeline in a full Pipeline (with encoder/imputer/scaler),
            # which we don't want to use as the base for other model types.
            # For optimal performance, list tree models first in mlframe_models.
            if cache_key.startswith("tree"):
                orig_pre_pipeline = pre_pipeline

            # Release XGB / LGB shim cache memory at strategy-iter end
            # (2026-04-24 follow-up; LGB shim added 2026-05-08).
            # Both shims cache the heavy binned dataset on
            # ``_cached_train_*`` / ``_cached_val_*`` instance attrs
            # as a fit-only scratchpad used by the inner
            # weight-schema loop (uniform -> recency swap in place
            # via ``set_label`` / ``set_weight``).
            # Once the inner loop has finished, nothing downstream
            # reads those attrs:
            #   * ensemble scoring (below) uses pre-computed probs
            #     stored on the SimpleNamespace wrappers in
            #     ``ens_models`` -- NOT ``.predict()`` calls;
            #   * ``.predict`` / ``.predict_proba`` route through
            #     ``_Booster`` (attached at fit end), not the cache;
            #   * model save goes through pickle / joblib, which
            #     our ``__getstate__`` override strips the cache
            #     from anyway.
            # 2026-05-13: if the first model under this pre_pipeline
            # was identity-equivalent, break the model loop now.
            if _break_model_loop:
                break

            # The prod log showed a 7.3M x 105 frame holding ~8 GB
            # of QuantileDMatrix memory per XGB iteration; LGB
            # Dataset binning is similarly heavy. Releasing this
            # cache between strategies frees ~30 % of peak RAM.
            # Duck-typed: only the shims expose ``clear_cache()``.
            # CB / sklearn estimators skip harmlessly via the
            # ``callable`` check.
            # Template held under models_params; release its cache.
            _maybe_clear_shim_cache(original_model)
            # Each previously-fitted model snapshot parked in
            # ``ens_models`` may also hold a cache ref (forward-
            # transfer at clone copied the reference, not moved it).
            # Release on all -- their probs are already recorded
            # and the cache is not read anywhere else.
            if ens_models:
                for _ens_ns in ens_models:
                    _maybe_clear_shim_cache(getattr(_ens_ns, "model", None))

            # B5: Release Polars originals after all tier-1 (Polars-native) models finish.
            # When transitioning to a lower tier, pre-pipeline Polars DFs are no longer needed.
            cur_tier = strategy.feature_tier()
            if prev_tier is not None and cur_tier != prev_tier and not strategy.supports_polars:
                if train_df_polars is not None:
                    del train_df_polars, val_df_polars, test_df_polars
                    train_df_polars = val_df_polars = test_df_polars = None
                    tier_dfs_cache.clear()
                    tier_enum_map_cache.clear()
                    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="tier transition")
                    if verbose:
                        logger.info("  Released pre-pipeline Polars originals (tier transition)")
            prev_tier = cur_tier

        if ens_models and len(ens_models) > 1:
            if verbose:
                logger.info(f"evaluating simple ensembles...")
            # Get feature count from transformed DataFrame for display
            ens_n_features = train_df_transformed.shape[1] if train_df_transformed is not None else None
            # Name the ensemble by its actual members so post-hoc log
            # grep shows *which* models participated. The old
            # ``"{N}models "`` label hid dropouts -- in the 2026-04-23
            # prod log LGB silently failed, yet the ensemble was
            # reported as "2models" with no hint that it was just
            # CB+XGB. Cap the list to keep headers readable:
            #   <=4 members -> "[cb+xgb+lgb]"
            #   >4        -> "[N=5]" (avoid bloated titles)
            # 2026-05-11 (user request): delegate to the shared ``short_model_tag`` helper, which ALSO strips the internal shim suffixes (``WithDMatrixReuse`` / ``WithDatasetReuse``) so ``LinearRegression`` stays as ``LinearRegression`` in the label, but ``XGBRegressorWithDMatrixReuse`` collapses to ``xgb`` (was already correct for tree families via prefix match; this also handles any future shimmed non-tree class cleanly).
            from .._format import short_model_tag as _short_tag_fn
            _member_tags = [_short_tag_fn(getattr(m, "model", m)) for m in ens_models]
            if len(_member_tags) <= 4:
                _members_label = "[" + "+".join(_member_tags) + "]"
            else:
                _members_label = f"[N={len(_member_tags)}]"
            # 2026-05-11 (user request): honour ``behavior_config.confidence_ensemble_quantile`` so users can disable the "Conf Ensemble" output entirely by setting the quantile to 0.0 -- previously hard-coded at the ``score_ensemble`` default 0.1 which produced 6 flavor x 2 split = 12 noisy log blocks per ensemble pass.
            _conf_q = float(getattr(behavior_config, "confidence_ensemble_quantile", 0.1))
            _ensembles = score_ensemble(  # Result used for side effects (logging/metrics)
                models_and_predictions=ens_models,
                ensemble_name=f"{pre_pipeline_name}{_members_label} ",
                n_features=ens_n_features,
                uncertainty_quantile=_conf_q,
                **common_params,
            )

    # ==================================================================================

    # Write back modified state to ctx
    ctx.models = models
    ctx.metadata = metadata
    ctx.trainset_features_stats = trainset_features_stats
    ctx.slug_to_original_target_type = slug_to_original_target_type
    ctx.slug_to_original_target_name = slug_to_original_target_name
    ctx._non_neural_train_times = _non_neural_train_times
    ctx.train_df_polars = train_df_polars
    ctx.val_df_polars = val_df_polars
    ctx.test_df_polars = test_df_polars
    ctx.train_df_pd = train_df_pd
    ctx.val_df_pd = val_df_pd
    ctx.test_df_pd = test_df_pd
    ctx.filtered_train_df = filtered_train_df
    ctx.filtered_val_df = filtered_val_df
    ctx.pipeline = pipeline
    ctx.can_skip_pandas_conv = can_skip_pandas_conv
    ctx.baseline_rss_mb = baseline_rss_mb
    ctx.train_df_size_bytes_cached = train_df_size_bytes_cached
    ctx.val_df_size_bytes_cached = val_df_size_bytes_cached
