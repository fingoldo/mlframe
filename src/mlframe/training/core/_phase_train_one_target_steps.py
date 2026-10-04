"""Helpers carved out of ``_phase_train_one_target_body`` to keep that module under its size budget."""

from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from sklearn.base import clone

from pyutilz.system import tqdmu_lazy_start

from ..phases import phase
from ..models import is_neural_model
from ..train_eval import process_model
from ..utils import compute_model_input_fingerprint, filter_existing
from ._misc_helpers import (
    _compute_neural_max_time,
    _elapsed_str,
    _filter_polars_cat_features_by_dtype,
    _maybe_clear_shim_cache,
)
from ._setup_helpers import (
    _build_process_model_kwargs,
    _entry_not_for_target,
    _should_skip_catboost_metamodel,
)
from ._phase_train_one_target_polars_fastpath import _prepare_strategy_inputs
from ._phase_train_one_target_schema import (
    _build_and_record_model_schema,
    _clone_model_with_sticky_flags,
)
from ._phase_train_one_target_mlp_helpers import (
    _apply_mlp_extreme_ar_output_activation,
    _apply_mlp_extreme_ar_weight_decay_bump,
    _drop_columns_for_mlp,
    _identify_per_group_columns,
)
from ._phase_train_one_target_post import (
    _evaluate_mlp_extreme_ar_gate,
    _forward_selector_sticky_attrs,  # re-exported for back-compat (test_weight_aware_fs_suite imports it from this module)
    _run_per_model_post_train_tail,
)
from ._phase_train_one_target_cache_helpers import (
    compute_cached_model_input_fingerprint,
    compute_model_pipeline_cache_key,
)
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _clone_base_pipeline_for_strategy(base_pipeline):
    """Fresh un-fitted clone of ``base_pipeline`` for one strategy, with the selector's sticky attrs forwarded.

    Custom non-BaseEstimator pipelines cannot be sklearn-cloned; the original reference is then reused, which is correct only for a stateless pipeline or one with
    its own per-call reset, so the fallback is logged with the pipeline type. A partially-fit selector shared across strategies trips ``imputer.transform`` on a
    feature-names mismatch.
    """
    if base_pipeline is None:
        return None
    try:
        cloned = clone(base_pipeline)
        _forward_selector_sticky_attrs(base_pipeline, cloned)
        return cloned
    except Exception as clone_err:
        log_throttle(
            logger,
            "train_one_target_clone_base_pipeline_failed",
            logging.WARNING,
            "  sklearn.clone failed for base_pipeline (%s); reusing "
            "original reference. If %s is a stateful selector with "
            "no per-call reset, downstream `pre_pipeline.fit` may "
            "see stale state from a prior model in the suite.",
            clone_err,
            type(base_pipeline).__name__,
        )
        return base_pipeline


def _train_one_target_ste_step1_model_entry_tqdmu(sorted_models, target_type, st, cur_target_name, _model_idx_in_run, _total_models_in_run, ctx, pre_pipeline_name, strategy_by_model, tier_dfs_cache, tier_enum_map_cache, orig_pre_pipeline, prepared_frames_cache, weight_schemas, ens_models, cur_target_values, _pp_name_stripped, prev_tier, pre_pipeline):
    """Step 1 of _train_one_target_step1_progress_long_suites: lines starting at ``for _model_entry in tqdmu_lazy_start(sorted_models, desc="mlframe mode``."""
    from mlframe.training.core._phase_train_one_target import (
        _build_feature_selection_report,
        _capture_dataset_reuse_cache,
        _compute_pipeline_cache_key,
        _forward_dataset_reuse_cache,
        _restore_dataset_reuse_cache,
        _selector_params_hash,
        _unwrap_selector,
    )

    _break_model_loop = False
    for _model_entry in tqdmu_lazy_start(sorted_models, desc="mlframe model"):
        # ``_model_entry`` is the raw mlframe_models entry (string tag, estimator instance, or
        # ``(name, estimator)`` tuple) -- it is the key into ``models_params`` / ``strategy_by_model``.
        # ``mlframe_model_name`` is the human/string label used everywhere a string is required
        # (model_category, file names, logging, ``== "mlp"`` gates); for string tags the two coincide.
        if isinstance(_model_entry, tuple) and len(_model_entry) == 2 and isinstance(_model_entry[0], str):
            mlframe_model_name = _model_entry[0]
        elif isinstance(_model_entry, str):
            mlframe_model_name = _model_entry
        else:
            mlframe_model_name = type(_model_entry).__name__
        if _should_skip_catboost_metamodel(mlframe_model_name, target_type, st.behavior_config) or _entry_not_for_target(_model_entry, cur_target_name):
            continue
        # Extreme-AR + group-aware MLP trigger predicate (shared by 3 protections: skip /
        # drop per-group aggregate cols / bump weight_decay 100x). Computed once per (target,
        # model) so the three protections agree on whether to fire. The gate body + its full
        # rationale are carved into ``_evaluate_mlp_extreme_ar_gate``; the ``continue`` stays
        # here so phase ordering is unchanged.
        _ea_skip, _mlp_extreme_ar_fired, _mlp_ea_lag1 = _evaluate_mlp_extreme_ar_gate(
            mlframe_model_name=mlframe_model_name,
            cur_target_name=cur_target_name,
            behavior_config=st.behavior_config,
            metadata=st.metadata,
            _model_idx_in_run=_model_idx_in_run,
            _total_models_in_run=_total_models_in_run,
        )
        if _ea_skip:
            continue
        _model_idx_in_run += 1
        if st.verbose:
            # Per-model RSS sample is intentional: localising OOM-blame to a specific
            # model+target+pre_pipeline tuple in the verbose-suite log saves hours of post-mortem
            # log-correlation. The ~3ms/call Windows cost is dwarfed by per-model fit times.
            # PSUTIL-IMPORT-HOT: ``psutil`` is now imported at module level (``_ps_module``);
            # the prior in-loop import paid ImportError lookup costs on every iter.
            try:
                # Same measure as every other "RAM usage" line: a raw rss read here reported 6.2GB one line
                # after the suite printed 45.2GB, because on Windows rss is the working set and clean_ram
                # evicts it.
                from mlframe.training._ram_helpers import get_reported_memory_gb, memory_measure_name

                _ram_gb_now = get_reported_memory_gb()
                _ram_measure = memory_measure_name()
            except Exception as e:
                logger.debug("memory probe failed: %s", e)
                _ram_gb_now = 0.0
                _ram_measure = "?"
            logger.info(
                "  process_model(%s) START -- model %d/%d, RAM=%.1fGB (%s)",
                mlframe_model_name,
                _model_idx_in_run,
                _total_models_in_run,
                _ram_gb_now,
                _ram_measure,
            )

        if _model_entry not in st.models_params:
            log_throttle(logger, "train_one_target_model_not_known", logging.WARNING, "mlframe model %s not known, skipping...", mlframe_model_name)
            continue

        # Cross-target dataset reuse: restore the prior target's _DATASET_REUSE_CACHE_ATTRS
        # snapshot onto the freshly-built model template BEFORE the weight loop's clone()
        # forward-transfer reads them. select_target() rebuilds models_params per target so
        # the cache attributes are absent on a virgin template - without this restore the
        # XGB/LGB shims would rebuild the binned dataset on target 2. The shim's
        # signature_of(X) check then matches against the same ctx-pinned train_df pointer
        # and triggers set_label / set_weight in place rather than a fresh build.
        _restore_dataset_reuse_cache(
            ctx,
            mlframe_model_name,
            st.models_params[_model_entry]["model"],
            pp_name=pre_pipeline_name,
        )

        strategy = strategy_by_model[id(_model_entry)]

        # Drop pre-pipeline Polars originals as soon as we hit the first non-Polars strategy. The
        # post-iteration release fires only on tier transitions, but same-tier siblings (e.g. XGB and
        # LGB share tier=(False,False)) would keep Polars frames alive into a lazy pandas conversion,
        # doubling peak RAM. Releasing upfront halves peak in mixed suites.
        _train_one_target_ste_step1_doubling_peak_ram(strategy, st, tier_dfs_cache, tier_enum_map_cache, ctx, mlframe_model_name)

        # Clone the base_pipeline per model so each iteration gets a fresh, un-fitted selector. Sharing a
        # fitted MRMR/RFECV across strategies caused `_is_fitted` to misreport True for a partially-fit
        # pipeline (selector fitted but encoder/imputer/scaler not), tripping imputer.transform on a
        # feature-names mismatch.
        _base_for_strategy = _clone_base_pipeline_for_strategy(orig_pre_pipeline)
        pre_pipeline = strategy.build_pipeline(
            base_pipeline=_base_for_strategy,
            cat_features=st.cat_features,
            category_encoder=st.category_encoder if st.cat_features else None,
            imputer=st.imputer,
            scaler=st.scaler,
            embedding_features=st.embedding_features,
            text_features=st.text_features,
        )
        # Cache key = strategy.cache_key + pre_pipeline_name + feature_tier + container kind + feature-list digest.
        # feature_tier is required because CB/LGB/XGB all share cache_key="tree" but have different
        # tiers; without it, CB's text/embedding-bearing frame would be served to LGB/XGB.
        # Kind suffix prevents Polars-native (XGB) and pandas-only (LGB) consumers from sharing entries
        # within a tier, which would otherwise undo the lazy pandas conversion downstream.
        # See _compute_pipeline_cache_key for the features-digest contract (frozenset, order-invariant).
        # Pass the polars train frame (if present) so dtype changes between targets / runs
        # invalidate the cache; for a non-polars strategy, train_df_pd is passed instead so the
        # same dtype/schema discriminator applies to pandas-consuming strategies too (see
        # compute_model_pipeline_cache_key's docstring for the incident this closes).
        cache_key = compute_model_pipeline_cache_key(
            strategy=strategy,
            pre_pipeline_name=pre_pipeline_name,
            cat_features=st.cat_features,
            text_features=st.text_features,
            embedding_features=st.embedding_features,
            train_df_polars=st.train_df_polars,
            train_df_pd=st.train_df_pd,
            cur_target_name=cur_target_name,
            current_train_target=st.current_train_target,
            _compute_pipeline_cache_key=_compute_pipeline_cache_key,
        )

        # Polars fastpath substitutes original Polars DataFrames for natively-Polars consumers
        # (CatBoost >= 1.2.7, HGB). Polars DFs are prepared once per model (outside the weight loop)
        # because prepare_polars_dataframe() allocates via .with_columns().
        polars_fastpath_active = st.train_df_polars is not None and strategy.supports_polars

        _prep_out = _prepare_strategy_inputs(
            polars_fastpath_active=polars_fastpath_active,
            mlframe_model_name=mlframe_model_name,
            strategy=strategy,
            cat_features=st.cat_features,
            text_features=st.text_features,
            embedding_features=st.embedding_features,
            train_df_polars=st.train_df_polars,
            val_df_polars=st.val_df_polars,
            test_df_polars=st.test_df_polars,
            prepared_frames_cache=prepared_frames_cache,
            tier_dfs_cache=tier_dfs_cache,
            tier_enum_map_cache=tier_enum_map_cache,
            common_params=st.common_params,
            pre_pipeline_name=pre_pipeline_name,
            ctx=ctx,
            verbose=st.verbose,
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
        # FP-KEY-OMITS-CONTENT: the key folds ``id(train_df)`` (strong-ref-pinned at this point) so
        # two different per-target frames hitting the same strategy/tier/kind/pp_name combination
        # never replay target 1's stale schema hash for target 2.
        _schema_hash, _input_schema = compute_cached_model_input_fingerprint(
            ctx=ctx,
            polars_fastpath_active=polars_fastpath_active,
            prepared_train=prepared_train,
            tier_pandas=tier_pandas,
            strategy=strategy,
            pre_pipeline_name=pre_pipeline_name,
            cat_features=st.cat_features,
            text_features=st.text_features,
            embedding_features=st.embedding_features,
            compute_model_input_fingerprint=compute_model_input_fingerprint,
        )

        # Per-PP NGBoost-fallback invariant: when ``clone(original_model)`` raises ``TypeError`` (NGBoost: ``get_params`` exposes non-constructor
        # attrs), the fallback path re-pays ``original_model.get_params(deep=False)`` + ``{k:v for k in sig}`` once per weight iteration. The snapshot
        # is invariant across weights (only ``sample_weight`` differs at fit-time), so cache it once outside the loop. Lazy: compute on first use to
        # avoid paying for models that don't hit the TypeError path.
        _ngb_fallback_snapshot: dict | None = None

        # Per-PP CB extras invariants: ``_filter_polars_cat_features_by_dtype`` + ``filter_existing(text/embedding)`` depend only on ``prepared_train`` +
        # the (cat/text/embedding) feature lists -- all invariant across the weight loop. Compute once here so the inner loop only stitches the result
        # into ``current_model_params["fit_params"]`` (avoids paying the dtype filter + ``filter_existing`` 3 scans per weight iteration).
        _cb_extra_fit_invariant: dict[str, Any] | None = None
        _cb_extra_fit_invariant = _train_one_target_ste_step2_into_current_model(polars_fastpath_active, mlframe_model_name, _cat_features, prepared_train, st, _cb_extra_fit_invariant)

        # Neural (MLP / ranker): the tabular Lightning models have no native embedding/text input layers, so the
        # estimator expands embedding-List + HF-embeds text columns itself at fit/predict. Thread the (prepared-frame
        # present) feature lists into fit_params so the estimator knows which columns to encode; invariant across the
        # weight loop. ``_encode_emb_text_fit`` pops these keys before they reach Lightning.
        # ``cat_features`` is threaded for the flat MLP (mlframe_model_name=="mlp") AND the recurrent tabular models (lstm/gru/rnn/transformer):
        # both factorize raw cats + learn an nn.Embedding per cat at their fit boundary when the strategy left them un-encoded
        # (requires_encoding=False via the learnable-cat-embeddings knob). For recurrent, this only matters in HYBRID / FEATURES_ONLY where a
        # tabular block exists; SEQUENCE_ONLY has no aux cats so the wrapper no-ops cleanly even if the kwarg is threaded. NGB shares the neural
        # strategy but cannot consume learnable embeddings, so it must NOT receive raw cat_features.
        _RECURRENT_MODEL_NAMES = ("lstm", "gru", "rnn", "transformer")
        _neural_threads_cats = mlframe_model_name in ("mlp", *_RECURRENT_MODEL_NAMES) and not getattr(strategy, "requires_encoding", True)
        _neural_extra_fit_invariant: dict[str, Any] | None = None
        _neural_extra_fit_invariant = _train_one_target_ste_step3_strategy_but_cannot(strategy, st, _neural_threads_cats, prepared_train, _neural_extra_fit_invariant)

        for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items(), desc="weighting schema"):
            cached_dfs, cloned_model, model_file_name, model_name_with_weight, original_model, pre_pipeline = _train_one_target_ste_step1_model_name_weight(st, mlframe_model_name, weight_name, weight_values, polars_fastpath_active, prepared_train, prepared_val, prepared_test, tier_pandas, cur_target_name, _schema_hash, cache_key, _model_entry, _ngb_fallback_snapshot, pre_pipeline)
            # Isolation copy: each weight iteration installs its own cloned_model and may
            # patch fit_params (CatBoost text/embedding fastpath); without copying we would
            # mutate the suite-level models_params template and the next target would inherit
            # this iteration's overrides.

            # MULTI_TARGET_REGRESSION build-time wiring.
            # Two things to do BEFORE the cloned_model lands in
            # current_model_params:
            #   * Native strategies (CatBoost / XGBoost): inject the
            #     library-specific objective kwargs (e.g.
            #     loss_function="MultiRMSE", multi_strategy="multi_output_tree")
            #     via set_params so the constructed regressor knows it's
            #     fitting (N, K) targets.
            #   * Non-native strategies (LightGBM / HGB): wrap the
            #     cloned model in sklearn.multioutput.MultiOutputRegressor
            #     so K independent fits stack into the (N, K) output.
            # The MLP estimator auto-detects (N, K) at fit-time so this
            # block is a no-op for "mlp" (NeuralNetStrategy.
            # supports_native_multi_target=True + empty kwargs).
            _is_neural, _timeout, cloned_model, current_model_params, process_model_kwargs = _train_one_target_ste_step2_supports_native_multi(target_type, _model_entry, cloned_model, mlframe_model_name, st, _cb_extra_fit_invariant, _neural_extra_fit_invariant, _mlp_extreme_ar_fired, model_name_with_weight, model_file_name, pre_pipeline, pre_pipeline_name, cur_target_name, ens_models, cached_dfs, strategy, polars_fastpath_active)
            _train_one_target_ste_step3_timeout_none(_timeout, current_model_params, st, mlframe_model_name)

            t0_model = timer()
            try:
                with phase("process_model", model=mlframe_model_name, weight=weight_name):
                    st.trainset_features_stats, pre_pipeline, st.train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                        **process_model_kwargs
                    )
            except Exception as model_err:
                # Skip-and-continue is opt-in. KeyboardInterrupt is intentionally not caught here;
                # native SIGSEGV that kills the process won't be caught either.
                if not st.behavior_config.continue_on_model_failure:
                    raise
                log_throttle(
                    logger, "train_one_target_process_model_failed", logging.ERROR,
                    "  process_model(%s, w=%s) FAILED after %s -- %s: %s. continue_on_model_failure=True -> skipping and moving on.",
                    mlframe_model_name,
                    weight_name,
                    _elapsed_str(t0_model),
                    type(model_err).__name__,
                    model_err,
                    exc_info=True,
                )
                st.metadata.setdefault("failed_models", []).append(
                    {
                        "model": mlframe_model_name,
                        "weighting": weight_name,
                        "error_type": type(model_err).__name__,
                        "error_message": str(model_err),
                    }
                )
                continue  # next weight_name in the inner loop
            if st.verbose:
                logger.info("  process_model(%s, w=%s) done -- %s", mlframe_model_name, weight_name, _elapsed_str(t0_model))
            if not _is_neural and t0_model is not None:
                st._non_neural_train_times.append(timer() - t0_model)
            # Per-model post-train tail (TTA uncertainty eval + composite
            # y-scale emit + adaptive RAM reclaim) carved into
            # ``_run_per_model_post_train_tail``; it mutates ``metadata`` /
            # ``ctx.models`` in place so nothing is returned.
            _run_per_model_post_train_tail(
                behavior_config=st.behavior_config,
                test_df_transformed=test_df_transformed,
                current_test_target=st.current_test_target,
                ctx=ctx,
                target_type=target_type,
                cur_target_name=cur_target_name,
                mlframe_model_name=mlframe_model_name,
                metadata=st.metadata,
                test_df_pd=st.test_df_pd,
                _train_idx=st._train_idx,
            )

            # Hand the dataset-reuse cache from cloned_model back to the template so the next
            # weight-schema iteration's clone() carries it forward (symmetric to the forward-transfer
            # block above). Without this the cache would be born and die in a single iteration.
            _forward_dataset_reuse_cache(cloned_model, original_model, skip_none=True)

            _build_and_record_model_schema(
                ctx=ctx,
                metadata=st.metadata,
                model_file_name=model_file_name,
                mlframe_model_name=mlframe_model_name,
                weight_name=weight_name,
                target_type=target_type,
                strategy=strategy,
                cur_target_name=cur_target_name,
                cur_target_values=cur_target_values,
                _train_idx=st._train_idx,
                pre_pipeline=pre_pipeline,
                pre_pipeline_name=pre_pipeline_name,
                train_df_transformed=st.train_df_transformed,
                _schema_hash=_schema_hash,
                _input_schema=_input_schema,
                _build_feature_selection_report=_build_feature_selection_report,
                _selector_params_hash=_selector_params_hash,
                _unwrap_selector=_unwrap_selector,
            )

            if cached_dfs is None:
                st.pipeline_cache.set(cache_key, st.train_df_transformed, val_df_transformed, test_df_transformed)
                st.pipeline_cache.set_fitted_pipeline(cache_key, pre_pipeline)

            # After the first model trains, if the pre_pipeline is identity-equivalent (kept all
            # columns) AND the ordinary branch is in the suite, the remaining models would see
            # identical data - skip them. Checked AFTER _build_and_record_model_schema /
            # pipeline_cache.set above (not before, as this used to be ordered): the model that
            # JUST trained and triggered this dedup detection must still get its own schema
            # recorded and its transformed frames cached like every other model does -- breaking
            # before those calls silently left it with no metadata['model_schemas'] entry,
            # disabling predict-time schema-drift hard-fail protection for it specifically.
            if (
                _model_idx_in_run == 1
                and _pp_name_stripped
                and st.use_ordinary_models
                and st.feature_selection_config.skip_identity_equivalent_pre_pipelines
                and getattr(pre_pipeline, "_mlframe_identity_equivalent", False)
            ):
                _skip_remaining = _total_models_in_run - 1
                if _skip_remaining > 0:
                    logger.info(
                        "[Dedup] pre_pipeline '%s' is "
                        "identity-equivalent to ordinary (kept "
                        "all %d columns); skipping remaining "
                        "%d model(s) for this target.",
                        _pp_name_stripped,
                        st.train_df_transformed.shape[1] if st.train_df_transformed is not None else 0,
                        _skip_remaining,
                    )
                _break_model_loop = True
                break  # exit weight_schema loop

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
        _train_one_target_ste_step4_tier_transition_into(prev_tier, cur_tier, strategy, st, tier_dfs_cache, tier_enum_map_cache, ctx)
        prev_tier = cur_tier
    return pre_pipeline


def _train_one_target_ste_step1_model_name_weight(st, mlframe_model_name, weight_name, weight_values, polars_fastpath_active, prepared_train, prepared_val, prepared_test, tier_pandas, cur_target_name, _schema_hash, cache_key, _model_entry, _ngb_fallback_snapshot, pre_pipeline):
    """Step 1 of _train_one_target_ste_step1_model_entry_tqdmu: lines starting at ``model_name_with_weight = st.common_params["model_name"]``."""
    from mlframe.training.core._phase_train_one_target import (
        _cached_init_params,
        _forward_dataset_reuse_cache,
    )

    model_name_with_weight = st.common_params["model_name"]
    model_file_name = f"{mlframe_model_name}"
    if weight_name != "uniform":
        model_name_with_weight += f" w={weight_name}"
        model_file_name += f"_{weight_name}"

    # Isolation copy: per-(model, weight) inner mutations (sample_weight, plot_file
    # decoration, lazy pandas conversion, fastpath frame swap) must not bleed into
    # the outer ``common_params`` template that the next iteration consumes. The
    # 4-deep nesting (target_type x target x pre_pipeline x model x weight) has been
    # verified across the suite -- removing the copy regresses the cross-weight
    # contamination tests. Do NOT inline.
    st.current_common_params = st.common_params.copy()
    st.current_common_params["sample_weight"] = weight_values

    if polars_fastpath_active:
        st.current_common_params["train_df"] = prepared_train
        if prepared_val is not None:
            st.current_common_params["val_df"] = prepared_val
        if prepared_test is not None:
            st.current_common_params["test_df"] = prepared_test
    else:
        st.current_common_params["train_df"] = tier_pandas["train_df"]
        if tier_pandas.get("val_df") is not None:
            st.current_common_params["val_df"] = tier_pandas["val_df"]
        if tier_pandas.get("test_df") is not None:
            st.current_common_params["test_df"] = tier_pandas["test_df"]

    # Drop per-group aggregate columns from the MLP's view of X.
    # Pattern matches ``group_*_(mean|std|min|max)`` by default.
    # Only the MLP sees the trimmed feature set; tree models in
    # the suite get the original columns. Gated on the knob +
    # the extreme-AR + group-aware trigger predicate (computed
    # above for the model-level skip). Drop applies to train /
    # val / test consistently so the predict path doesn't see
    # extra columns the network wasn't trained on.
    if mlframe_model_name == "mlp" and bool(getattr(st.behavior_config, "mlp_drop_per_group_constants", False)):
        _drop_pattern = str(
            getattr(
                st.behavior_config,
                "mlp_drop_per_group_constants_pattern",
                r"^group_.*_(mean|std|min|max)$",
            )
        )
        _train_df_now = st.current_common_params.get("train_df")
        _cols_now = list(getattr(_train_df_now, "columns", []) or []) if _train_df_now is not None else []
        _per_group_cols = _identify_per_group_columns(_cols_now, _drop_pattern)
        if _per_group_cols:
            logger.info(
                "MLP per-group-aggregate column drop fired for target='%s': "
                "dropping %d columns matching %r (e.g. %s). Tree models "
                "still see them; only MLP gets the trimmed feature set.",
                cur_target_name,
                len(_per_group_cols),
                _drop_pattern,
                _per_group_cols[:3],
            )
            st.current_common_params["train_df"] = _drop_columns_for_mlp(
                st.current_common_params.get("train_df"),
                _per_group_cols,
            )
            if st.current_common_params.get("val_df") is not None:
                st.current_common_params["val_df"] = _drop_columns_for_mlp(
                    st.current_common_params.get("val_df"),
                    _per_group_cols,
                )
            if st.current_common_params.get("test_df") is not None:
                st.current_common_params["test_df"] = _drop_columns_for_mlp(
                    st.current_common_params.get("test_df"),
                    _per_group_cols,
                )
    if getattr(st.behavior_config, "model_file_hash_suffix", True):
        model_file_name += f"__sch_{_schema_hash}"

    if weight_name != "uniform" and st.current_common_params.get("plot_file"):
        st.current_common_params["plot_file"] = st.current_common_params["plot_file"] + weight_name + "_"

    cached_dfs = st.pipeline_cache.get(cache_key)
    # Cached frames came out of the pre_pipeline fitted by the model that populated the entry; this model never fits its
    # own, so it carries that fitted one (otherwise predict would skip the transform and feed its inner raw columns).
    if cached_dfs is not None and st.pipeline_cache.get_fitted_pipeline(cache_key) is not None:
        pre_pipeline = st.pipeline_cache.get_fitted_pipeline(cache_key)

    # INTENTIONAL: clone() lives INSIDE the weight loop. Each weight schema produces a
    # different trained model stored separately in models[type][target]; without per-iteration
    # cloning all in-memory entries would alias to the same last-trained sklearn object and
    # only the .dump snapshots would be correct. Do NOT move clone() outside the loop.
    original_model = st.models_params[_model_entry]["model"]
    cloned_model, _ngb_fallback_snapshot = _clone_model_with_sticky_flags(
        original_model=original_model,
        _cached_init_params=_cached_init_params,
        _ngb_fallback_snapshot=_ngb_fallback_snapshot,
        _forward_dataset_reuse_cache=_forward_dataset_reuse_cache,
        logger_obj=logger,
    )
    return cached_dfs, cloned_model, model_file_name, model_name_with_weight, original_model, pre_pipeline


def _train_one_target_ste_step2_supports_native_multi(target_type, _model_entry, cloned_model, mlframe_model_name, st, _cb_extra_fit_invariant, _neural_extra_fit_invariant, _mlp_extreme_ar_fired, model_name_with_weight, model_file_name, pre_pipeline, pre_pipeline_name, cur_target_name, ens_models, cached_dfs, strategy, polars_fastpath_active):
    """Step 2 of _train_one_target_ste_step1_model_entry_tqdmu: lines starting at ``if target_type.is_multi_target_regression:``."""
    if target_type.is_multi_target_regression:
        from mlframe.training.strategies import get_strategy

        _mtr_strategy = get_strategy(_model_entry)
        _mtr_obj_kwargs = _mtr_strategy.get_multi_target_objective_kwargs()
        if _mtr_obj_kwargs:
            try:
                cloned_model.set_params(**_mtr_obj_kwargs)
            except (ValueError, TypeError) as _mtr_set_err:
                # Some estimators (CatBoost on certain versions)
                # don't accept all params via set_params; fall
                # back to direct attribute assignment which
                # CatBoost does honour at fit-time.
                log_throttle(
                    logger, "train_one_target_mtr_set_params_failed", logging.WARNING,
                    "MTR set_params(%s) on %s failed (%s); " "falling back to setattr.",
                    _mtr_obj_kwargs,
                    mlframe_model_name,
                    _mtr_set_err,
                )
                for _k, _v in _mtr_obj_kwargs.items():
                    setattr(cloned_model, _k, _v)
        cloned_model = _mtr_strategy.wrap_multi_target(cloned_model)

    current_model_params = st.models_params[_model_entry].copy()
    current_model_params["model"] = cloned_model

    # CatBoost is the only Polars-native consumer that accepts cat_features / text_features / embedding_features at fit time; XGB and HGB
    # auto-detect via enable_categorical=True. Hoisted invariant ``_cb_extra_fit_invariant`` carries the filtered cat/text/embedding lists
    # (invariant across weights); stitch them into the per-weight fit_params here.
    if _cb_extra_fit_invariant and "fit_params" in current_model_params:
        if _cb_extra_fit_invariant:
            current_model_params["fit_params"] = {**current_model_params["fit_params"], **_cb_extra_fit_invariant}

    # Neural estimators self-encode embedding/text columns; thread the feature lists into their fit_params.
    if _neural_extra_fit_invariant and "fit_params" in current_model_params:
        current_model_params["fit_params"] = {**current_model_params["fit_params"], **_neural_extra_fit_invariant}

    # MLP extreme-AR + group-aware protections. Trigger predicate
    # ``_mlp_extreme_ar_fired`` is set above (per target, per
    # model). Both modifications land on the per-weight CLONED
    # model, so other weight schemas of this target see the same
    # overrides; the cross-target template is untouched because
    # we mutate ``current_model_params["model"]`` not
    # ``models_params``.
    if mlframe_model_name == "mlp" and _mlp_extreme_ar_fired:
        # Fix 1: bounded output activation (tanh -> hard cap).
        _apply_mlp_extreme_ar_output_activation(cloned_model)
        # Fix 3: L2 weight_decay bump by factor (default 100x).
        _wd_factor = float(
            getattr(
                st.behavior_config,
                "mlp_extreme_ar_weight_decay_factor",
                100.0,
            )
        )
        _wd_base = float(
            getattr(
                st.behavior_config,
                "mlp_extreme_ar_weight_decay_base",
                1e-4,
            )
        )
        _apply_mlp_extreme_ar_weight_decay_bump(
            cloned_model,
            factor=_wd_factor,
            base_weight_decay=_wd_base,
        )

    # Build process_model kwargs using helper
    process_model_kwargs = _build_process_model_kwargs(
        model_file=st.model_file,
        model_name_with_weight=model_name_with_weight,
        model_file_name=model_file_name,
        target_type=target_type,
        pre_pipeline=pre_pipeline,
        pre_pipeline_name=pre_pipeline_name,
        cur_target_name=cur_target_name,
        models=st.models,
        model_params=current_model_params,
        common_params=st.current_common_params,
        ens_models=ens_models,
        trainset_features_stats=st.trainset_features_stats,
        verbose=st.verbose,
        cached_dfs=cached_dfs,
        # Per-strategy decision on whether preprocessing for this strategy is already done.
        # Two sufficient conditions:
        #   (1) the suite-level polars-ds pipeline ran AND this strategy consumes polars natively;
        #   (2) the polars fastpath is active for this strategy (its frame is the polars native
        #       one, so sklearn encoder/scaler/imputer would be redundant and crash anyway).
        # Note: requires_encoding=True is NOT a re-run trigger (HGB declares it for pandas-fallback
        # only; on the polars fastpath HGB consumes pl.Categorical natively). Only non-Polars
        # strategies fall through to their own pre_pipeline run in trainer.py.
        polars_pipeline_applied=((st.polars_pipeline_applied and strategy.supports_polars) or polars_fastpath_active),
        mlframe_model_name=mlframe_model_name,
        metadata_columns=st.metadata.get("columns"),
    )

    _is_neural = is_neural_model(mlframe_model_name)
    _timeout = _compute_neural_max_time(st._non_neural_train_times) if _is_neural else None
    return _is_neural, _timeout, cloned_model, current_model_params, process_model_kwargs


def _train_one_target_ste_step3_timeout_none(_timeout, current_model_params, st, mlframe_model_name):
    """Step 3 of _train_one_target_ste_step1_model_entry_tqdmu: lines starting at ``if _timeout is not None:``."""
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
                if st.verbose:
                    logger.info(
                        "  [NeuralTimeout] %s max_time=%dh%02dm%02ds " "(P95 of %d prior non-neural train times: %.0fs)",
                        mlframe_model_name,
                        _max_time_dict["hours"],
                        _max_time_dict["minutes"],
                        _max_time_dict["seconds"],
                        _n,
                        _p95,
                    )


def _train_one_target_ste_step1_doubling_peak_ram(strategy, st, tier_dfs_cache, tier_enum_map_cache, ctx, mlframe_model_name):
    """Step 1 of _train_one_target_ste_step1_model_entry_tqdmu: lines starting at ``if not strategy.supports_polars and st.train_df_polars is not None:``."""
    from mlframe.training.core._phase_train_one_target import (
        _release_ctx_polars_frames,
    )

    if not strategy.supports_polars and st.train_df_polars is not None:
        # Drop locals AND ctx attributes -- ctx still pins the strong ref to the same frames
        # assigned via ctx.*_df_polars at function entry, so a bare ``del`` of the locals would
        # leave maybe_clean_ram_and_gpu with nothing to reclaim and turn the log line into a lie.
        del st.train_df_polars, st.val_df_polars, st.test_df_polars
        st.train_df_polars = st.val_df_polars = st.test_df_polars = None
        # Drop polars-tier entries only - pandas-tier entries hang on the SAME cache dicts
        # (these locals now reference suite-scoped dicts in _per_pp_cache) and must survive
        # the release. ``_invalidate_polars_feature_side_cache(ctx)`` runs further down the
        # _release_ctx_polars_frames path and does the same for the prepared_frames sub-
        # cache; the tier_dfs / tier_enum_map dicts that pre-date this hoist are scrubbed
        # here so a same-target pandas-tier sibling reads a clean enum-map slot.
        for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
            tier_dfs_cache.pop(_pl_only_key, None)
        tier_enum_map_cache.clear()  # All entries are polars-only (populated only on the polars fastpath).
        st.baseline_rss_mb = _release_ctx_polars_frames(
            ctx,
            st.baseline_rss_mb,
            st.df_size_mb,
            verbose=st.verbose,
            reason="non-polars-native strategy entry",
        )
        if st.verbose:
            logger.info(
                "  Released pre-pipeline Polars originals before %s (non-polars-native strategy).",
                mlframe_model_name,
            )


def _train_one_target_ste_step2_into_current_model(polars_fastpath_active, mlframe_model_name, _cat_features, prepared_train, st, _cb_extra_fit_invariant):
    """Step 2 of _train_one_target_ste_step1_model_entry_tqdmu: lines starting at ``if polars_fastpath_active and mlframe_model_name == "cb":``."""
    if polars_fastpath_active and mlframe_model_name == "cb":
        _cb_extra_fit_invariant = {}
        if _cat_features:
            _valid_cat_inv = _filter_polars_cat_features_by_dtype(prepared_train, _cat_features)
            if _valid_cat_inv:
                _cb_extra_fit_invariant["cat_features"] = _valid_cat_inv
        if st.text_features:
            _cb_text_inv = filter_existing(prepared_train, st.text_features)
            if _cb_text_inv:
                _cb_extra_fit_invariant["text_features"] = _cb_text_inv
        if st.embedding_features:
            _cb_emb_inv = filter_existing(prepared_train, st.embedding_features)
            if _cb_emb_inv:
                _cb_extra_fit_invariant["embedding_features"] = _cb_emb_inv
    return _cb_extra_fit_invariant


def _train_one_target_ste_step3_strategy_but_cannot(strategy, st, _neural_threads_cats, prepared_train, _neural_extra_fit_invariant):
    """Step 3 of _train_one_target_ste_step1_model_entry_tqdmu: lines starting at ``if getattr(strategy, "cache_key", None) in ("neural", "recurrent") and``."""
    if getattr(strategy, "cache_key", None) in ("neural", "recurrent") and (st.text_features or st.embedding_features or _neural_threads_cats):
        _neural_extra_fit_invariant = {}
        if st.text_features:
            _ntxt_inv = filter_existing(prepared_train, st.text_features)
            if _ntxt_inv:
                _neural_extra_fit_invariant["text_features"] = _ntxt_inv
        if st.embedding_features:
            _nemb_inv = filter_existing(prepared_train, st.embedding_features)
            if _nemb_inv:
                _neural_extra_fit_invariant["embedding_features"] = _nemb_inv
        if _neural_threads_cats and st.cat_features:
            _ncat_inv = filter_existing(prepared_train, st.cat_features)
            if _ncat_inv:
                _neural_extra_fit_invariant["cat_features"] = _ncat_inv
    return _neural_extra_fit_invariant


def _train_one_target_ste_step4_tier_transition_into(prev_tier, cur_tier, strategy, st, tier_dfs_cache, tier_enum_map_cache, ctx):
    """Step 4 of _train_one_target_ste_step1_model_entry_tqdmu: lines starting at ``if prev_tier is not None and cur_tier != prev_tier and not strategy.su``."""
    from mlframe.training.core._phase_train_one_target import (
        _release_ctx_polars_frames,
    )

    if prev_tier is not None and cur_tier != prev_tier and not strategy.supports_polars:
        if st.train_df_polars is not None:
            # Same rationale as the entry-site release: locals AND ctx attributes must both drop their refs.
            del st.train_df_polars, st.val_df_polars, st.test_df_polars
            st.train_df_polars = st.val_df_polars = st.test_df_polars = None
            # Selective drop: see same-shape comment at the non-polars-native entry site
            # above for rationale. These cache references are suite-scoped now, so a blanket
            # .clear() would also wipe pandas-tier entries that survived the polars release.
            for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
                tier_dfs_cache.pop(_pl_only_key, None)
            tier_enum_map_cache.clear()
            st.baseline_rss_mb = _release_ctx_polars_frames(ctx, st.baseline_rss_mb, st.df_size_mb, verbose=st.verbose, reason="tier transition")
            if st.verbose:
                logger.info("  Released pre-pipeline Polars originals (tier transition)")
