"""Single (model, weight-schema) training iteration carved out of ``_phase_train_one_target_body``.

One call = one pass of the innermost loop inside ``_train_one_target``: build per-weight params, clone
the model, wire multi-target-regression kwargs, apply MLP extreme-AR protections, call ``process_model``,
run the per-model post-train tail, and record the model schema. Carved out purely to keep the parent
module under the file-size ceiling -- the control flow (loop over weight_schemas, break/continue
handling) stays in the parent; this function returns a result dict the parent uses to update its loop
state and decide whether to break/continue.
"""

from __future__ import annotations

import logging
from timeit import default_timer as timer

from ..phases import phase
from ..models import is_neural_model
from ..train_eval import process_model
from ._misc_helpers import _compute_neural_max_time, _elapsed_str
from ._setup_helpers import _build_process_model_kwargs
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
from ._phase_train_one_target_post import _run_per_model_post_train_tail

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _run_one_weight_iteration(
    *,
    ctx,
    weight_name,
    weight_values,
    common_params,
    mlframe_model_name,
    polars_fastpath_active,
    prepared_train,
    prepared_val,
    prepared_test,
    tier_pandas,
    behavior_config,
    cur_target_name,
    cur_target_values,
    _schema_hash,
    _input_schema,
    _mlp_extreme_ar_fired,
    model_file,
    target_type,
    pre_pipeline,
    pre_pipeline_name,
    models,
    _model_entry,
    models_params,
    trainset_features_stats,
    verbose,
    pipeline_cache,
    cache_key,
    polars_pipeline_applied,
    strategy,
    metadata,
    _non_neural_train_times,
    test_df_pd,
    current_test_target,
    _train_idx,
    _cached_init_params,
    _forward_dataset_reuse_cache,
    _build_feature_selection_report,
    _selector_params_hash,
    _unwrap_selector,
    ens_models,
    _model_idx_in_run,
    _total_models_in_run,
    _pp_name_stripped,
    use_ordinary_models,
    feature_selection_config,
    _cb_extra_fit_invariant,
    _neural_extra_fit_invariant,
    _ngb_fallback_snapshot,
) -> dict:
    """Run one (model, weight-schema) iteration of the inner training loop.

    Returns a dict: ``trainset_features_stats``, ``pre_pipeline``, ``train_df_transformed``,
    ``current_common_params``, ``cache_key``, ``_ngb_fallback_snapshot`` (all threaded forward into the
    next iteration by the caller), plus ``skip`` (mirrors the original inline ``continue`` on a caught
    model failure) and ``break_model_loop`` (mirrors the original inline ``break`` on the identity-
    equivalent-pre_pipeline dedup shortcut).
    """
    model_name_with_weight = common_params["model_name"]
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
    if mlframe_model_name == "mlp" and bool(getattr(behavior_config, "mlp_drop_per_group_constants", False)):
        _drop_pattern = str(
            getattr(
                behavior_config,
                "mlp_drop_per_group_constants_pattern",
                r"^group_.*_(mean|std|min|max)$",
            )
        )
        _train_df_now = current_common_params.get("train_df")
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
            current_common_params["train_df"] = _drop_columns_for_mlp(
                current_common_params.get("train_df"),
                _per_group_cols,
            )
            if current_common_params.get("val_df") is not None:
                current_common_params["val_df"] = _drop_columns_for_mlp(
                    current_common_params.get("val_df"),
                    _per_group_cols,
                )
            if current_common_params.get("test_df") is not None:
                current_common_params["test_df"] = _drop_columns_for_mlp(
                    current_common_params.get("test_df"),
                    _per_group_cols,
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
    original_model = models_params[_model_entry]["model"]
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

    # F-34 (2026-05-31): MULTI_TARGET_REGRESSION build-time wiring.
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
                logger.warning(
                    "MTR set_params(%s) on %s failed (%s); " "falling back to setattr.",
                    _mtr_obj_kwargs,
                    mlframe_model_name,
                    _mtr_set_err,
                )
                for _k, _v in _mtr_obj_kwargs.items():
                    setattr(cloned_model, _k, _v)
        cloned_model = _mtr_strategy.wrap_multi_target(cloned_model)

    current_model_params = models_params[_model_entry].copy()
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
                behavior_config,
                "mlp_extreme_ar_weight_decay_factor",
                100.0,
            )
        )
        _wd_base = float(
            getattr(
                behavior_config,
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
        polars_pipeline_applied=((polars_pipeline_applied and strategy.supports_polars) or polars_fastpath_active),
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
                        "  [NeuralTimeout] %s max_time=%dh%02dm%02ds " "(P95 of %d prior non-neural train times: %.0fs)",
                        mlframe_model_name,
                        _max_time_dict["hours"],
                        _max_time_dict["minutes"],
                        _max_time_dict["seconds"],
                        _n,
                        _p95,
                    )

    t0_model = timer()
    try:
        with phase("process_model", model=mlframe_model_name, weight=weight_name):
            trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(**process_model_kwargs)
    except Exception as model_err:
        # Skip-and-continue is opt-in. KeyboardInterrupt is intentionally not caught here;
        # native SIGSEGV that kills the process won't be caught either.
        if not behavior_config.continue_on_model_failure:
            raise
        logger.exception(
            "  process_model(%s, w=%s) FAILED after %s -- %s: %s. continue_on_model_failure=True -> skipping and moving on.",
            mlframe_model_name,
            weight_name,
            _elapsed_str(t0_model),
            type(model_err).__name__,
            model_err,
        )
        metadata.setdefault("failed_models", []).append(
            {
                "model": mlframe_model_name,
                "weighting": weight_name,
                "error_type": type(model_err).__name__,
                "error_message": str(model_err),
            }
        )
        return {
            "trainset_features_stats": trainset_features_stats,
            "pre_pipeline": pre_pipeline,
            "train_df_transformed": None,
            "current_common_params": current_common_params,
            "cache_key": cache_key,
            "break_model_loop": False,
            "skip": True,
            "_ngb_fallback_snapshot": _ngb_fallback_snapshot,
        }
    if verbose:
        logger.info("  process_model(%s, w=%s) done -- %s", mlframe_model_name, weight_name, _elapsed_str(t0_model))
    if not _is_neural and t0_model is not None:
        _non_neural_train_times.append(timer() - t0_model)
    # Per-model post-train tail (TTA uncertainty eval + composite
    # y-scale emit + adaptive RAM reclaim) carved into
    # ``_run_per_model_post_train_tail``; it mutates ``metadata`` /
    # ``ctx.models`` in place so nothing is returned.
    _run_per_model_post_train_tail(
        behavior_config=behavior_config,
        test_df_transformed=test_df_transformed,
        current_test_target=current_test_target,
        ctx=ctx,
        target_type=target_type,
        cur_target_name=cur_target_name,
        mlframe_model_name=mlframe_model_name,
        metadata=metadata,
        test_df_pd=test_df_pd,
        _train_idx=_train_idx,
    )

    # After the first model trains, if the pre_pipeline is identity-equivalent (kept all
    # columns) AND the ordinary branch is in the suite, the remaining models would see
    # identical data - skip them.
    if (
        _model_idx_in_run == 1
        and _pp_name_stripped
        and use_ordinary_models
        and feature_selection_config.skip_identity_equivalent_pre_pipelines
        and getattr(pre_pipeline, "_mlframe_identity_equivalent", False)
    ):
        _skip_remaining = _total_models_in_run - 1
        if _skip_remaining > 0:
            logger.info(
                "[Dedup] pre_pipeline '%s' is " "identity-equivalent to ordinary (kept " "all %d columns); skipping remaining " "%d model(s) for this target.",
                _pp_name_stripped,
                train_df_transformed.shape[1] if train_df_transformed is not None else 0,
                _skip_remaining,
            )
        return {
            "trainset_features_stats": trainset_features_stats,
            "pre_pipeline": pre_pipeline,
            "train_df_transformed": train_df_transformed,
            "current_common_params": current_common_params,
            "cache_key": cache_key,
            "break_model_loop": True,
            "skip": False,
            "_ngb_fallback_snapshot": _ngb_fallback_snapshot,
        }

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

    return {
        "trainset_features_stats": trainset_features_stats,
        "pre_pipeline": pre_pipeline,
        "train_df_transformed": train_df_transformed,
        "current_common_params": current_common_params,
        "cache_key": cache_key,
        "break_model_loop": False,
        "skip": False,
        "_ngb_fallback_snapshot": _ngb_fallback_snapshot,
    }
