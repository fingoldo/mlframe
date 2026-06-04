"""Pure-helper carves of ``_main_train_suite.train_mlframe_models_suite``.

Each helper is a value-in / value-out transform whose body was lifted verbatim from the original ``train_mlframe_models_suite`` so behavioural equivalence is preserved by construction. The helpers live here so the parent function body shrinks toward the <700 LOC budget without introducing a state dataclass (``ctx`` already carries every cross-phase mutable for the parent's other phases).

Order follows the original control flow of ``train_mlframe_models_suite``.
"""
from __future__ import annotations

import copy as _copy_mod
import logging
from typing import Any, Optional

import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def apply_module_global_patches(module) -> None:
    """Apply the third-party patches the suite relies on (LGBMModel.feature_names_in_ setter; dataset-build logging) lazily at suite entry.

    Previously these ran as import-time side effects of ``mlframe.training``; now bare imports leave joblib / lightgbm / catboost / xgboost untouched. The two functions are looked up on ``module`` (the parent of this helper, ``_main_train_suite``) so test code can monkeypatch them in-place to observe call ordering.

    Mutates ``module.apply_loky_cpu_count_override`` / ``module.apply_third_party_patches_once`` on first call to bind the real implementations.
    """
    if module.apply_loky_cpu_count_override is None:
        from .. import apply_loky_cpu_count_override as _apply_loky
        module.apply_loky_cpu_count_override = _apply_loky
    if module.apply_third_party_patches_once is None:
        from .._model_factories import apply_third_party_patches_once as _apply_patches
        module.apply_third_party_patches_once = _apply_patches
    module.apply_loky_cpu_count_override()
    module.apply_third_party_patches_once()


def unpack_ctx_configs_to_locals(ctx: Any) -> dict:
    """Mirror the post-``setup_configuration`` ctx attributes the original function body bound as locals.

    Returns a dict so the caller can do ``locals().update(...)``-style splatter (sadly Python forbids that for function-locals); the typical caller pattern is ``cfgs = unpack_ctx_configs_to_locals(ctx); preprocessing_config = cfgs['preprocessing_config']; ...``.

    Why keep this even though ``ctx.X`` reads are short: every downstream callsite in ``train_mlframe_models_suite`` passes the local name (not the ``ctx.X``) to its phase function, and the original control flow had this 25-attr block inline. Lifting it preserves the explicit-locals semantics + shortens the parent.
    """
    return {
        "preprocessing_config": ctx.preprocessing_config,
        "pipeline_config": ctx.pipeline_config,
        "feature_types_config": ctx.feature_types_config,
        "split_config": ctx.split_config,
        "hyperparams_config": ctx.hyperparams_config,
        "behavior_config": ctx.behavior_config,
        "reporting_config": ctx.reporting_config,
        "output_config": ctx.output_config,
        "outlier_detection_config": ctx.outlier_detection_config,
        "feature_selection_config": ctx.feature_selection_config,
        "baseline_diagnostics_config": ctx.baseline_diagnostics_config,
        "dummy_baselines_config": ctx.dummy_baselines_config,
        "quantile_regression_config": ctx.quantile_regression_config,
        "composite_target_discovery_config": ctx.composite_target_discovery_config,
        "data_dir": ctx.data_dir,
        "models_dir": ctx.models_dir,
        "save_charts": ctx.save_charts,
        "outlier_detector": ctx.outlier_detector,
        "od_val_set": ctx.od_val_set,
        "use_mrmr_fs": ctx.use_mrmr_fs,
        "mrmr_kwargs": ctx.mrmr_kwargs,
        "rfecv_models": ctx.rfecv_models,
        "custom_pre_pipelines": ctx.custom_pre_pipelines,
        "common_params_dict": ctx.common_params_dict,
        "mlframe_models": ctx.mlframe_models,
        "metadata": ctx.metadata,
    }


def warn_on_empty_target_by_type(target_by_type: Any) -> None:
    """Empty target_by_type means the extractor returned no targets - usually a caller-side mis-configuration.

    Common cause is passing classification_exact_values / classification_thresholds without classification_targets=[...] (the default ``SimpleFeaturesAndTargetsExtractor.build_targets`` gates those branches on classification_targets being truthy). Pre-fix this short-circuited silently to ``(empty_models, metadata)`` making such misconfigurations look like a fast successful run; loud WARN surfaces them at suite entry instead.
    """
    if not target_by_type:
        logger.warning(
            "train_mlframe_models_suite: features_and_targets_extractor produced an "
            "empty target_by_type. No models will be trained. Check the extractor's "
            "configuration - common cause is passing classification_exact_values / "
            "classification_thresholds without classification_targets=[...] (the "
            "default SimpleFeaturesAndTargetsExtractor.build_targets gates those "
            "branches on classification_targets being truthy)."
        )


def validate_suite_inputs(
    df: Any,
    target_name: Any,
    model_name: Any,
    features_and_targets_extractor: Any,
) -> Any:
    """Validate the four required positional kwargs of ``train_mlframe_models_suite``.

    Wave 29 P2 fix (2026-05-20): pre-fix rejected ``pathlib.Path`` with a confusing "must be ... path string" message. Path is a natural caller idiom (yaml config + Path / Click + Path); coerce to str at the boundary so the downstream parquet-read path stays unchanged.

    Returns the (possibly-coerced) ``df`` so callers can rebind their local in one statement.
    """
    import os as _os_for_pathlike

    if isinstance(df, _os_for_pathlike.PathLike):
        df = str(df)
    if not isinstance(df, (pd.DataFrame, pl.DataFrame, str)):
        raise TypeError(f"df must be pandas DataFrame, polars DataFrame, or path string / PathLike, " f"got {type(df).__name__}")
    if isinstance(df, str) and not df.lower().endswith(".parquet"):
        raise ValueError(f"File path must be a .parquet file, got: {df}")

    if target_name is None or not isinstance(target_name, str):
        raise TypeError(f"target_name must be a non-empty string, got {type(target_name).__name__}")
    if not target_name.strip():
        raise ValueError("target_name cannot be empty or whitespace-only")
    if model_name is None or not isinstance(model_name, str):
        raise TypeError(f"model_name must be a non-empty string, got {type(model_name).__name__}")
    if not model_name.strip():
        raise ValueError("model_name cannot be empty or whitespace-only")
    if features_and_targets_extractor is None:
        raise ValueError("features_and_targets_extractor is required")
    return df


def check_precomputed_fingerprint(
    precomputed: Any,
    train_df: Any,
) -> bool:
    """PRECOMP-NO-FP-CHECK: when the caller stamped ``train_df_fingerprint`` on the bundle, verify it matches the live train frame.

    A mismatch (caller passed a bundle from a different run) is a silent label-leak vector -- we WARN-and-recompute rather than trust the precomputed stats.
    """
    _precomp_fp_ok = True
    if (
        precomputed is not None
        and precomputed.train_df_fingerprint
        and train_df is not None
    ):
        try:
            from ..feature_handling.fingerprint import fingerprint_df as _fp
            _live_fp = _fp(train_df).short()
            _bundle_fp = str(precomputed.train_df_fingerprint)
            if _bundle_fp not in (_live_fp, _live_fp[:len(_bundle_fp)]):
                logger.warning(
                    "precomputed.train_df_fingerprint (%s) does not match live train_df fingerprint (%s); "
                    "ignoring the precomputed bundle and recomputing inline.",
                    _bundle_fp, _live_fp,
                )
                _precomp_fp_ok = False
        except Exception as _fp_err:
            logger.debug("precompute fingerprint cross-check skipped (%s)", _fp_err)
    return _precomp_fp_ok


def compute_or_fetch_trainset_features_stats(
    _precomp_fp_ok: bool,
    precomputed: Any,
    train_df: Any,
    train_df_polars_pre: Any,
    verbose: int,
) -> Any:
    """Compute (or fetch from precomputed bundle) ``trainset_features_stats``.

    Routing:
    - precomputed (truthy gate) -> use it directly
    - train_df is pl.DataFrame -> polars backend
    - train_df_polars_pre is pl.DataFrame (live train_df has been pandas-converted upstream) -> polars backend using the pre-conversion original
    - else -> pandas backend
    """
    from ..helpers import (
        get_trainset_features_stats,
        get_trainset_features_stats_polars,
    )
    from ..phases import phase

    # Truthy gate (not "is not None"): empty dicts/Series must NOT silently disable the inline compute -- the
    # ``precompute_*`` stubs historically returned ``{}`` and at-call callers occasionally pass through partial
    # bundles from disk.
    if _precomp_fp_ok and precomputed is not None and precomputed.trainset_features_stats:
        if verbose:
            logger.info("Using caller-supplied trainset_features_stats (precomputed bundle); skipping inline compute.")
        return precomputed.trainset_features_stats
    if isinstance(train_df, pl.DataFrame):
        if verbose:
            logger.info("Computing trainset_features_stats on Polars...")
        with phase("trainset_features_stats", backend="polars"):
            return get_trainset_features_stats_polars(train_df)
    if isinstance(train_df_polars_pre, pl.DataFrame):
        # Polars fastpath fallback: when the live ``train_df`` has been pandas-converted upstream (a non-native preprocessor forced the conversion) but
        # the pre-conversion polars original is still pinned, prefer the polars backend. The pandas ``get_trainset_features_stats`` iterates cat columns
        # in pure Python (one ``.unique()`` collect per col) while the polars backend batches them via a single ``.collect()`` with ``implode()``.
        if verbose:
            logger.info("Computing trainset_features_stats on Polars (using train_df_polars_pre; live train_df is pandas)...")
        with phase("trainset_features_stats", backend="polars"):
            return get_trainset_features_stats_polars(train_df_polars_pre)
    if verbose:
        logger.info("Computing trainset_features_stats on pandas...")
    with phase("trainset_features_stats", backend="pandas"):
        return get_trainset_features_stats(train_df)


def maybe_apply_composite_target_specs_precomputed(
    _precomp_fp_ok: bool,
    precomputed: Any,
    metadata: dict,
    verbose: int,
) -> bool:
    """Opt-in fast path: caller-supplied composite_target_specs bypasses the discovery phase entirely.

    Pre-seed metadata so downstream readers (per-target dummy-baseline inversion, predict-time composite inverse transforms) find the specs in their usual location. target_by_type is left unchanged because the caller-supplied specs imply the augmented targets already live in it.

    Returns True when the precomputed branch fired (caller should skip discovery); False otherwise.

    Truthy gate (not "is not None"): an empty composite spec dict carries zero discovered targets, which is indistinguishable from "discovery skipped"; if we let it through, the suite would silently lose every composite target. The stub helpers used to return {} -- truthy gate is the defensive fix.
    """
    if not (_precomp_fp_ok and precomputed is not None and precomputed.composite_target_specs):
        return False
    if verbose:
        logger.info("Using caller-supplied composite_target_specs (precomputed bundle); skipping discovery.")
    # Deep-copy so a downstream phase that ever appends to metadata['composite_target_specs']
    # doesn't mutate the caller's precomputed bundle. Pre-fix this was a shared reference;
    # a notebook calling the suite once per fold while reusing the same precomputed bundle
    # would see call N's late-arrived spec stamped INTO the bundle and resurface in call N+1.
    metadata["composite_target_specs"] = _copy_mod.deepcopy(precomputed.composite_target_specs)
    return True


def maybe_apply_dummy_baselines_precomputed(
    _precomp_fp_ok: bool,
    precomputed: Any,
    metadata: dict,
    dummy_baselines_config: Any,
    ctx: Any,
    verbose: int,
) -> Any:
    """Opt-in fast path: caller-supplied dummy_baselines bypasses the per-target dummy compute.

    Pre-seed metadata so downstream summary / verdict consumers find the payload in its usual location, then shallow-copy dummy_baselines_config with enabled=False so the per-target ``run_dummy_baselines`` short-circuits (its first guard checks ``config.enabled``).

    Truthy gate (not "is not None"): empty dummy_baselines would silently disable every per-target compute -- callers must supply real values; "I passed it but it's empty" must fall through to inline compute.
    """
    if not (_precomp_fp_ok and precomputed is not None and precomputed.dummy_baselines):
        return dummy_baselines_config
    if verbose:
        logger.info("Using caller-supplied dummy_baselines (precomputed bundle); skipping per-target compute.")
    # Deep-copy: same cross-suite alias hazard as composite_target_specs above. Caller's
    # bundle stays immutable across multiple suite calls reusing the same precomputed object.
    metadata["dummy_baselines"] = _copy_mod.deepcopy(precomputed.dummy_baselines)
    try:
        dummy_baselines_config = dummy_baselines_config.model_copy(update={"enabled": False})
    except AttributeError:
        # Defensive: when the config slot is plain dict / SimpleNamespace fall back to attribute set.
        # Narrow except: only the failures that mean "not a pydantic / not attr-settable" object.
        # A pydantic v2 frozen model raises ValidationError on attribute set; treat the same.
        try:
            dummy_baselines_config.enabled = False
        except (AttributeError, TypeError) as _set_err:
            # Pre-fix this swallowed bare Exception (including ValidationError on frozen pydantic
            # models). enabled stayed True, run_dummy_baselines would re-compute from scratch and
            # OVERWRITE metadata["dummy_baselines"] -- silently discarding the caller-supplied
            # precomputed payload that the user supplied via the precomputed bundle. Raise so
            # the caller knows the precompute fast-path isn't usable for their config type
            # rather than silently re-running and corrupting the precomputed result.
            raise RuntimeError(
                f"dummy_baselines_config of type {type(dummy_baselines_config).__name__} "
                f"is neither pydantic-copyable (no .model_copy) nor attr-settable "
                f"(.enabled = False raised {type(_set_err).__name__}: {_set_err}). "
                f"The precompute fast-path can't disable downstream re-compute, which "
                f"would silently overwrite your precomputed dummy_baselines. Pass a "
                f"pydantic DummyBaselinesConfig or a writable dataclass / SimpleNamespace."
            ) from _set_err
    ctx.dummy_baselines_config = dummy_baselines_config
    return dummy_baselines_config


def apply_polars_cat_fixes_and_back_write_ctx(
    pr_module: Any,
    ctx: Any,
    train_df_polars: Any,
    val_df_polars: Any,
    test_df_polars: Any,
    train_df_pd: Any,
    val_df_pd: Any,
    test_df_pd: Any,
    filtered_train_df: Any,
    filtered_val_df: Any,
    cat_features: Any,
    behavior_config: Any,
    defer_pandas_conv: Any,
    was_polars_input: Any,
    metadata: dict,
    verbose: int,
    _bulk_setattr_to_ctx: Any,
) -> tuple:
    """Run polars cat fixes + persist Enum domains + back-write filled frames into ctx.

    Persist the train+val Enum domain into metadata so predict-time XGB-polars cat-cast lands on pl.Enum (no global string cache widening) and OOV test-only categories cast to null via strict=False -- matches training's "truly unseen test" treatment and prevents silent stale-category accumulation across inference calls.

    Write the filled frames BACK to ctx. ``_train_one_target`` later does ``train_df_polars = ctx.train_df_polars``; without this back-write it would read the pre-fix frames with nulls still in cat columns, causing CB Arrow Pool to crash with 'Data with nulls is not supported for categorical columns'. Covered by ``test_sensor_polars_utf8_nullable_cat_fills_before_cb`` + ``test_sensor_enum_null_fill_reaches_lazy_pandas_conversion``.
    """
    _polars_fixes_result = pr_module.apply_polars_categorical_fixes(
        train_df_polars=train_df_polars,
        val_df_polars=val_df_polars,
        test_df_polars=test_df_polars,
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        test_df_pd=test_df_pd,
        filtered_train_df=filtered_train_df,
        filtered_val_df=filtered_val_df,
        cat_features=cat_features,
        align_polars_categorical_dicts=behavior_config.align_polars_categorical_dicts,
        defer_pandas_conv=defer_pandas_conv,
        was_polars_input=was_polars_input,
        verbose=bool(verbose),
    )
    (
        train_df_polars, val_df_polars, test_df_polars,
        train_df_pd, val_df_pd, test_df_pd,
        filtered_train_df, filtered_val_df,
    ) = _polars_fixes_result[:8]
    _enum_domains_export = getattr(_polars_fixes_result, "enum_domains", None) or {}
    if _enum_domains_export:
        metadata.setdefault("enum_domains", {}).update(_enum_domains_export)
    _bulk_setattr_to_ctx(ctx, (
        "train_df_polars", "val_df_polars", "test_df_polars",
        "train_df_pd", "val_df_pd", "test_df_pd",
        "filtered_train_df", "filtered_val_df",
    ), locals())
    return (
        train_df_polars, val_df_polars, test_df_polars,
        train_df_pd, val_df_pd, test_df_pd,
        filtered_train_df, filtered_val_df,
    )


def run_recurrent_finalize_and_composite_post(
    ctx: Any,
    pr_module: Any,
    recurrent_config: Any,
    train_sequences: Any,
    val_sequences: Any,
    test_sequences: Any,
    train_df: Any,
    train_df_pd: Any,
    val_df_pd: Any,
    test_df_pd: Any,
    target_by_type: Any,
    train_idx: Any,
    val_idx: Any,
    test_idx: Any,
    _non_neural_train_times: Any,
    model_name: str,
    target_name: str,
    composite_target_discovery_config: Any,
    filtered_train_df: Any,
    filtered_val_df: Any,
    filtered_train_idx: Any,
    filtered_val_idx: Any,
    dummy_baselines_config: Any,
    reporting_config: Any,
    verbose: int,
) -> Any:
    """Run the post-target-loop tail: train_recurrent_models + finalize_suite + run_composite_post_processing.

    CODE-P1-12: read recurrent_models from ctx (not the closed-over function param) so any mid-flow mutation of ctx.recurrent_models propagates correctly to train_recurrent_models. ctx is threaded through so train_recurrent_models can rerun score_ensemble with the recurrent member entries appended (otherwise the recurrent models silently bypass the ensemble that already ran for the same target during _train_one_target). test_df_pd added for the same reason - the helper needs all three splits to compute per-member preds for the rerun.

    Returns the (models, metadata) tuple that the caller stamps as ``return dict(models), metadata``.
    """
    from ..utils import log_phase, log_ram_usage

    models = pr_module.train_recurrent_models(
        models=ctx.models,
        recurrent_models=ctx.recurrent_models,
        recurrent_config=recurrent_config,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        train_df=train_df,
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        test_df_pd=test_df_pd,
        target_by_type=target_by_type,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        _non_neural_train_times=_non_neural_train_times,
        model_name=model_name,
        verbose=bool(verbose),
        ctx=ctx,
    )
    ctx.models = models

    if verbose:
        log_phase(f"Training suite completed for {model_name}, {sum(len(v) for targets in models.values() for v in targets.values())} models.")
        log_ram_usage()

    metadata = pr_module.finalize_suite(ctx)

    models, metadata = pr_module.run_composite_post_processing(
        models=models,
        metadata=metadata,
        target_by_type=target_by_type,
        composite_target_discovery_config=composite_target_discovery_config,
        target_name=target_name,
        model_name=model_name,
        filtered_train_df=filtered_train_df,
        filtered_val_df=filtered_val_df,
        test_df_pd=test_df_pd,
        filtered_train_idx=filtered_train_idx,
        filtered_val_idx=filtered_val_idx,
        test_idx=test_idx,
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        train_idx=train_idx,
        val_idx=val_idx,
        dummy_baselines_config=dummy_baselines_config,
        reporting_config=reporting_config,
        plot_file=None,
        verbose=bool(verbose),
    )
    return models, metadata


def export_votenrank_leaderboards(
    ctx: Any,
    data_dir: Optional[str],
    verbose: int,
) -> None:
    """VOTENRANK-DISCONNECT (post-loop): collect any per-target ``_leaderboard`` payloads that ``score_ensemble`` parked on the ensemble result dicts and surface them under a single ``metadata["votenrank_leaderboard"]`` key.

    CSV export lands under ``output_config.data_dir/.leaderboard.csv`` when data_dir is set. The wire is read-only on ``ctx.ensembles`` so it can't reach back into the per-target loop and is safe to skip when F2 has not emitted any leaderboard yet (forward-compat with older score_ensemble builds).
    """
    try:
        _leaderboards = {}
        for _tt, _by_name in (ctx.ensembles or {}).items():
            for _tname, _ens_dict in (_by_name or {}).items():
                if not isinstance(_ens_dict, dict):
                    continue
                _lb = _ens_dict.get("_leaderboard")
                if _lb is None:
                    continue
                _leaderboards.setdefault(str(_tt), {})[_tname] = _lb
        if _leaderboards:
            ctx.metadata["votenrank_leaderboard"] = _leaderboards
            if data_dir:
                from pathlib import Path as _P
                _csv_path = _P(data_dir) / ".leaderboard.csv"
                try:
                    # Concatenate per-(type, target) frames with two index columns so a reader can
                    # filter back to one slice. Honour pl.DataFrame vs pd.DataFrame via .write_csv /
                    # .to_csv duck typing; mixed cases concat after a unified to_pandas hop.
                    import pandas as _pd
                    _frames = []
                    for _tt_s, _by_name in _leaderboards.items():
                        for _tname, _lb in _by_name.items():
                            if isinstance(_lb, pl.DataFrame):
                                # Route through Arrow split-blocks bridge (~32x vs default to_pandas, preserves Enum/Categorical/datetime dtypes); the prior pl->CSV->pandas re-read densified all dtypes to string.
                                from ..utils import get_pandas_view_of_polars_df as _get_pandas_view
                                _frame = _get_pandas_view(_lb)
                            elif isinstance(_lb, _pd.DataFrame):
                                _frame = _lb
                            else:
                                _frame = _pd.DataFrame(_lb)
                            _frame.insert(0, "target_type", _tt_s)
                            _frame.insert(1, "target_name", _tname)
                            _frames.append(_frame)
                    if _frames:
                        _all_lb = _pd.concat(_frames, ignore_index=True)
                        _all_lb.to_csv(_csv_path, index=False)
                        if verbose:
                            logger.info("votenrank leaderboard exported: %s (%d rows)", _csv_path, len(_all_lb))
                except Exception as _csv_err:
                    logger.warning("votenrank leaderboard CSV export failed: %s", _csv_err)
    except Exception as _vn_err:
        logger.warning("votenrank leaderboard wiring failed: %s", _vn_err)


def maybe_autoroute_autodetected_ltr(
    ctx: Any,
    target_type: Any,
    target_by_type: Any,
    raw_df: Any,
    features_and_targets_extractor: Any,
) -> Any:
    """Auto-detected-LTR safety net: the param-based early dispatch only fires for an EXPLICIT target_type=LEARNING_TO_RANK arg.

    When the caller leaves target_type=None, build_targets can still classify a target as LEARNING_TO_RANK; that target would otherwise reach the standard per-target loop and build a tree CLASSIFIER with a multiclass objective + an LTR eval metric -> LightGBMError ("Multiclass objective and metrics don't match"). target_by_type is resolved by call time, so route LTR to the ranker suite using the raw pre-preprocessing df (it re-transforms internally).

    Returns the ranker-suite result when the auto-route fires, else None (caller falls through to the standard loop).
    """
    from ..configs import TargetTypes
    from .utils import _maybe_dispatch_to_ltr_ranker_suite

    if target_type is None and TargetTypes.LEARNING_TO_RANK in target_by_type:
        return _maybe_dispatch_to_ltr_ranker_suite(
            ctx,
            target_type=TargetTypes.LEARNING_TO_RANK,
            df=raw_df,
            features_and_targets_extractor=features_and_targets_extractor,
        )
    return None
