"""Randomized fuzz coverage for ``train_mlframe_models_suite``.

Feeds ~150 unique, pairwise-covering combos through the suite and records
every combo's outcome to ``_fuzz_results.jsonl`` for later analysis.

Known-bug xfail rules live in ``_fuzz_combo.KNOWN_XFAIL_RULES`` and are
applied automatically per combo via ``pytest.mark.xfail`` in the test
function — new bugs discovered by fuzzing should be added there once
they're traced to a specific combo predicate.
"""
from __future__ import annotations

import os
import time
import traceback

import pytest

# Fuzz combos run ~150 train_mlframe_models_suite iterations and are deselected
# from the default test run; pass pytest --run-fuzz to include.
pytestmark = pytest.mark.fuzz

from tests.training._fuzz_combo import (
    FuzzCombo,
    build_frame_for_combo,
    enumerate_combos,
    log_combo_outcome,
    xfail_reason,
    build_mrmr_kwargs,
)
from tests.training.shared import SimpleFeaturesAndTargetsExtractor

# 2026-04-27: train_mlframe_models_suite signature collapsed several
# top-level kwargs (outlier_detector / data_dir / use_mrmr_fs / ...) into
# typed configs (see CHANGELOG ``2026-04-27 — Calibration reporting
# upgrades + suite-config sweep``). The fuzz suite uses these new
# configs directly at the suite call site; module-level imports here so
# the local imports inside ``_configs_for_combo`` aren't load-bearing.
from mlframe.training import (
    OutputConfig,
    OutlierDetectionConfig,
    FeatureSelectionConfig,
    ReportingConfig,
    ConfidenceAnalysisConfig,
)

# Enumerate once at import time — small, pure Python, no heavy deps.
# FUZZ_SEED env var overrides the default (driver scripts use this to
# sweep many seeds in sequence without editing the file; each pytest
# invocation reads the env fresh so 10k-combo campaigns can span 60+
# seeds × 150 combos each without the parent process sharing state).
_FUZZ_MASTER_SEED = int(os.environ.get("FUZZ_SEED", "20260422"))
COMBOS: list[FuzzCombo] = enumerate_combos(target=150, master_seed=_FUZZ_MASTER_SEED)


# Non-test helpers carved into a sibling module so this stays a lean
# pytest-discoverable test file (CLAUDE.md: 'Monolith split').
from tests.training._fuzz_suite_helpers import (
    _safe_cfg_kwargs,
    _config_for_models,
    _configs_for_combo,
    _recurrent_sequences_for_combo,
    _recurrent_config_for_combo,
    _outlier_detector_for_combo,
    _custom_pre_pipelines_for_combo,
    _boruta_shap_kwargs_for_combo,
    _maybe_to_parquet,
    _preprocessing_for_combo,
    _skip_if_deps_missing,
    _assert_prediction_invariants,
    _assert_serialization_roundtrip,
)


@pytest.fixture(autouse=True)
def _fuzz_combo_cleanup():
    """Between fuzz combos: close matplotlib figures, clear CB/XGB/LGB
    internal caches, drop generated models — state accumulation across the
    150-combo run has been observed to trigger native-level crashes
    (SIGSEGV on combo 6 in a sequential run on 2026-04-22)."""
    yield
    # 1. Matplotlib figures (mlframe emits per-model feature_importance plots).
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass
    # 2. mlframe's in-process caches (CB val Pool cache, tier-DF cache).
    try:
        from mlframe.training import trainer as _tr
        for attr in ("_CB_POOL_CACHE", "_CB_VAL_POOL_CACHE"):
            cache = getattr(_tr, attr, None)
            if hasattr(cache, "clear"):
                cache.clear()
    except Exception:
        pass
    # 3. CatBoost internal state — force full GPU/CPU resource release.
    try:
        import catboost
        # catboost.utils doesn't expose a global cleanup; deleting module-level
        # state is unsafe. Best-effort: trigger a GC pass twice so CB's
        # C++-side memory pools see zero Python refs before the next combo
        # allocates.
    except ImportError:
        pass
    # 4. Double GC — first pass collects Python objects, second pass lets
    # finalizers (including native lib close-outs) run before we return.
    import gc
    gc.collect()
    gc.collect()
    # 5. clean_ram: on Linux returns memory to OS via malloc_trim(0);
    # on Windows trims working-set via SetProcessWorkingSetSizeEx (RSS
    # only, not commit). Wired here as best-effort against multi-combo
    # native heap fragmentation that historically OOMs around combo #36
    # of 150 on Win32 multi-classification × ensembles paths.
    try:
        from pyutilz.system import clean_ram
        clean_ram()
    except Exception:
        pass


@pytest.mark.slow
@pytest.mark.slow_only
@pytest.mark.timeout(300)
@pytest.mark.parametrize("combo", COMBOS, ids=[c.pytest_id() for c in COMBOS])
def test_fuzz_train_mlframe_models_suite(combo: FuzzCombo, tmp_path, request):
    """Run ``train_mlframe_models_suite`` on one random combo; log the outcome.

    FUZZ-1 (2026-05-23) -- when ``MLFRAME_FUZZ_PERF_MODE`` env var is set
    (any truthy value: 1/yes/true/on), each combo is downgraded to a tiny
    config-coverage run: n_rows=1000, iterations=1, MRMR/Boruta/ensembles
    /baseline_diagnostics/dummy_baselines all disabled. Goal: verify suite
    wiring on every combo in seconds instead of minutes. Quality / metric
    assertions are NOT meaningful in this mode -- it's a smoke test only.

    ``MLFRAME_FUZZ_FORCE_N_ROWS`` (bug-hunt mode) -- when set to a positive
    int, overrides ONLY ``n_rows`` via ``dataclasses.replace``. ``FuzzCombo``
    is frozen with no eager canonicalisation at construction -- n_rows-gated
    rules (rare-imbalance clamp, ocsvm gate, RFECV/recurrent-model gates,
    viz-rendering tier) are all COMPUTED LIVE off ``self.n_rows`` inside
    ``canonical_key()`` / property methods, so they stay consistent for the
    new size automatically; nothing needs to be "re-run". Unlike perf-mode,
    every subsystem (MRMR, BorutaShap, ensembles, diagnostics) stays ON:
    this is for exhaustive bug-hunting at a fixed fast row count, not a
    wiring smoke test. Applied AFTER perf-mode (so perf-mode's n_rows wins
    if both are set) and BEFORE ``xfail_reason`` so n_rows-gated xfail rules
    see the forced size.

    Default (env unset): full combo runs unchanged.
    """
    import os as _os
    if _os.environ.get("MLFRAME_FUZZ_PERF_MODE", "").lower() in ("1", "yes", "true", "on"):
        from tests.training._fuzz_combo import apply_perf_mode
        combo = apply_perf_mode(combo)
    _forced_n_rows = _os.environ.get("MLFRAME_FUZZ_FORCE_N_ROWS", "")
    if _forced_n_rows.isdigit() and int(_forced_n_rows) > 0:
        import dataclasses as _dataclasses
        combo = _dataclasses.replace(combo, n_rows=int(_forced_n_rows))
    _skip_if_deps_missing(combo.models)

    # Apply xfail automatically for known bugs. pytest's runtime-xfail marker
    # works via ``request.node.add_marker``.
    reason = xfail_reason(combo)
    if reason is not None:
        # strict=True so an XPASS (combo now passes because the underlying fix landed) is a
        # visible regression -- the developer must remove the rule from KNOWN_XFAIL_RULES.
        # Pre-fix this was strict=False, which silently greened combos whether they passed or
        # failed and lost track of fix landings.
        request.node.add_marker(pytest.mark.xfail(reason=reason, strict=True))

    df, target_col, _cat_names = build_frame_for_combo(combo)

    # #16 invariant: capture caller-frame schema + shape before the
    # suite runs; re-assert identity after. Applies when input stays
    # in-memory (parquet-path combos have no Python-level caller frame
    # to preserve — the parquet file is the source of truth).
    frame_schema_before = None
    frame_shape_before = None
    frame_cols_before = None
    if combo.input_storage == "memory":
        if hasattr(df, "schema"):
            frame_schema_before = dict(df.schema)
        elif hasattr(df, "dtypes"):
            frame_schema_before = {c: str(df[c].dtype) for c in df.columns}
        frame_shape_before = getattr(df, "shape", None)
        frame_cols_before = tuple(df.columns) if hasattr(df, "columns") else None

    # Resolve target_type for FTE — maps combo's string target_type to
    # the TargetTypes enum. Multilabel + multi_target_regression get
    # explicit TargetTypes to trigger the 2-D target unpack path in FTE.
    #
    # multi_target_regression disambiguation: build_frame_for_combo emits
    # a 2-D (N, K) list column (target_col == "target") only when every
    # model in the combo natively handles a 2-D continuous target; for any
    # non-native combo it downgrades the frame to a 1-D "target_reg" column
    # (see _NATIVE_MTR_MODELS gate). The emitted target_col is therefore the
    # authoritative signal for which target_type the FTE should see -- a
    # downgraded MTR combo is REGRESSION at the data level.
    from mlframe.training.configs import TargetTypes as _TT
    _effective_target_type = combo.target_type
    if combo.target_type == "multi_target_regression" and target_col != "target":
        _effective_target_type = "regression"
    _combo_tt = {
        "regression": _TT.REGRESSION,
        "binary_classification": _TT.BINARY_CLASSIFICATION,
        "multiclass_classification": _TT.MULTICLASS_CLASSIFICATION,
        "multilabel_classification": _TT.MULTILABEL_CLASSIFICATION,
        "learning_to_rank": _TT.LEARNING_TO_RANK,
        "multi_target_regression": _TT.MULTI_TARGET_REGRESSION,
    }[_effective_target_type]
    # LTR combos: build_frame_for_combo adds a 'qid' column for queries;
    # surface it as the FTE's group_field so the ranker suite picks it up.
    _is_ltr = combo.target_type == "learning_to_rank"
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
        target_type=_combo_tt,
        # 2026-04-27 Session 7 batch 6: when the combo injects a
        # datetime column ('ts' from build_frame_for_combo), surface it
        # as ts_field so train_mlframe_models_suite's temporal_audit
        # auto-detect kicks in. Without this the audit stays silent
        # for fuzz combos and the auto-detect path is untested.
        ts_field=("ts" if combo.with_datetime_col else None),
        # 2026-05-04: LTR combos need group_field for the ranker suite.
        group_field=("qid" if _is_ltr else None),
        target_carrier=combo.target_carrier,
        # 2026-05-21 iter150 -- wire weight_schemas through (latent bug:
        # the axis has existed since iter113 but the FTE init was missing
        # the kwarg, so every combo silently fell back to
        # ``sample_weights={}``. Combos still dedup'd distinct via the
        # canonical_key BUT had identical runtime behaviour, leaving
        # the recency-weight code path (FTE._build_sample_weights, the
        # suite's per-weight loop, recency vs uniform branch in
        # _phase_train_one_target) entirely unfuzzed).
        weight_schemas=combo.weight_schemas,
        # 2026-05-21 iter150 -- multi-target axis. FTE adds synthetic
        # extra targets to target_by_type per combo.extra_targets so the
        # suite's per-target outer loop runs more than once.
        extra_targets=combo.extra_targets,
    )

    # Resolve combo-specific kwargs (outlier detector, custom prep,
    # parquet path). These feed directly into train_mlframe_models_suite.
    df_input = _maybe_to_parquet(combo, df, tmp_path)
    outlier_detector = _outlier_detector_for_combo(combo)
    custom_pre = _custom_pre_pipelines_for_combo(combo)

    # Chart/report rendering: mirror the canonical_key gate (small n_rows tier only) so the suite call below renders charts exactly on the
    # combos whose identity reflects rendering-on. Force the matplotlib Agg backend so the figure path runs without a display (headless box).
    _viz_on = bool(combo.enable_viz_rendering_cfg) and combo.n_rows <= 1000
    if _viz_on:
        import matplotlib
        matplotlib.use("Agg", force=True)

    from mlframe.training.core import train_mlframe_models_suite

    # LTR combos: filter mlframe_models to {cb,xgb,lgb} (HGB/Linear have
    # no native ranker) and build a ranking_config from the combo axis.
    # Pass target_type=LEARNING_TO_RANK explicitly so the suite's early
    # dispatch routes to train_mlframe_ranker_suite.
    # 2026-07-13 -- Batch E extra_registry_model_cfg: append one
    # explicit-allowlist-only composite registry key (gated_outlier / bagging /
    # composite_classification) to the sampled model subset when the combo
    # asks for one and it's compatible with the resolved target_type.
    _extra_registry_model = combo._canonical_extra_registry_model()
    _ltr_models = list(combo.models) + (
        [_extra_registry_model] if _extra_registry_model is not None else []
    )
    _ltr_ranking_config = None
    if _is_ltr:
        _supported = {"cb", "xgb", "lgb", "mlp"}  # 2026-05-07: MLP via RankNet/ListNet
        _filtered = [m for m in combo.models if m.lower() in _supported]
        if not _filtered:
            # No supported model in this combo -- skip; not a real bug.
            pytest.skip(
                f"LTR combo {combo.short_id()}: requested models "
                f"{combo.models} have no native ranker (need cb/xgb/lgb/mlp)"
            )
        _ltr_models = _filtered
        from mlframe.training.configs import LearningToRankConfig
        # iter162: nested LTR knobs -- cb_loss_fn, lgb_objective, rrf_k.
        # iter170: mlp_loss_fn + eval_at (defensive).
        _ltr_eval_at = (1, 5, 10) if combo.ltr_eval_at_cfg == "default" else (1, 3, 5, 10, 20)
        _ltr_ranking_config = LearningToRankConfig(
            ensemble_method=combo.ranking_ensemble_method,
            cb_loss_fn=combo.ltr_cb_loss_fn_cfg,
            lgb_objective=combo.ltr_lgb_objective_cfg,
            rrf_k=combo.ltr_rrf_k_cfg,
            **_safe_cfg_kwargs(
                LearningToRankConfig,
                mlp_loss_fn=combo.ltr_mlp_loss_fn_cfg,
                eval_at=_ltr_eval_at,
            ),
        )

    # 2026-05-21 iter151 -- P0 suite-level kwargs (built once per combo
    # and forwarded into _suite_kwargs below). Each is None when the
    # respective axis is disabled, so the production default behaviour
    # is preserved.
    # P0-1 quantile_regression_config: only on regression primaries.
    # iter162 extends with nested crossing_fix / coverage_pairs / wrapper_n_jobs axes.
    _quantile_cfg = None
    if combo.enable_quantile_regression_cfg and combo.target_type == "regression" and not _is_ltr:
        from mlframe.training.configs import QuantileRegressionConfig
        _coverage_pairs = ((0.1, 0.9),) if combo.quantile_coverage_pairs_cfg == "default" else ((0.05, 0.95),)
        _quantile_cfg = QuantileRegressionConfig(
            alphas=(0.1, 0.5, 0.9) if combo.quantile_coverage_pairs_cfg == "default" else (0.05, 0.5, 0.95),
            crossing_fix=combo.quantile_crossing_fix_cfg,
            point_estimate_alpha=0.5,
            coverage_pairs=_coverage_pairs,
            wrapper_n_jobs=combo.quantile_wrapper_n_jobs_cfg,
        )
    # P0-2 linear_model_config: only meaningful when "linear" in models.
    _linear_cfg = None
    if "linear" in combo.models:
        from mlframe.training.configs import LinearModelConfig
        # 2026-05-28 LinearModelConfig.l1_ratio (ElasticNet mix). Only honoured by
        # the saga solver; lbfgs/liblinear raise on l1_ratio != 0. Mirror the
        # canonical_key gating in FuzzCombo so the LinearModelConfig instance
        # the suite consumes never hits sklearn's solver-mismatch ValueError.
        _l1_ratio = (
            combo.linear_l1_ratio_cfg if combo.linear_solver_cfg == "saga" else 0.0
        )
        # liblinear genuinely cannot do multiclass in current sklearn (it
        # raises "The 'liblinear' solver does not support multiclass
        # classification (n_classes >= 3)" -- an actionable sklearn
        # constraint, not an mlframe bug). liblinear+multiclass is therefore
        # an invalid user choice the fuzz harness must not generate; collapse
        # it to the multiclass-safe lbfgs default so the linear leg of a
        # multiclass combo trains instead of crashing the whole suite.
        _solver = combo.linear_solver_cfg
        if _solver == "liblinear" and combo.target_type == "multiclass_classification":
            _solver = "lbfgs"
        _linear_cfg = LinearModelConfig(**_safe_cfg_kwargs(
            LinearModelConfig,
            alpha=combo.linear_alpha_cfg,
            solver=_solver,
            l1_ratio=_l1_ratio,
        ))
    # P0-3 feature_handling_config: instantiate with nested sub-config
    # overrides per the iter162 deep-kwargs audit. Each sub-config field
    # falls back to library defaults when not on the axis OR when import
    # of the relevant sub-config class fails (FHC has heavy optional deps).
    _fhc = None
    if combo.enable_feature_handling_config_cfg:
        try:
            from mlframe.training.feature_handling.config import (
                FeatureHandlingConfig,
                CacheConfig,
                MemoryConfig,
                AutoDeriveConfig,
                TextDetectionConfig,
                ReproConfig,
            )
            _cache = CacheConfig(**_safe_cfg_kwargs(
                CacheConfig,
                eviction_strategy=combo.fhc_cache_eviction_strategy_cfg,
                allow_pickle=combo.fhc_cache_allow_pickle_cfg,
                # iter170 deep cache axes (defensive).
                prefetch_enabled=combo.fhc_cache_prefetch_enabled_cfg,
                prefetch_vram_safety_factor=combo.fhc_cache_prefetch_vram_safety_factor_cfg,
                # iter180 DEPTH-4 -- persistence mode gates disk-tier sub-fields.
                persistence=combo.fhc_cache_persistence_cfg,
            ))
            _memory = MemoryConfig(**_safe_cfg_kwargs(
                MemoryConfig,
                auto_derive=AutoDeriveConfig(cache_ram_fraction=combo.fhc_cache_ram_fraction_cfg),
                # iter170 memory axis (defensive).
                pressure_watermark_pct=combo.fhc_memory_pressure_watermark_pct_cfg,
            ))
            _textdet = TextDetectionConfig(**_safe_cfg_kwargs(
                TextDetectionConfig,
                definite_text_mean_chars=combo.fhc_text_definite_text_mean_chars_cfg,
                min_alphabet_entropy=combo.fhc_text_min_alphabet_entropy_cfg,
                # iter170 text-detection axes (defensive).
                text_min_mean_tokens=combo.fhc_text_min_mean_tokens_cfg,
                text_min_unique_ratio=combo.fhc_text_min_unique_ratio_cfg,
                respect_explicit_categorical_dtype=combo.fhc_text_respect_explicit_cat_dtype_cfg,
                # 2026-05-28 text_min_cardinality axis -- cat-vs-text promotion floor.
                text_min_cardinality=combo.fhc_text_min_cardinality_cfg,
            ))
            _repro = ReproConfig(**_safe_cfg_kwargs(
                ReproConfig,
                deterministic_torch=combo.fhc_repro_deterministic_torch_cfg,
                # iter170 repro axes (defensive).
                langdetect_seed=combo.fhc_repro_langdetect_seed_cfg,
                pinned_svd_solver_params=combo.fhc_repro_pinned_svd_solver_params_cfg,
                forbid_nonatomic_fs=combo.fhc_repro_forbid_nonatomic_fs_cfg,
                deterministic_eviction=combo.fhc_repro_deterministic_eviction_cfg,
            ))
            # iter170: PricingConfig + LoggingConfig (defensive -- may not exist).
            _pricing = None
            _logging = None
            try:
                from mlframe.training.feature_handling.config import PricingConfig
                _pricing = PricingConfig(**_safe_cfg_kwargs(
                    PricingConfig,
                    cap_usd=combo.fhc_pricing_cap_usd_cfg,
                    warn_above_usd=combo.fhc_pricing_warn_above_usd_cfg,
                ))
            except (ImportError, AttributeError):
                pass
            try:
                from mlframe.training.feature_handling.config import LoggingConfig
                _logging = LoggingConfig(**_safe_cfg_kwargs(
                    LoggingConfig,
                    verbose=combo.fhc_logging_verbose_cfg,
                ))
            except (ImportError, AttributeError):
                pass
            _fhc_kw = dict(
                cache=_cache,
                memory=_memory,
                text_detection=_textdet,
                repro=_repro,
                auto_locale_detection=combo.fhc_auto_locale_detection_cfg,
            )
            if _pricing is not None:
                _fhc_kw["pricing"] = _pricing
            if _logging is not None:
                _fhc_kw["logging"] = _logging
            _fhc = FeatureHandlingConfig(**_safe_cfg_kwargs(FeatureHandlingConfig, **_fhc_kw))
        except Exception:
            _fhc = None  # tolerate import / construction failure
    # P0-4 precomputed: build the trainset_features_stats bundle. The
    # other slots (dummy_baselines, composite_target_specs) raise
    # NotImplementedError if requested -- precompute_all only fills the
    # stats slot, which is what we want.
    _precomputed = None
    if combo.enable_precomputed_cfg:
        try:
            from mlframe.training.helpers import precompute_all
            _precomputed = precompute_all(df_input if not isinstance(df_input, str) else df, target_by_type=None)
        except Exception:
            _precomputed = None  # tolerate parquet-path / FTE-shape edge cases

    t0 = time.perf_counter()
    outcome = "pass"
    err_class = None
    err_summary = None
    try:
        _suite_kwargs = dict(
            df=df_input,
            target_name=combo.short_id(),
            model_name=combo.short_id(),
            features_and_targets_extractor=fte,
        )
        # 2026-07-13 -- gated_outlier point-mass auto-detection only fires
        # off the implicit top-level mlframe_models=None default
        # (mlframe_models_is_default_allowlist, see DEFAULTS_CHANGELOG.md).
        # LTR always needs an explicit filtered allowlist (native rankers
        # only), so it keeps the explicit path unconditionally.
        if _is_ltr or combo._canonical_mlframe_models_explicit():
            _suite_kwargs["mlframe_models"] = _ltr_models
        if _is_ltr:
            _suite_kwargs["target_type"] = _combo_tt
            # Wave 21: assume_comparable_scales axis on LTR ensembling.
            _ltr_ranking_config = _ltr_ranking_config.model_copy(
                update={"assume_comparable_scales": combo.ltr_assume_comparable_scales_cfg}
            )
            _suite_kwargs["ranking_config"] = _ltr_ranking_config
        # 2026-05-21 iter151 P0 suite-level kwargs.
        if _quantile_cfg is not None:
            _suite_kwargs["quantile_regression_config"] = _quantile_cfg
        if _linear_cfg is not None:
            _suite_kwargs["linear_model_config"] = _linear_cfg
        if _fhc is not None:
            _suite_kwargs["feature_handling_config"] = _fhc
        if _precomputed is not None:
            _suite_kwargs["precomputed"] = _precomputed
        # RFECV lever kwargs (n_features_selection_rule / stability_selection / enable_permutation_importance /
        # prescreen / swap_top_k) fold into FeatureSelectionConfig.rfecv_kwargs regardless of their value, so a
        # non-default sampled lever with RFECV resolved OFF (rfecv_models=None) trips "rfecv_kwargs supplied but
        # rfecv_models is None/empty" -- canonical_key dedups on the RESOLVED estimator but does not rewrite the
        # raw combo field, so gate the actual kwargs passed here on the SAME resolved predicate that gates
        # rfecv_models below (real ValidationError fuzz surfaced, 2026-07-05).
        _rfecv_on = combo._canonical_rfecv_estimator() is not None
        trained, _meta = train_mlframe_models_suite(
            **_suite_kwargs,
            hyperparams_config=_config_for_models(
                combo.models, combo.n_rows,
                iterations=combo.iterations,
                early_stopping_rounds=combo.early_stopping_rounds_cfg,
                mlp_predict_batch_size=combo.mlp_predict_batch_size_cfg,
                # iter170 per-backend hyperparams.
                lgb_feature_fraction=combo.lgb_feature_fraction_cfg,
                lgb_num_leaves=combo.lgb_num_leaves_cfg,
                xgb_max_depth=combo.xgb_max_depth_cfg,
                xgb_colsample_bynode=combo.xgb_colsample_bynode_cfg,
                cb_border_count=combo.cb_border_count_cfg,
                hgb_max_leaf_nodes=combo.hgb_max_leaf_nodes_cfg,
                rfecv_cv_n_splits=combo.rfecv_cv_n_splits_cfg,
                # 2026-06-03 FS-coverage audit -- RFECV.__init__ knobs.
                rfecv_votes_aggregation=combo.rfecv_votes_aggregation_cfg,
                rfecv_search_method=combo.rfecv_search_method_cfg,
                # iter180 DEPTH-4 booster sub-params.
                lgb_boosting_type=combo.lgb_boosting_type_cfg,
                lgb_dart_drop_rate=combo.lgb_dart_drop_rate_cfg,
                lgb_goss_top_rate=combo.lgb_goss_top_rate_cfg,
                xgb_tree_method=combo.xgb_tree_method_cfg,
                xgb_hist_max_bin=combo.xgb_hist_max_bin_cfg,
                cb_bootstrap_type=combo.cb_bootstrap_type_cfg,
                cb_bayesian_bagging_temperature=combo.cb_bayesian_bagging_temperature_cfg,
                cb_bernoulli_subsample=combo.cb_bernoulli_subsample_cfg,
                cb_grow_policy=combo.cb_grow_policy_cfg,
                cb_lossguide_max_leaves=combo.cb_lossguide_max_leaves_cfg,
            ),
            preprocessing_config=_preprocessing_for_combo(combo),
            verbose=0,
            use_ordinary_models=True,
            use_mlframe_ensembles=combo.use_ensembles,
            outlier_detection_config=OutlierDetectionConfig(
                detector=outlier_detector,
                apply_to_val=combo.apply_outlier_to_val_cfg,
            ),
            feature_selection_config=FeatureSelectionConfig(
                use_mrmr_fs=combo.use_mrmr_fs,
                # 2026-07-13 -- Batch A: all four flipped True by default;
                # fs_new_selectors_enabled_cfg exercises the now-non-default
                # opt-out (False) path for all four together.
                **_safe_cfg_kwargs(
                    FeatureSelectionConfig,
                    use_forward_select_fs=combo.fs_new_selectors_enabled_cfg,
                    use_greedy_backward_elimination_fs=combo.fs_new_selectors_enabled_cfg,
                    use_zero_importance_pruning_fs=combo.fs_new_selectors_enabled_cfg,
                    use_cascade_select_fs=combo.fs_new_selectors_enabled_cfg,
                ),
                # 2026-05-18 -- delegate to shared builder. Adding a new
                # MRMR axis now only edits build_mrmr_kwargs_from_flat in
                # _fuzz_combo.py; the pytest suite + 1M harness both
                # consume the same builder.
                mrmr_kwargs=build_mrmr_kwargs(combo),
                # rfecv_models: pass exactly the canonical estimator (None when
                # the combo would mis-use it) — wrap in a single-element list
                # because the field expects List[str].
                rfecv_models=(
                    [combo._canonical_rfecv_estimator()]
                    if _rfecv_on
                    else None
                ),
                custom_pre_pipelines=custom_pre or {},
                # 2026-05-21 iter151 P1-7/P1-8/P2-16/P2-17/P2-18a/P2-18b:
                # FS-related fill-ins from the audit. Each canonicalised in
                # FuzzCombo.canonical_key when the gating axis is off.
                use_boruta_shap=combo.use_boruta_shap_cfg,
                # 2026-05-21 iter151: BorutaShap fuzz-speed knobs + 2026-06-03/04
                # FS-coverage axes + 5-min budget, built by the signature-guarded
                # helper so a knob that is not yet committed is dropped rather than
                # rejected by the boruta_shap_kwargs validator. None when off.
                boruta_shap_kwargs=_boruta_shap_kwargs_for_combo(combo),
                use_sample_weights_in_fs=combo.use_sample_weights_in_fs_cfg,
                mrmr_identity_cache_scope=combo.mrmr_identity_cache_scope_cfg,
                skip_identity_equivalent_pre_pipelines=combo.skip_identity_equivalent_pre_pipelines_cfg,
                rfecv_leakage_corr_threshold=combo.rfecv_leakage_corr_threshold_cfg,
                rfecv_mbh_adaptive_threshold=combo.rfecv_mbh_adaptive_threshold_cfg,
                # 2026-05-22 iter170 deep FS knobs (defensive).
                **_safe_cfg_kwargs(
                    FeatureSelectionConfig,
                    rfecv_n_features_selection_rule=(combo.rfecv_n_features_selection_rule_cfg if _rfecv_on else None),
                    rfecv_stability_selection=(combo.rfecv_stability_selection_cfg if _rfecv_on else False),
                    rfecv_leakage_action=combo.rfecv_leakage_action_cfg,
                    # 2026-05-28 pre_screen_null_fraction_threshold axis -- the
                    # null-fraction sibling of the existing variance threshold
                    # axis. Gated on fs_pre_screen_unsupervised_cfg in
                    # canonical_key; thread the value through unconditionally
                    # here (the suite already builds the FS config only when
                    # the pre-screen branch fires).
                    pre_screen_null_fraction_threshold=combo.fs_pre_screen_null_fraction_threshold_cfg,
                    # 2026-06-03 FS-coverage audit -- these two axes were
                    # sampled + canonicalised (distinct dedup buckets) and
                    # applied in _build_combo, BUT the value never reached
                    # FeatureSelectionConfig, so every combo ran with the field
                    # defaults (pre_screen_unsupervised=True, variance=0.0). The
                    # False / 0.01 samples were INERT. Thread them through here
                    # so the unsupervised-prescreen OFF branch + the non-zero
                    # variance-floor drop branch actually exercise.
                    pre_screen_unsupervised=combo.fs_pre_screen_unsupervised_cfg,
                    pre_screen_variance_threshold=combo.fs_pre_screen_variance_threshold_cfg,
                    # RFECV first-class lever fields (D-surface). canonical_key collapses each to the dataclass default
                    # unless an RFECV selector is in the chain, so they never split dedup buckets when RFECV is off --
                    # but that's DEDUP identity only, not the value passed here, so gate on ``_rfecv_on`` too (else a
                    # non-default sample folds into rfecv_kwargs with rfecv_models=None and raises).
                    rfecv_enable_permutation_importance=(combo.rfecv_enable_permutation_importance_cfg if _rfecv_on else False),
                    rfecv_prescreen=(combo.rfecv_prescreen_cfg if _rfecv_on else None),
                    rfecv_swap_top_k=(combo.rfecv_swap_top_k_cfg if _rfecv_on else None),
                ),
            ),
            # Chart rendering is OFF by default (the ~150-combo × ~5-fig run compounds to >2 GB and historically blew up pytest's traceback
            # formatter with MemoryError / INTERNALERROR, 2026-04-27). The enable_viz_rendering_cfg axis turns it ON for a gated subset (small
            # n_rows tier only, see canonical_key) so the chart/report-generation code (perf chart, FI, calibration/reliability, slice_finder,
            # model_card, decision_curve, pdp_ice, shap_panels, model_comparison, risk_coverage) actually executes and is fuzz-exercised. The
            # matplotlib Agg backend is forced below so a headless box renders without a display; _fuzz_combo_cleanup plt.close("all")s per combo.
            output_config=OutputConfig(
                data_dir=str(tmp_path), models_dir="models", save_charts=_viz_on,
                # 2026-07-13 -- Batch C: run_diagnostics defaults (None) to all 6
                # registered diagnostics per DEFAULTS_CHANGELOG.md. "subset"/"empty"
                # exercise the explicit-override paths; None keeps the default.
                **_safe_cfg_kwargs(
                    OutputConfig,
                    run_diagnostics=(
                        ["cv_informativeness", "group_leakage"] if combo.run_diagnostics_cfg == "subset"
                        else [] if combo.run_diagnostics_cfg == "empty"
                        else None
                    ),
                ) if combo.run_diagnostics_cfg is not None else {},
            ),
            reporting_config=ReportingConfig(
                show_perf_chart=_viz_on, show_fi=_viz_on,
                # iter162: nested ReportingConfig fields. matplotlib_rcparams
                # parsed from JSON-string axis value (so the axis dict stays
                # hashable for canonical_key).
                prob_histogram_yscale=combo.reporting_prob_histogram_yscale_cfg,
                title_metrics_template=combo.reporting_title_metrics_template_cfg,
                matplotlib_rcparams=(
                    None if combo.reporting_matplotlib_rcparams_cfg is None
                    else __import__("json").loads(combo.reporting_matplotlib_rcparams_cfg)
                ),
                multiclass_panels=combo.reporting_multiclass_panels_cfg,
                # 2026-05-28 W5: ReportingConfig.mase_seasonality (int, default
                # 1 at _reporting_configs.py:140). Thread the fuzz-axis value
                # through so regression combos exercise the non-default
                # seasonality on the report-metadata path.
                mase_seasonality=combo.reporting_mase_seasonality_cfg,
                # iter170 deep reporting axes -- defensive _safe_cfg_kwargs
                # absorbs fields that don't exist post-refactor.
                **_safe_cfg_kwargs(
                    ReportingConfig,
                    figsize=((15, 5) if combo.reporting_figsize_cfg == "default" else (10, 4)),
                    plot_dpi=combo.reporting_plot_dpi_cfg,
                    quantile_panels=(None if combo.reporting_quantile_panels_cfg == "default"
                                     else "RELIABILITY PINBALL_BY_ALPHA"),
                    ltr_panels=(None if combo.reporting_ltr_panels_cfg == "default"
                                else "NDCG_K LIFT"),
                    plotly_template=combo.reporting_plotly_template_cfg,
                    matplotlib_style=combo.reporting_matplotlib_style_cfg,
                ),
            ),
            # 2026-07-13 -- Batch F: RegressionCalibrationConfig.apply_confidence_shrinkage
            # flipped True (DEFAULTS_CHANGELOG.md). Never previously threaded into the fuzz
            # suite call at all (no regression_calibration_config kwarg was passed).
            regression_calibration_config=__import__(
                "mlframe.training._reporting_configs", fromlist=["RegressionCalibrationConfig"]
            ).RegressionCalibrationConfig(
                apply_confidence_shrinkage=combo.apply_confidence_shrinkage_cfg,
            ),
            # recurrent_models + sequences: synthetic per-row sequences
            # (T=8, F=2) emitted only on canonical-recurrent combos so
            # the suite exercises the sequence-pipeline. Hyperparams
            # tuned for fuzz speed (small hidden_size, 2 epochs).
            recurrent_models=(
                [combo._canonical_recurrent_model()]
                if combo._canonical_recurrent_model() is not None
                else None
            ),
            sequences=_recurrent_sequences_for_combo(combo, df=df_input),
            recurrent_config=_recurrent_config_for_combo(combo),
            # 2026-04-28 batch 4 followup - confidence-analysis axis exercises
            # the test-set confidence pass at trainer.py:4019 (distinct code
            # path with its own metrics/report side-effects). ``use_cache``
            # is per-model not suite-level, so it stays out of the fuzz
            # axis space.
            confidence_analysis_config=ConfidenceAnalysisConfig(
                include=combo.include_confidence_analysis_cfg,
                # iter162: nested model_kwargs (n_estimators / max_depth).
                # "default" = empty dict (library defaults); "small_trees" pins
                # tiny trees so the conf-analysis branch runs faster in fuzz.
                model_kwargs=(
                    {} if combo.confidence_model_kwargs_cfg == "default"
                    else {"n_estimators": 20, "max_depth": 4}
                ),
            ),
            # Wave 21: dummy-baselines + baseline-diagnostics enabled toggles.
            dummy_baselines_config=__import__(
                "mlframe.training.configs", fromlist=["DummyBaselinesConfig"]
            ).DummyBaselinesConfig(
                enabled=combo.dummy_baselines_enabled_cfg,
                # iter170 deep dummy-baseline axes (defensive).
                **_safe_cfg_kwargs(
                    __import__("mlframe.training.configs", fromlist=["DummyBaselinesConfig"]).DummyBaselinesConfig,
                    stratified_n_repeats=combo.dummy_stratified_n_repeats_cfg,
                    paired_bootstrap_n_resamples=combo.dummy_paired_bootstrap_n_resamples_cfg,
                ),
            ),
            baseline_diagnostics_config=__import__(
                "mlframe.training.configs", fromlist=["BaselineDiagnosticsConfig"]
            ).BaselineDiagnosticsConfig(
                enabled=combo.baseline_diagnostics_enabled_cfg,
                # iter170 deep baseline-diagnostic axes (defensive).
                **_safe_cfg_kwargs(
                    __import__("mlframe.training.configs", fromlist=["BaselineDiagnosticsConfig"]).BaselineDiagnosticsConfig,
                    quick_model_n_estimators=combo.baseline_quick_model_n_estimators_cfg,
                    quick_model_num_leaves=combo.baseline_quick_model_num_leaves_cfg,
                    quick_model_learning_rate=combo.baseline_quick_model_learning_rate_cfg,
                    sample_n=combo.baseline_sample_n_cfg,
                    high_potential_min_dominance_pct=combo.baseline_high_potential_min_dominance_pct_cfg,
                    best_model_min_lift=combo.baseline_best_model_min_lift_cfg,
                ),
            ),
            # 2026-05-21 -- mini-HPT toggle. When True, the suite runs the
            # target-distribution analyzer + feature-distribution analyzer
            # after the split, gap-fill-merges target-side recommendations
            # into hyperparams_config, and stamps both reports into metadata.
            # When False both analyzers skip. Default True matches suite
            # signature; toggling exercises the skip path.
            enable_target_distribution_analyzer=combo.enable_target_distribution_analyzer_cfg,
            **_configs_for_combo(combo),
        )
        # An empty ``trained`` dict is acceptable ONLY when
        # ``continue_on_model_failure=True`` AND the suite recorded
        # each failure in ``metadata['failed_models']``. Any other
        # empty-trained outcome is a bug — the suite should have
        # either raised or produced ≥1 model.
        if not trained:
            if (
                combo.continue_on_model_failure
                and _meta is not None
                and _meta.get("failed_models")
            ):
                pass  # graceful skip of a configurably-failing combo
            else:
                raise AssertionError(
                    f"empty models dict for combo {combo.short_id()} "
                    f"(continue_on_failure={combo.continue_on_model_failure}, "
                    f"failed_models={(_meta or {}).get('failed_models')})"
                )

        # --- Post-train invariants (free on every combo) ---
        # #16 no caller-frame mutation (skip for parquet-path).
        if combo.input_storage == "memory" and frame_cols_before is not None:
            assert tuple(df.columns) == frame_cols_before, (
                f"caller-frame columns mutated: before={frame_cols_before} "
                f"after={tuple(df.columns)}"
            )
            shape_after = getattr(df, "shape", None)
            assert shape_after == frame_shape_before, (
                f"caller-frame shape mutated: before={frame_shape_before} "
                f"after={shape_after}"
            )
        # #20 metadata schema: load-bearing keys present.
        # ``model_schemas`` is only populated when at least one model
        # successfully trained — combos that legitimately degrade to
        # an empty trained dict (continue_on_failure=True + all models
        # failed) won't have it. Check the always-present keys
        # unconditionally; model_schemas only when trained non-empty.
        if _meta is not None:
            for k in ("columns", "cat_features", "outlier_detection"):
                assert k in _meta, (
                    f"metadata missing load-bearing key {k!r}; "
                    f"keys={list(_meta)[:20]}"
                )
            if trained:
                assert "model_schemas" in _meta, (
                    "metadata missing 'model_schemas' despite non-empty "
                    f"trained dict; keys={list(_meta)[:20]}"
                )

        # --- Fix C property invariants (cheap, per-combo) ---
        # Catches silent degeneracy that a "no exception" assertion misses:
        # dead features, all-zero predictions, NaN leakage to the model
        # head, val-slice misalignment.
        _assert_prediction_invariants(trained, _meta, combo)
        # --- R3-3 I4 serialization roundtrip (env-gated, off by default) ---
        if os.environ.get("MLFRAME_FUZZ_ROUNDTRIP") == "1":
            _assert_serialization_roundtrip(trained, str(tmp_path), combo)
    except Exception as exc:
        outcome = "fail"
        err_class = type(exc).__name__
        err_summary = traceback.format_exception_only(type(exc), exc)[-1].strip()
        log_combo_outcome(
            combo, outcome,
            duration_s=time.perf_counter() - t0,
            error_class=err_class,
            error_summary=err_summary,
        )
        raise

    log_combo_outcome(
        combo, outcome, duration_s=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------------
# Meta-tests: sanity-check the enumerator itself
# ---------------------------------------------------------------------------


def test_enumerator_is_deterministic():
    """Same master_seed must yield byte-identical combo list."""
    a = enumerate_combos(target=50, master_seed=2026_04_22)
    b = enumerate_combos(target=50, master_seed=2026_04_22)
    assert [c.canonical_key() for c in a] == [c.canonical_key() for c in b]


def test_enumerator_produces_unique_combos():
    """No canonical-key duplicates in the 150-combo run."""
    keys = [c.canonical_key() for c in COMBOS]
    assert len(keys) == len(set(keys)), "Fuzz enumerator produced duplicates"


def test_enumerator_hits_all_models():
    """Every supported model must appear at least once across the 150 combos."""
    from tests.training._fuzz_combo import MODELS
    seen = {m for c in COMBOS for m in c.models}
    missing = set(MODELS) - seen
    assert not missing, f"Models never exercised by fuzz: {missing}"


def test_enumerator_target_count():
    assert len(COMBOS) == 150
