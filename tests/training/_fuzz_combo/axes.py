"""Axis space for train_mlframe_models_suite fuzzing: MODELS + the AXES grid."""
from __future__ import annotations

from typing import Any


MODELS: tuple[str, ...] = ("cb", "xgb", "lgb", "hgb", "linear", "mlp")

AXES: dict[str, tuple[Any, ...]] = {
    "input_type": ("pandas", "polars_utf8", "polars_enum", "polars_nullable"),
    # 2026-05-21: bumped from (300, 600, 1000, 5000) to (1000, 200_000) so the
    # large tier matches the /loop iteration target (profile_one_combo.py
    # --rows 200000) and the size the library is actually deployed against.
    # Most recent features (FE subsample, mini-HPT, composite-discovery,
    # mrmr GPU dispatch, kernel_tuning_cache, target-distribution-analyzer)
    # only branch above ~50k rows; without a 200k tier the fuzz suite never
    # exercises those code paths.
    #
    # 1000 stays as the "small tier" so the n_rows-gated slow models
    # (recurrent at >600, RFECV at >1200) still get fuzz coverage. Both
    # canonicalisers (rare_1pct min=4000, rare_5pct min=800) tolerate 1000:
    # rare_5pct still fuzz-active, rare_1pct canonicalises to balanced
    # (covered by the 200k tier where rare_1pct fits comfortably).
    "n_rows": (1000, 200_000),
    # cat_feature_count 15 stresses one-hot blow-up (~15 levels × 15 cols ~=
    # 225 generated features) and exercises the high-card branch of MRMR /
    # encoder. Was 20 in 2026-04 — reduced 2026-04-27 for fuzz speed; the
    # one-hot blow-up code path is the same, just the absolute feature count
    # is smaller.
    "cat_feature_count": (0, 1, 3, 8, 15),
    "null_fraction_cats": (0.0, 0.1, 0.3),
    "use_mrmr_fs": (False, True),
    # 2026-05-21 iter150: added ("recency",) -- non-uniform-only schema.
    # Exercises the _phase_train_one_target.py:1498 "uniform weighting not
    # included" log branch + paths where every sample gets non-uniform
    # weights (which is a real prod path when caller disables
    # use_uniform_weighting). Pre-iter150 fuzz only exercised the
    # uniform-present branches.
    "weight_schemas": (("uniform",), ("uniform", "recency"), ("recency",)),
    # 2026-04-24 Session 3: multilabel_classification re-added — FTE now
    # 2-D-aware (Session-2 landing); multilabel combos generate (N, 3)
    # targets via build_frame_for_combo's correlated-label logic and the
    # fuzz test runner uses MultilabelDispatchConfig via multilabel_strategy_cfg.
    "target_type": (
        "binary_classification",
        "regression",
        "multiclass_classification",
        "multilabel_classification",
        # 2026-05-04: LTR axis -- frame builder emits graded relevance
        # 0..3 + qid column; canonicaliser forces sensible defaults
        # (group_field='qid', mlframe_models in {cb,xgb,lgb}, RRF
        # ensembling). Non-LTR combos canonicalise to no group_field.
        "learning_to_rank",
        # 2026-05-31 audit-pass-9 #8 (F-24): K-independent-target
        # regression. Enum lives at _configs_base.py:126. Frame builder
        # branch emits y of shape (N, K>=2) float32 so the new estimator
        # code path (num_classes=K head sharing trunk, MSE on (N, K))
        # surfaces in fuzz coverage. Canon: collapses to "regression"
        # when only 1-D y is reachable on the combo (e.g. CB+LGB lack
        # native multi-target regression -- the suite-side dispatch
        # design doc is the source of truth, but canon stays defensive:
        # only models with documented native MTR support keep the value).
        "multi_target_regression",
    ),
    # 2026-05-13: target carrier itself is a fuzz axis. The old fixture
    # always normalised Polars targets to numpy, so MRMR never saw the
    # production ``pl.Series`` path that crashed on ``y.values``.
    "target_carrier": ("numpy", "native"),
    "multilabel_strategy_cfg": ("auto", "wrapper", "chain"),  # parametric on multilabel; canonicalised to "auto" for non-multilabel
    # 2026-05-04: ranking ensemble method -- only relevant when
    # target_type=learning_to_rank. Canonicaliser collapses to "rrf" for
    # non-LTR combos so the combo identity remains stable.
    "ranking_ensemble_method": ("rrf", "borda"),
    "auto_detect_cats": (True, False),
    "align_polars_categorical_dicts": (True, False),
    # 2026-04-24 expansion: flags that previously had NO coverage despite
    # being runtime-visible knobs. AXES-driven ≠ actually-wired — these
    # flags need corresponding prop-through in test_fuzz_suite.py runner.
    "prefer_polarsds": (True, False),
    "use_text_features": (True, False),
    "honor_user_dtype": (True, False),
    # Rare column types that the frame builder previously omitted — this
    # is the gap the production 'skills_text' / embedding features caught
    # us on. A non-zero count forces the builder to emit at least one
    # high-cardinality text column or one pl.List embedding column.
    "text_col_count": (0, 1),
    "embedding_col_count": (0, 1),
    # 2026-04-24 combo extension — pull the test_suite_coverage_gaps
    # gap-analysis items into the fuzz axis space so cross-axis
    # interactions are exercised (e.g. OD × polars_utf8 × MRMR × linear).
    # Each axis doubles the pairwise-coverage work; the pairwise sampler
    # keeps the combo count at the target by selecting informative
    # combinations rather than a full cartesian product.
    "outlier_detection": (None, "isolation_forest", "lof", "ocsvm"),  # #3, batch 3 +LOF/OCSVM
    "use_ensembles": (False, True),                              # #5
    "continue_on_model_failure": (False, True),                  # #21
    # iterations: 3 (single-iter sanity) + 10 (multi-iter ES/convergence).
    # Was (3, 30) in 2026-04 — 30 reduced to 15 on 2026-04-27 for fuzz speed;
    # 15 -> 10 on 2026-05-23 (iter185) per user instruction "у всех моделей
    # сделай поменьше дефолтное число итераций". Multi-iter boosting code path
    # is the same and ES still triggers (default patience 5-10).
    "iterations": (3, 10),                                       # #15
    "prefer_calibrated_classifiers": (False, True),              # #32
    "inject_degenerate_cols": (False, True),                     # #7 (const + all-null)
    "inject_inf_nan": (False, True),                             # #10
    "with_datetime_col": (False, True),                          # #11
    "inject_zero_col": (False, True),                            # #40 (uninformative)
    "fairness_col": (None, "cat_0"),                             # #31
    "custom_prep": (None, "pca2"),                               # #29
    "input_storage": ("memory", "parquet"),                      # #33
    # 2026-04-24 (round 2): config fields previously hard-coded to
    # defaults despite being user-facing knobs. Each axis exercises
    # a distinct code path that prior fuzz couldn't reach.
    "fillna_value_cfg": (None, 0.0),                             # PreprocessingConfig.fillna_value
    "scaler_name_cfg": ("standard", "robust", None),             # PreprocessingBackendConfig.scaler_name
    "categorical_encoding_cfg": ("ordinal", "onehot"),           # PreprocessingBackendConfig.categorical_encoding
    "skip_categorical_encoding_cfg": (False, True),              # PreprocessingBackendConfig.skip_categorical_encoding
    "val_placement_cfg": ("forward", "backward"),                # TrainingSplitConfig.val_placement
    "test_size_cfg": (0.1, 0.2),                                 # TrainingSplitConfig.test_size
    "trainset_aging_limit_cfg": (None, 0.5),                     # TrainingSplitConfig.trainset_aging_limit
    "cat_text_card_threshold_cfg": (50, 300),                    # FeatureTypesConfig.cat_text_cardinality_threshold
    "early_stopping_rounds_cfg": (None, 10),                     # ModelHyperparamsConfig.early_stopping_rounds — was 20 in 2026-04, reduced 2026-04-27 for fuzz speed (still tests ES path with smaller patience)
    "use_robust_eval_metric_cfg": (False, True),                 # TrainingBehaviorConfig.use_robust_eval_metric
    # 2026-04-24 (Fix G): adversarial axis values — synthetic patterns
    # that stress-test the pipeline for bugs real-world synthetic data
    # alone cannot reach. Each of these is a 2-value axis.
    "inject_label_leak": (False, True),                          # feature = target + ε; val metric must be near-perfect
    "inject_rank_deficient": (False, True),                      # colinear feature pair; linear-model edge
    "inject_all_nan_col": (False, True),                         # whole column is NaN; pipeline guard test
    # 2026-04-24 (R3): drift, imbalance, weird-cat axes.
    "inject_test_drift": (None, "unseen_category", "out_of_range_numeric", "shifted_distribution"),  # R3-1
    "imbalance_ratio": ("balanced", "rare_5pct", "rare_1pct"),   # R3-4
    "weird_cat_content": (None, "empty", "unicode", "null_like"),# R3-5
    # 2026-04-26 batch 1 — config fields previously hard-coded to defaults.
    "fix_infinities_cfg": (True, False),                         # PreprocessingConfig.fix_infinities
    "ensure_float32_cfg": (True, False),                         # PreprocessingConfig.ensure_float32_dtypes
    "remove_constant_columns_cfg": (True, False),                # PreprocessingConfig.remove_constant_columns
    "imputer_strategy_cfg": ("mean", "median", "most_frequent", None),  # PreprocessingBackendConfig.imputer_strategy
    "shuffle_val_cfg": (False, True),                            # TrainingSplitConfig.shuffle_val
    "shuffle_test_cfg": (False, True),                           # TrainingSplitConfig.shuffle_test
    "wholeday_splitting_cfg": (True, False),                     # TrainingSplitConfig.wholeday_splitting
    "val_sequential_fraction_cfg": (0.0, 0.5, 1.0),              # TrainingSplitConfig.val_sequential_fraction
    # 2026-04-26 batch 3 — multilabel dispatch fields (only meaningful when
    # target_type=multilabel + strategy=chain). Now actually wired through
    # train_mlframe_models_suite → select_target → configure_training_params
    # → strategy.wrap_multilabel via the new multilabel_dispatch_config kwarg.
    "multilabel_n_chains_cfg": (2, 3),                           # MultilabelDispatchConfig.n_chains — was (3, 5) in 2026-04, reduced 2026-04-27 for fuzz speed (chain dispatch path identical, fewer chains)
    "multilabel_chain_order_cfg": ("random", "by_frequency"),    # MultilabelDispatchConfig.chain_order_strategy
    "multilabel_cv_cfg": (2, 3),                                 # MultilabelDispatchConfig.cv — was (3, 5) in 2026-04, reduced 2026-04-27 for fuzz speed (CV path identical, fewer folds)
    # 2026-04-26 batch 4 — PreprocessingExtensionsConfig (entire config was
    # untested). When ANY of these is non-None the sklearn-bridge fires in
    # fit_and_transform_pipeline (polars-native fastpath bypassed) — even
    # tree models then consume the shared transformed frame.
    "prep_ext_scaler_cfg": (None, "StandardScaler", "RobustScaler"),  # PreprocessingExtensionsConfig.scaler
    "prep_ext_kbins_cfg": (None, 5),                             # PreprocessingExtensionsConfig.kbins
    "prep_ext_polynomial_degree_cfg": (None, 2),                 # PreprocessingExtensionsConfig.polynomial_degree
    "prep_ext_dim_reducer_cfg": (None, "PCA", "TruncatedSVD"),   # PreprocessingExtensionsConfig.dim_reducer
    "prep_ext_nonlinear_cfg": (None, "RBFSampler"),              # PreprocessingExtensionsConfig.nonlinear_features
    # 2026-05-15 — PySR symbolic regression: gated to small combos because
    # each PySR fit triggers Julia compilation + Pareto-front search (30-60s+).
    # Canonicalised to False for: classification targets (PySR is regression-only),
    # n_rows > 2000 (sample-down inside _apply_pysr_fe but still slow), text/
    # embedding cols (the temp _pysr_y_ injection assumes numeric frame).
    "prep_ext_pysr_enabled_cfg": (False, True),                  # PreprocessingExtensionsConfig.pysr_enabled
    # 2026-05-15 — MRMR NaN handling strategy. Drives _validate_inputs +
    # categorize_dataset behaviour. Three values currently supported; canon
    # to "separate_bin" when use_mrmr_fs is False (axis is a no-op then).
    "mrmr_nan_strategy_cfg": ("separate_bin", "ffill_bfill", "fillna_zero"),
    # 2026-04-26 batch 5 — RFECV (Recursive Feature Elimination with CV).
    # rfecv_models is the list passed to train_mlframe_models_suite; the
    # axis picks ONE estimator name. lgb_rfecv requires GPU build, skip.
    # Canonicalised to None when the underlying model isn't in
    # combo.models (rfecv needs an mlframe model to wrap) and when n_rows
    # is large (RFECV's iterative re-fit dominates fuzz runtime).
    "rfecv_estimator_cfg": (None, "cb_rfecv", "xgb_rfecv"),
    # 2026-04-26 batch 6 — recurrent models (lstm/gru/transformer). Need
    # a parallel ``sequences`` argument (list of (T, F) np.ndarray) plus
    # a companion tabular df. Canonicalised to None for combos whose
    # axes break the sequence-builder contract (multilabel target,
    # text/embedding cols, large n_rows where training time blows up).
    "recurrent_model_cfg": (None, "lstm", "gru", "transformer"),
    # 2026-04-28 batch 4 followup - ConfidenceAnalysisConfig.include adds
    # the test-set confidence pass at ``trainer.py:4019`` (distinct code
    # path with its own metrics/report side-effects). ``use_cache`` is
    # per-model not suite-level, so it stays out of the fuzz axis space.
    "include_confidence_analysis_cfg": (False, True),
    # 2026-05-11 Wave 15: MRMR-internal knobs. Prior to this all 75
    # use_mrmr_fs=True combos used identical hardcoded MRMR kwargs
    # (full_npermutations=2, quantization_nbins=5, fe_max_steps=default
    # =1, interactions_max_order=default=1, cat_fe=default=enabled).
    # Wave 14 fixes (categorical_vars after cat-FE; polars->pandas via
    # ``.to_pandas()``; ``_FIT_CACHE`` col_names) require explicit fuzz
    # axis coverage so future regressions surface deterministically.
    # Canonicalised to defaults when ``use_mrmr_fs=False`` so the dedup
    # pass collapses identical-behaviour entries.
    "mrmr_interactions_max_order_cfg": (1, 2, 3),
    "mrmr_fe_max_steps_cfg": (0, 1, 2),
    "mrmr_cat_fe_enable_cfg": (True, False),
    # 2026-05-11 Wave 21: high-value missing axes from full configs.py audit.
    # Each toggles a distinct code path that prior fuzz axes did not exercise.
    # Canonicalised individually below.
    "dummy_baselines_enabled_cfg": (True, False),
    "baseline_diagnostics_enabled_cfg": (True, False),
    # ``auto_detect_feature_types`` is already covered by the legacy
    # ``auto_detect_cats`` axis (wired via FeatureTypesConfig); don't
    # double-add. ``use_groups`` toggles the group-aware splitter path
    # (groupwise validation when wholeday_splitting + datetime present).
    "use_groups_cfg": (True, False),
    "apply_outlier_to_val_cfg": (True, False),
    "multilabel_allow_uncalibrated_cfg": (True, False),
    "report_residual_audit_cfg": (True, False),
    "ltr_assume_comparable_scales_cfg": (True, False),
    # 2026-05-18 — composite-target discovery (Packs J + K end-to-end).
    # When enabled the CompositeTargetDiscovery searches for a transform
    # T = f(y, base) such that the model on T outperforms the model on
    # raw y. Default OFF; the fuzz axis turns it on for regression
    # combos so the discovery loop + per-base scoring + auto-promotion
    # actually run end-to-end inside the suite. Canonicalised to False
    # for non-regression targets (the discovery is regression-only).
    "composite_discovery_enabled_cfg": (False, True),
    # 2026-05-18 — composite-discovery transform palette. ``None`` =
    # the full default 14 (6 legacy + 4 Pack J unary + 4 Pack K chain).
    # ``unary_only`` restricts to the Pack J unary transforms (cbrt_y,
    # log_y, yeo_johnson_y, quantile_normal_y) — exercises the
    # ``requires_base=False`` path and the discovery-loop dedup that
    # evaluates each unary once across all bases. ``chain_only``
    # restricts to the Pack K bivariate+unary chains
    # (chain_linres_cbrt, chain_linres_yj, chain_monres_cbrt,
    # chain_monres_yj) — exercises the chain composer. ``legacy``
    # restricts to the pre-Pack-J/K set so dedup-vs-default regressions
    # surface. Canonicalised to None when discovery is disabled.
    "composite_transforms_mode_cfg": (None, "unary_only", "chain_only", "legacy"),
    # 2026-05-18 -- MRMR feature-engineering FE-search knobs. Each
    # toggles a distinct code path inside ``mrmr.fit`` that prior fuzz
    # axes did NOT exercise. All canonicalise to defaults when
    # ``use_mrmr_fs=False`` so dedup collapses identical-behaviour
    # combos.
    #
    # fe_npermutations: classical unary/binary FE permutation-confirmation
    # budget. 0 = disabled fast path; >0 fires the permutation-test
    # confirmation step. Pin small (10) for fuzz so the inner FE pass
    # doesn't dominate runtime.
    "mrmr_fe_npermutations_cfg": (0, 10),
    # fe_ntop_features: how many top-ranked features get pollinated with
    # unary/binary transforms. 0 = FE step OFF; >0 = small pollination
    # pool. Pin small for fuzz speed.
    "mrmr_fe_ntop_features_cfg": (0, 5),
    # fe_unary_preset: which preset of unary transforms to generate.
    # MRMR ships {"minimal","medium","maximal"}; fuzz skips "maximal"
    # (too slow on inner-loop FE) but covers the minimal->medium path.
    "mrmr_fe_unary_preset_cfg": ("minimal", "medium"),
    # fe_binary_preset: same for binary transforms (hypot/atan2/...).
    "mrmr_fe_binary_preset_cfg": ("minimal", "medium"),
    # fe_smart_polynom_iters: smart orthogonal-polynom FE via Optuna.
    # 0 = disabled; >0 = N study-restarts. Pin small (1) for fuzz.
    "mrmr_fe_smart_polynom_iters_cfg": (0, 1),
    # fe_smart_polynom_optimization_steps: trials within one study.
    # Only effective when iters > 0; pin tiny (10) for fuzz so the
    # Optuna sweep doesn't blow runtime.
    "mrmr_fe_smart_polynom_steps_cfg": (10,),
    # fe_(min,max)_polynom_degree: polynomial degree range. Library
    # default is (3, 8); fuzz tightens the upper bound to keep the
    # search space cheap.
    "mrmr_fe_min_polynom_degree_cfg": (3,),
    "mrmr_fe_max_polynom_degree_cfg": (3, 5),
    # CatFEConfig.include_numeric: when True, MRMR's cat-FE also pulls
    # in discretized numeric columns alongside categoricals. Default
    # False (avoid spurious aliasing from noisy floats).
    "mrmr_cat_fe_include_numeric_cfg": (False, True),
    # 2026-05-19 — PreprocessingExtensionsConfig knobs added in iter-69
    # (byte-aware polynomial auto-tune) and the 390-finding audit
    # (polynomial_max_features cap, polynomial_interaction_only default
    # flip). Prior fuzz coverage stopped at polynomial_degree -- without
    # these axes the auto-tune branches (flip interaction_only,
    # decrement degree, skip polynomial) have no fuzz exposure.
    #
    # polynomial_max_features: None / 0 = disabled; non-zero = projected
    # cols above the cap trigger auto-tune. Pin small (100) to force the
    # auto-tune path so canonicalisation can't silently absorb it.
    "prep_ext_polynomial_max_features_cfg": (None, 100, 10_000),
    # polynomial_interaction_only: now True by default (2026-05-18). False
    # exercises the legacy pure-power-term path that the auto-tune flips
    # away from first.
    "prep_ext_polynomial_interaction_only_cfg": (True, False),
    # memory_safety_max_bytes: 500MB default; None disables the byte-cap
    # (column-count-only behaviour). A tight cap (1MB) forces the
    # byte-aware auto-tune path on wide+long frames.
    "prep_ext_memory_safety_max_bytes_cfg": (None, 1_000_000, 500_000_000),
    # 2026-05-19 — composite-discovery stacked-residual knobs.
    # use_stacked_discovery enables the parallel "fit_stacked_on_raw"
    # path. use_stacked_discovery_residual is the residual variant
    # (mutually exclusive with use_stacked_discovery; residual wins per
    # configs.py:2128). skip_wrap_pass_predict toggles whether predict
    # replays the per-component wrap pass (default True saves 5-15min on
    # large suites). Canonicalised away when composite discovery is OFF.
    "composite_use_stacked_discovery_cfg": (False, True),
    "composite_use_stacked_discovery_residual_cfg": (False, True),
    "composite_skip_wrap_pass_predict_cfg": (True, False),
    # 2026-05-22 — six gate-flip axes from the TVT-MLP-collapse cascade.
    # Each pair holds (post-fix default, pre-fix default). The pre-fix
    # value is kept in the fuzz space so a regression that re-enables
    # one of the gates still gets exercised against the full model zoo
    # via fuzz coverage; canonicalised away when composite discovery
    # is OFF.
    "composite_skip_raw_dominates_ratio_cfg": (0.0, 0.03),
    "composite_skip_ablation_delta_pct_cfg": (0.0, 500.0),
    "composite_eps_mi_gain_cfg": (-10.0, -0.5),
    "composite_top_k_after_mi_cfg": (32, 8),
    "composite_require_beats_raw_baseline_cfg": (False, True),
    "composite_per_bin_n_bins_cfg": (0, 5),
    # 2026-05-22 — tiny-screening proxy axis. "per_family" with
    # ("lightgbm","linear") is the new default; "single_lgbm" is the
    # legacy proxy that hurt downstream linear / neural models.
    "composite_tiny_screening_mode_cfg": ("per_family", "single_lgbm"),
    # 2026-05-22 — additive_residual transform inclusion (default ON).
    # Disable via False to verify the discovery still functions on the
    # legacy transform set + that fallback compositions still emerge.
    "composite_include_additive_residual_cfg": (True, False),
    # 2026-05-22 — MLP activation footgun axis. Currently NOT plumbed
    # into the suite call (fuzz suite uses MLP defaults), so the axis
    # exists for completeness but doesn't yet flip the network config.
    # Activate via a follow-up that threads ``hyperparams_config`` into
    # ``_config_for_models``. Until then the direct unit test
    # ``test_tvt_mlp_audit_followups.TestIdentityMLPGuard`` covers the
    # train-time guard.
    "mlp_activation_cfg": ("ReLU", "Identity"),
    # 2026-05-21 -- mini-HPT analyzer (target + feature side). When True,
    # ``train_mlframe_models_suite`` runs both analyze_target_distribution
    # and analyze_feature_distribution after the split; target-side
    # recommendations gap-fill-merge into hyperparams_config, feature-side
    # report is stamped into metadata["feature_distribution_report"] for
    # operator review. False skips both analyzers. Default True matches
    # the suite signature default; toggling exercises the skip path.
    "enable_target_distribution_analyzer_cfg": (True, False),
    # 2026-05-21 -- MRMR FE pair-check subsample knob. When 0, the check
    # runs on the full frame (legacy + n_rows < FE_DEFAULT_SUBSAMPLE_N
    # path). When set positive AND below n_rows, the MI sweep runs on a
    # uniform sample of that size while survivor columns are still rebuilt
    # at full-n (caller contract). 50_000 forces the subsample path at
    # n_rows=200_000 so the survivor-rebuild branch + _rebuild_full_survivor_col
    # are exercised; canonicalised to 0 (no-op) when use_mrmr_fs is False
    # OR fe_ntop_features_cfg / fe_npermutations_cfg are both 0 (the FE
    # block doesn't run at all).
    "fe_check_pairs_subsample_n_cfg": (0, 50_000),
    # 2026-05-21 iter150 -- multi-target / multi-target-type combos. Pre-
    # iter150 every fuzz combo emitted exactly ONE (target_type, target_name)
    # entry into target_by_type, so the suite's per-target outer loop at
    # core/main.py:952-957 had zero multi-iteration coverage. The suite
    # ITERATES BOTH target_type.items() AND targets.items() within type, so
    # this axis exercises both axes simultaneously.
    #
    # Values:
    #   None                  -- legacy single-target behaviour.
    #   "same_type_2"         -- 2 distinct targets of the SAME primary type
    #                            (e.g. predict 2 regression targets); exercises
    #                            targets.items() inner loop.
    #   "mixed_reg_bin"       -- regression primary + binary classification
    #                            secondary; exercises target_by_type.items()
    #                            outer loop with 2 different keys.
    #   "mixed_reg_bin_2each" -- 2 regression + 2 binary classification
    #                            simultaneously. Exercises BOTH the outer
    #                            target_by_type.items() loop (2 keys) AND
    #                            the inner targets.items() loop (2 names
    #                            per type), so the cross-product 4 targets
    #                            are trained in one suite invocation. The
    #                            hardest combo: stress-tests per-target
    #                            isolation (no cross-contamination of
    #                            preprocessing / FS caches between targets),
    #                            ensemble flavour assembly across heterogeneous
    #                            target types, and metadata layout for the
    #                            ``{target_type: {target_name: ...}}`` 2-level dict.
    #
    # Canonicalised to None for ``multilabel_classification`` (already 2-D
    # within one target) and ``learning_to_rank`` (special ranker dispatch
    # path that doesn't iterate over multiple targets). "mixed_reg_bin" +
    # "mixed_reg_bin_2each" additionally canonicalise to None when primary
    # != regression.
    "extra_targets": (None, "same_type_2", "mixed_reg_bin", "mixed_reg_bin_2each"),
    # =====================================================================
    # 2026-05-21 iter151 -- P0/P1/P2 audit fill-in.
    # Each axis below was identified by the explore-agent audit of the
    # train_mlframe_models_suite signature as either (a) a suite kwarg
    # never passed by the fuzz runner, (b) a config field with documented
    # behavioural effect and zero pre-iter151 fuzz coverage, or (c) a
    # tuning knob whose non-default branch was never exercised.
    # =====================================================================
    # P0-1: quantile_regression_config. Pre-iter151 the entire quantile
    # regression dispatch path (CB MultiQuantile / XGB quantileerror /
    # LGB wrapper / HGB wrapper / Linear QuantileRegressor) was unfuzzed.
    # When True AND target_type=regression, the runner builds a
    # QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9)) and passes it via
    # the ``quantile_regression_config`` kwarg. Canonicalised to False
    # for non-regression primaries (quantile dispatch is regression-only).
    "enable_quantile_regression_cfg": (False, True),
    # P0-2: linear_model_config. Pre-iter151 every linear model used
    # defaults. Two axes pin the most impactful knobs:
    #   - linear_alpha_cfg controls regularisation strength
    #     (1.0 default vs 0.01 very-light vs 100.0 heavy).
    #   - linear_solver_cfg controls logistic-regression solver path
    #     (lbfgs default / liblinear / saga).
    # Canonicalised to defaults when "linear" not in combo.models.
    "linear_alpha_cfg": (1.0, 0.01, 100.0),
    "linear_solver_cfg": ("lbfgs", "liblinear", "saga"),
    # P0-3: feature_handling_config. Pre-iter151 the polars-native FHC
    # fastpath was entirely unfuzzed (the legacy
    # split_config/pipeline_config/preprocessing_config path always
    # picked instead). When True the runner builds a default
    # FeatureHandlingConfig() and passes via ``feature_handling_config``.
    "enable_feature_handling_config_cfg": (False, True),
    # P0-4: precomputed. TrainMlframeSuitePrecomputed bundle was never
    # injected -- only the inline-compute path had coverage. When True
    # the runner builds a bundle via precompute_all() and passes via
    # ``precomputed``. Exercises the cache-reuse path in the suite.
    "enable_precomputed_cfg": (False, True),
    # P1-5: TrainingSplitConfig.test_sequential_fraction. None (default,
    # uniform-random test) or 0.5 (50% of test rows pulled from the
    # tail of the time axis). Untested pre-iter151.
    "test_sequential_fraction_cfg": (None, 0.5),
    # P1-6: TrainingSplitConfig.calib_size. None (no post-hoc
    # calibration set) or 0.05 (5% calib carve-out). The
    # calibration-reserve handoff path in the trainer was unfuzzed.
    "calib_size_cfg": (None, 0.05),
    # P1-7: FeatureSelectionConfig.use_boruta_shap. False (default,
    # no SHAP-driven FS) or True. Boruta-SHAP path (tree feature
    # importance + shadow features) had zero fuzz coverage.
    "use_boruta_shap_cfg": (False, True),
    # 2026-06-03: BorutaShap importance driver (default flipped to "gini" in
    # 8b3994da). 'gini' = tree feature_importances_, 'shap' = SHAP values; both
    # leak the top features but differ in speed/ranking, so both paths need
    # fuzz coverage. Gated to use_boruta_shap_cfg in canonical_key; wired into
    # boruta_shap_kwargs in test_fuzz_suite.py.
    "boruta_importance_measure_cfg": ("gini", "shap"),
    # 2026-06-03 FS-coverage audit: BorutaShap.__init__ knobs that were
    # tunable but never fuzzed. All forwarded into boruta_shap_kwargs in
    # test_fuzz_suite.py and gated to use_boruta_shap_cfg in canonical_key
    # so they never split dedup buckets when BorutaShap is off.
    #   * optimistic (default True): keep tentative features alongside
    #     accepted. False = strict Boruta semantics (fewer features kept) ->
    #     distinct selection-finalisation branch in boruta_shap.py.
    #   * train_or_test (default "train"): SHAP/permutation attributed on the
    #     training fold ("train") vs an internal held-out split ("test").
    #     "test" exercises BorutaShap's own train/test carve + the held-out
    #     importance path that "train" never reaches.
    #   * premerge_clusters (default False): collapse |corr|>=premerge_corr_thr
    #     columns to one representative BEFORE the shadow-importance test, then
    #     re-expand accepted reps. True activates the in-class correlation
    #     pre-merge path (distinct from the registry's GroupAwareMRMR wrap,
    #     which the FeatureSelectionConfig.boruta_shap_kwargs validator rejects
    #     -- see FS-coverage audit notes).
    "boruta_optimistic_cfg": (True, False),
    "boruta_train_or_test_cfg": ("train", "test"),
    "boruta_premerge_clusters_cfg": (False, True),
    # 2026-06-04 FS-coverage follow-up: BorutaShap margin-gated adaptive
    # trial-stop knobs (BorutaShap.__init__ early_stop_*). All forwarded into
    # boruta_shap_kwargs in test_fuzz_suite.py.
    #   * early_stop_tentative (default False): master toggle. True activates
    #     the residual-tentative-tail stop -- the trial loop ends before the
    #     n_trials cap once the accepted set is unchanged for
    #     early_stop_patience trials AND no still-tentative feature is within
    #     early_stop_margin of a binomial decision threshold. This exercises a
    #     distinct loop-termination branch in boruta_shap/_fit_explain that the
    #     fixed-cap default never reaches.
    #   * early_stop_patience (default 20): consecutive-unchanged trial count
    #     before the margin gate is evaluated. Only meaningful when
    #     early_stop_tentative=True; collapsed to 20 otherwise in canonical_key.
    #     5 makes the stop reachable inside the fuzz n_trials=10 budget.
    #   * early_stop_margin (default 0.15): relative slack on the binomial
    #     decision threshold. Only meaningful when early_stop_tentative=True;
    #     collapsed to 0.15 otherwise. 0.5 widens the "near a boundary" band so
    #     the stop is refused more often (the conservative branch).
    "boruta_early_stop_tentative_cfg": (False, True),
    "boruta_early_stop_patience_cfg": (20, 5),
    "boruta_early_stop_margin_cfg": (0.15, 0.5),
    # P1-8: FeatureSelectionConfig.use_sample_weights_in_fs. Weight-
    # aware FS (MRMR / RFECV fit with sample_weight). When True AND
    # weight_schemas includes non-uniform, FS refits per weight and the
    # cache invalidation logic kicks in. Untested pre-iter151.
    "use_sample_weights_in_fs_cfg": (False, True),
    # P1-9: PreprocessingBackendConfig.fallback_to_sklearn. True
    # (default) enables polars-ds -> sklearn fallback bridge when a
    # requested op is missing. False disables (forces error). Polars-ds
    # fallback path never exercised pre-iter151.
    "fallback_to_sklearn_cfg": (True, False),
    # P1-10a/b: TrainingBehaviorConfig device-selection toggles. CPU is
    # hardcoded in fuzz hyperparams; the per-model GPU/CPU dispatch
    # branches in compute_*_general_classif_params went unfuzzed.
    "prefer_gpu_configs_cfg": (True, False),
    "prefer_cpu_for_lightgbm_cfg": (True, False),
    # P2-16: FeatureSelectionConfig.mrmr_identity_cache_scope. "ctx"
    # (default, per-context cache) or "process" (process-level cache,
    # exercises inter-suite cache pollution / dedup / invalidation).
    "mrmr_identity_cache_scope_cfg": ("ctx", "process"),
    # P2-17: FeatureSelectionConfig.skip_identity_equivalent_pre_pipelines.
    # True (default, dedup-skip identity pipelines) or False (force
    # re-train for ensemble diversity).
    "skip_identity_equivalent_pre_pipelines_cfg": (True, False),
    # P2-18a: rfecv_leakage_corr_threshold. 0.95 default, 0.80
    # exercises the aggressive per-fit leakage filter.
    "rfecv_leakage_corr_threshold_cfg": (0.95, 0.80),
    # P2-18b: rfecv_mbh_adaptive_threshold. 30 default (CB surrogate
    # crossover), 100 forces ExtraTreesRegressor surrogate longer.
    "rfecv_mbh_adaptive_threshold_cfg": (30, 100),
    # 2026-06-03 FS-coverage audit: RFECV.__init__ knobs forwarded through the
    # suite's rfecv_kwargs (-> COMMON_RFECV_PARAMS update -> RFECV ctor). Both
    # are RFECV constructor params so FeatureSelectionConfig.rfecv_kwargs
    # validation accepts them; both are str-Enums so the string form works.
    # Gated to rfecv_estimator_cfg is not None in canonical_key (only meaningful
    # when an RFECV selector is actually in the pre-pipeline chain).
    #   * votes_aggregation_method (default Borda): how per-fold feature votes
    #     combine into the final ranking. "Plurality" / "Copeland" exercise
    #     distinct aggregators in wrappers/_rfecv.py (different selected sets on
    #     correlated features). Values validated against VotesAggregation enum
    #     in wrappers/_enums.py.
    #   * top_predictors_search_method (default ModelBasedHeuristic): the
    #     feature-count optimiser. "ExhaustiveDichotomic" replaces the surrogate
    #     model with a binary search over feature counts -- a different and
    #     faster (fuzz-budget-friendly) code path the heuristic never reaches.
    #     Values validated against OptimumSearch enum in wrappers/_enums.py.
    "rfecv_votes_aggregation_cfg": ("Borda", "Plurality", "Copeland"),
    "rfecv_search_method_cfg": ("ModelBasedHeuristic", "ExhaustiveDichotomic"),
    # =====================================================================
    # 2026-05-22 iter162 -- nested-config / depth-2 audit fill-in. 28 fields
    # surfaced by the second-pass explore agent; each is a config FIELD
    # (not a top-level kwarg) buried inside *_config object with material
    # behavioural effect and zero pre-iter162 fuzz coverage.
    # =====================================================================
    # --- FeatureHandlingConfig sub-configs (only meaningful when
    # enable_feature_handling_config_cfg=True; canon collapses otherwise)
    "fhc_cache_eviction_strategy_cfg": ("size_weighted", "lru", "lfu"),
    "fhc_cache_allow_pickle_cfg": (False, True),
    "fhc_cache_ram_fraction_cfg": (0.3, 0.5),
    "fhc_text_definite_text_mean_chars_cfg": (100, 50),
    "fhc_text_min_alphabet_entropy_cfg": (4.5, 3.0),
    "fhc_repro_deterministic_torch_cfg": (False, True),
    "fhc_auto_locale_detection_cfg": ("fallback_only", "off", "always"),
    # 2026-06-14 -- chart/report RENDERING path. Every reporting axis above tunes ReportingConfig FIELDS, but the fuzz runner hardcoded
    # show_perf_chart=False / show_fi=False / save_charts=False, so the actual matplotlib figure-generation code (perf chart, FI plot,
    # calibration/reliability panels, slice_finder, model_card, decision_curve, pdp_ice, shap_panels, model_comparison, risk_coverage, ...)
    # was NEVER invoked by any combo -- a large untested surface where iters 54-63 historically found perf bugs/hotspots. When True the runner
    # forces the matplotlib Agg backend (no display) + save_charts=True + show_perf_chart=True + show_fi=True so the rendering code executes and
    # is caught by the post-train invariants. The autouse _fuzz_combo_cleanup fixture already plt.close("all")s after every combo, so leaked
    # figures don't compound. Canonicalised to False on the large n_rows tier (rendering ~5 figs/combo × 200k-row scores is the memory blow-up
    # that motivated the original hardcoded-off) -- the small (1000-row) tier carries the rendering coverage cheaply.
    "enable_viz_rendering_cfg": (False, True),
    # --- ReportingConfig nested DSL / matplotlib
    "reporting_prob_histogram_yscale_cfg": ("auto", "log", "linear"),
    "reporting_title_metrics_template_cfg": (
        "ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC",
        "ICE ECE LL ROC_AUC",
    ),
    "reporting_matplotlib_rcparams_cfg": (None, '{"font.size":10}'),
    "reporting_multiclass_panels_cfg": (
        "CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC",
        "CONFUSION ROC",
    ),
    # --- ConfidenceAnalysisConfig model_kwargs dict
    "confidence_model_kwargs_cfg": ("default", "small_trees"),
    # --- CompositeTargetDiscoveryConfig nested
    "composite_mi_estimator_cfg": ("bin", "knn"),
    "composite_mi_nbins_cfg": (16, 8),
    "composite_mi_aggregation_cfg": ("mean", "sum"),
    "composite_mi_sample_strategy_cfg": ("random", "stratified_quantile"),
    "composite_stacked_residual_aggregation_cfg": ("mean", "first"),
    "composite_discovery_n_jobs_cfg": (1, 2),
    # --- QuantileRegressionConfig nested (only when enable_quantile=True)
    "quantile_crossing_fix_cfg": ("sort", "isotonic", "none"),
    "quantile_coverage_pairs_cfg": ("default", "wide"),
    "quantile_wrapper_n_jobs_cfg": ("auto", 1),
    # --- ModelHyperparamsConfig MLP predict batch size
    "mlp_predict_batch_size_cfg": (None, 512, 8192),
    # --- LearningToRankConfig nested (canon collapsed for non-LTR)
    "ltr_cb_loss_fn_cfg": ("YetiRankPairwise", "YetiRank", "QuerySoftMax"),
    "ltr_lgb_objective_cfg": ("lambdarank", "rank_xendcg"),
    "ltr_rrf_k_cfg": (60, 30),
    # --- RecurrentConfig nested (canon collapsed when recurrent_model=None)
    "recurrent_precision_cfg": ("32-true", "16-mixed"),
    "recurrent_sequence_preprocessing_cfg": ("none", "per_sequence_zscore"),
    # =====================================================================
    # 2026-05-22 iter170 -- 4-agent wave3 depth-3+ audit. ~60 fields
    # buried in per-backend hyperparam dicts + FHC sub-configs +
    # RFECV/MRMR/CatFE/Composite/Reporting/BaselineDiag deep fields.
    # =====================================================================
    # --- Per-backend hyperparam inner knobs (only on relevant model)
    "lgb_feature_fraction_cfg": (1.0, 0.7),
    "lgb_num_leaves_cfg": (31, 63),
    "xgb_max_depth_cfg": (6, 4),
    "xgb_colsample_bynode_cfg": (1.0, 0.7),
    "cb_border_count_cfg": (254, 64),
    "hgb_max_leaf_nodes_cfg": (31, 15),
    "rfecv_cv_n_splits_cfg": (2, 3),
    # --- PreprocessingBackendConfig + PreprocessingExtensionsConfig
    "robust_q_low_cfg": (0.01, 0.05),
    "robust_q_high_cfg": (0.99, 0.95),
    "tfidf_max_features_cfg": (5000, 1000),
    "kbins_encode_cfg": ("ordinal", "onehot"),
    "nonlinear_n_components_cfg": (100, 50),
    "pysr_operator_preset_cfg": ("standard", "minimal", "physics"),
    # --- TrainingBehaviorConfig + FeatureTypesConfig deep
    "confidence_ensemble_quantile_cfg": (0.1, 0.2),
    "cat_text_card_threshold_pct_cfg": (0.001, 0.01),
    # --- RFECV deep knobs
    # 2026-06-03: + "plateau" (e97b2417, parsimony rule) and "one_se_min"
    # (parsimonious 1-SE) so the fuzz exercises all 5 RFECV selection rules,
    # not just 3. All flow verbatim into RFECV.n_features_selection_rule
    # (validated against the 5-rule set in wrappers/_rfecv.py:425).
    "rfecv_n_features_selection_rule_cfg": ("auto", "argmax", "one_se_max", "one_se_min", "plateau"),
    "rfecv_stability_selection_cfg": (False, True),
    "rfecv_leakage_action_cfg": ("warn", "exclude"),
    # --- MRMR deep
    "mrmr_fe_adaptive_threshold_relax_cfg": (True, False),
    "mrmr_use_simple_mode_cfg": (False, True),
    "mrmr_identity_cache_include_y_cfg": (True, False),
    # 2026-05-27 -- MRMR friend-graph + cluster-aggregate features (added in
    # recent commits to mrmr.py). build_friend_graph builds a node-link
    # redundancy diagram + classifies features green/red/yellow (diagnostic
    # by default); friend_graph_prune drops red suspected-sink features from
    # support_ (CHANGES the selected set). cluster_aggregate_enable builds a
    # denoised aggregate of correlated "reflection" clusters; mode "augment"
    # adds the aggregate (keeps members), "replace" substitutes the cluster.
    # All gated to use_mrmr_fs in canonical_key; prune/mode further gated on
    # their parent toggle being on.
    "mrmr_build_friend_graph_cfg": (True, False),
    "mrmr_friend_graph_prune_cfg": (False, True),
    "mrmr_cluster_aggregate_enable_cfg": (True, False),
    "mrmr_cluster_aggregate_mode_cfg": ("augment", "replace"),
    # 2026-05-28 -- ShapProxiedFS (SHAP-coalition-proxy selector,
    # feature_selection/shap_proxied_fs.py, registry name "ShapProxiedFS").
    # Independent FS branch gated by its own enable flag (parallel to MRMR).
    # optimizer drives _resolve_optimizer/_run_search dispatch ("auto" picks
    # bruteforce<=22 else beam; "greedy_forward" hits the heuristic path);
    # revalidate toggles the honest disjoint-holdout retrain of the top-N;
    # trust_guard toggles the proxy-trust spearman diagnostic + the bias
    # corrector it feeds; interaction_aware adds the O(P^2) SHAP-interaction
    # coalition candidates; cluster_features ("auto"/False) toggles correlated
    # -feature clustering before SHAP. Sub-knobs collapse to ShapProxiedFS
    # __init__ defaults when use_shap_proxied_fs is off (canonical_key).
    "use_shap_proxied_fs": (False, True),
    "shap_proxied_optimizer_cfg": ("auto", "greedy_forward"),
    "shap_proxied_revalidate_cfg": (True, False),
    "shap_proxied_trust_guard_cfg": (True, False),
    "shap_proxied_interaction_aware_cfg": (False, True),
    "shap_proxied_cluster_features_cfg": ("auto", False),
    # 2026-05-28 -- ShapProxiedFS extension axes (active_learning + prefilter_method).
    # active_learning gates a separate acquisition loop (active_learning_revalidate)
    # that bypasses bruteforce/beam/greedy; prefilter_method drives the
    # _shap_proxy_prefilter dispatch ("auto" picks per HW; "univariate" is the
    # univariate-score ranking; "fast_model" is the cheap interaction-aware fast
    # booster). 2026-05-28 NOTE: manifest's suggested "mi"/"spearman" don't exist
    # in PREFILTER_METHODS = ("model","univariate","fast_model","gpu_model") --
    # we use real method names so ShapProxiedFS doesn't raise at runtime. Both
    # axes collapse to ShapProxiedFS __init__ defaults when use_shap_proxied_fs
    # is off (canonical_key).
    "shap_proxied_active_learning_cfg": (False, True),
    "shap_proxied_prefilter_method_cfg": ("auto", "univariate", "fast_model"),
    # 2026-05-28 -- ShapProxiedFS deeper extension axes (B1-B6 from audit pass 2).
    # All gate on use_shap_proxied_fs=True and collapse to ShapProxiedFS.__init__
    # defaults (verified against feature_selection/shap_proxied_fs.py:41-89).
    # config_jitter exercises the config-jitter SHAP branch (commit 20d8fa86);
    # uncertainty_penalty rewires the ranking objective (commit 20d8fa86);
    # within_cluster_refine False is the regression-comparison surface (commit
    # 8bd5b6d3); use_bias_corrector False disables the bias-corrected ranking
    # head (commit 0072b9f1); refine_n_estimators / trust_guard_n_estimators
    # None lifts the 100-tree booster cap added in commit bef1a9b4 (uncapped
    # full-depth booster vs. fast-fit cap branch).
    "shap_proxied_config_jitter_cfg": (False, True),
    "shap_proxied_uncertainty_penalty_cfg": (0.0, 0.5),
    "shap_proxied_within_cluster_refine_cfg": (True, False),
    "shap_proxied_use_bias_corrector_cfg": (True, False),
    "shap_proxied_refine_n_estimators_cfg": (100, None),
    "shap_proxied_trust_guard_n_estimators_cfg": (100, None),
    # 2026-05-28 -- ShapProxiedFS audit-pass-3 axes (W3). Defaults verified
    # against ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py:69-79).
    # cluster_weighting drives the per-cluster representative aggregation (pc1
    # vs. factor-score head); max_interaction_features sets the cap on the
    # interaction-tensor width used by the interaction_aware branch -- the 64
    # alternate is required because the default 16 cap may starve the wired
    # interaction_aware=True combos so the interaction tensor never fires;
    # prefilter_top caps the prefilter survivor set (None lifts the cap into
    # the full-feature regime); prefilter_n_estimators caps the prefilter
    # booster tree count (None disables the cap, matching legacy uncapped
    # behaviour).
    "shap_proxied_cluster_weighting_cfg": ("pca_pc1", "factor_score"),
    # iter624 (audit-pass-13 INFORMATIONAL): iter67 added two new
    # ShapProxiedFS ctor params for SU-pairwise clustering via MRMR
    # precomputed bins. Both default opt-IN (True / 0.5); the OFF
    # branch (cluster_use_precomputed_bins=False) is the legacy
    # Pearson-only fallback. Gate: use_shap_proxied_fs=True AND
    # cluster_features != False.
    # Verified prod defaults at shap_proxied_fs.py:228 (True) and :229
    # (0.5).
    "shap_proxied_cluster_use_precomputed_bins_cfg": (False, True),
    "shap_proxied_cluster_su_threshold_cfg": (0.3, 0.5),
    "shap_proxied_max_interaction_features_cfg": (16, 64),
    "shap_proxied_prefilter_top_cfg": (2000, None),
    "shap_proxied_prefilter_n_estimators_cfg": (100, None),
    # 2026-05-28 ShapProxiedFS audit-pass-5 axes (W5). 8 trust-guard / fidelity
    # axes verified against feature_selection/shap_proxied_fs.py:62, 78, 89-94.
    # All gate on use_shap_proxied_fs=True and canonicalise to ShapProxiedFS
    # __init__ defaults when off. trust_guard_stratified_anchors + uniform_tail_frac
    # further require trust_guard=True AND prefilter_method in ("two_stage",
    # "univariate") to materially activate; uniform_tail_frac additionally only
    # bites when stratified_anchors=True (anchor-weight presence). zipf_alpha
    # only meaningful when cardinality_dist=="zipf".
    "shap_proxied_trust_guard_stratified_anchors_cfg": (False, True),
    "shap_proxied_trust_guard_uniform_tail_frac_cfg": (0.2, 0.0),
    "shap_proxied_trust_guard_cardinality_dist_cfg": ("zipf", "uniform"),
    "shap_proxied_trust_guard_zipf_alpha_cfg": (0.25, 1.0),
    "shap_proxied_trust_guard_fidelity_weights_cfg": ((0.6, 0.4), (0.5, 0.5)),
    "shap_proxied_trust_guard_metric_cfg": ("proxy_fidelity_score", "spearman"),
    "shap_proxied_fidelity_floor_cfg": (0.5, 0.7),
    "shap_proxied_oof_shap_n_estimators_cfg": (100, None),
    # 2026-05-28 audit-pass-2 PART A: 4 LOW-tier coverage-gap axes deferred
    # from coverage_agent W11C wave.
    # ensembling_degenerate_class_ratio: EnsemblingConfig.degenerate_class_ratio
    # (default 0.01 at _model_configs.py:981); 0.05 widens the degenerate-subset
    # gate. Gated on use_ensembles AND classification target.
    "ensembling_degenerate_class_ratio_cfg": (0.01, 0.05),
    # target_temporal_audit_granularity: TrainingBehaviorConfig.target_temporal_audit_granularity
    # (default "auto" at _model_configs.py:561). Drives _phase_temporal_audit
    # bin freq. Gated on target_temporal_audit_column_cfg == 'ts_col'.
    "target_temporal_audit_granularity_cfg": ("auto", "day", "month"),
    # prep_ext_dim_n_components: PreprocessingExtensionsConfig.dim_n_components
    # (default 50 at _preprocessing_configs.py:340). 10 / 100 magnify the
    # dim-reducer code path already covered by prep_ext_dim_reducer_cfg.
    # Gated on prep_ext_dim_reducer_cfg in (PCA, TruncatedSVD).
    "prep_ext_dim_n_components_cfg": (50, 10, 100),
    # 2026-05-28 -- TextDetectionConfig.text_min_cardinality. Cardinality floor
    # for the cat-vs-text promotion. Default 300; 50 flips many short-string
    # cats into the text path (TF-IDF + cat_text_card_threshold interplay).
    # Gated on enable_feature_handling_config_cfg=True.
    "fhc_text_min_cardinality_cfg": (300, 50),
    # 2026-05-28 -- CompositeTargetDiscoveryConfig deep knobs.
    # auto_skip_on_baseline_optimal: short-circuit when baseline_diagnostics
    # composite_recommendation=='unlikely_to_help' (entirely separate branch).
    # mi_n_neighbors: kNN MI k -- only fires when mi_estimator='knn'.
    # auto_base_null_perms: permutation-MI null test budget (0 disables; 50 is heavy).
    # multi_base_max_k: forward-stepwise multi-base ceiling (1 disables promotion;
    # 5 quintuples per-kept-spec OLS refit count).
    "composite_auto_skip_on_baseline_optimal_cfg": (False, True),
    "composite_mi_n_neighbors_cfg": (3, 5, 10),
    "composite_auto_base_null_perms_cfg": (20, 0, 50),
    "composite_multi_base_max_k_cfg": (3, 1, 5),
    # 2026-05-28 -- TrainingBehaviorConfig.extreme_ar_group_aware_skip_models.
    # Which model families get skipped on extreme-AR + group-aware regimes.
    # "default_neural" = (mlp,ngb,lstm,gru,rnn,transformer); "include_linear"
    # adds linear; "empty" disables the skip entirely. Gated on
    # mlp_extreme_ar_group_aware_skip_cfg=True AND 'mlp' in combo.models.
    "extreme_ar_group_aware_skip_models_cfg": ("default_neural", "include_linear", "empty"),
    # 2026-05-28 -- FeatureSelectionConfig.pre_screen_null_fraction_threshold.
    # Sibling of fs_pre_screen_variance_threshold_cfg; 0.5 aggressively drops
    # half-null columns BEFORE MRMR/RFECV/BorutaShap. Gated on
    # fs_pre_screen_unsupervised_cfg=True.
    "fs_pre_screen_null_fraction_threshold_cfg": (0.99, 0.5),
    # 2026-05-28 -- LinearModelConfig.l1_ratio (ElasticNet mixing param).
    # 0.0=Ridge, 0.5=ElasticNet (default), 1.0=LASSO. Canonicalised to 0.0
    # when linear_solver_cfg != 'saga' (sklearn raises l1_ratio>0 with
    # lbfgs/liblinear). Gated on 'linear' in combo.models.
    "linear_l1_ratio_cfg": (0.5, 0.0, 1.0),
    # 2026-05-28 -- RecurrentConfig.hidden_size. RNN hidden-state width.
    # Library default 128; 32 is the cheap-fit variant. Canon to 128 when
    # recurrent_model_cfg is None.
    "recurrent_hidden_size_cfg": (128, 32),
    # --- CatFE deep (only when use_mrmr_fs + cat_fe_enable)
    "catfe_fwer_correction_cfg": ("none", "bh_fdr", "bonferroni"),
    "catfe_perm_budget_strategy_cfg": ("bandit_ucb1", "fixed"),
    "catfe_permutation_null_cfg": ("joint_independence", "conditional"),
    "catfe_bootstrap_ci_n_replicates_cfg": (0, 50),
    "catfe_use_miller_madow_cfg": (None, True, False),
    "catfe_refine_passes_cfg": (0, 1),
    "catfe_enable_streaming_cache_cfg": (False, True),
    "catfe_unknown_strategy_cfg": ("clip", "sentinel", "raise"),
    # --- Composite discovery deep
    "composite_screening_cfg": ("hybrid", "mi", "tiny_model"),
    "composite_tiny_model_num_leaves_cfg": (15, 31),
    "composite_tiny_model_learning_rate_cfg": (0.1, 0.05),
    "composite_raw_baseline_tolerance_cfg": (1.02, 1.10),
    "composite_use_wilcoxon_gate_cfg": (False, True),
    "composite_detect_alpha_drift_cfg": (True, False),
    "composite_reject_on_alpha_drift_cfg": (False, True),
    # --- Reporting deep
    "reporting_figsize_cfg": ("default", "small"),
    "reporting_plot_dpi_cfg": (None, 80),
    "reporting_quantile_panels_cfg": ("default", "minimal"),
    "reporting_ltr_panels_cfg": ("default", "minimal"),
    "reporting_plotly_template_cfg": (None, "ggplot2"),
    "reporting_matplotlib_style_cfg": (None, "ggplot"),
    # --- BaselineDiagnostics deep
    # 2026-06-04 profiling-budget: cap the baseline quick-model size to 5 (was 200/50) so the
    # baseline-diagnostics floor stays cheap at n=100k. It's a quick-floor model, not a tuned one.
    "baseline_quick_model_n_estimators_cfg": (5,),
    "baseline_quick_model_num_leaves_cfg": (31, 15),
    "baseline_quick_model_learning_rate_cfg": (0.05, 0.1),
    "baseline_sample_n_cfg": (50_000, 10_000),
    "baseline_high_potential_min_dominance_pct_cfg": (5.0, 10.0),
    "baseline_best_model_min_lift_cfg": (1.5, 2.0),
    # --- DummyBaselines deep
    "dummy_stratified_n_repeats_cfg": (20, 5),
    "dummy_paired_bootstrap_n_resamples_cfg": (1000, 200),
    # --- LTR deep
    "ltr_mlp_loss_fn_cfg": ("ranknet", "listnet"),
    "ltr_eval_at_cfg": ("default", "extended"),
    # --- Multilabel deep
    "multilabel_force_native_xgb_cfg": (False, True),
    # --- FHC pricing / logging / repro / cache deep (only when enable_fhc)
    "fhc_pricing_cap_usd_cfg": (None, 1.0),
    "fhc_pricing_warn_above_usd_cfg": (1.0, 0.5),
    "fhc_logging_verbose_cfg": (False, True),
    "fhc_repro_langdetect_seed_cfg": (0, 42),
    "fhc_repro_pinned_svd_solver_params_cfg": (True, False),
    "fhc_repro_forbid_nonatomic_fs_cfg": (False, True),
    "fhc_repro_deterministic_eviction_cfg": (False, True),
    "fhc_cache_prefetch_enabled_cfg": (True, False),
    "fhc_cache_prefetch_vram_safety_factor_cfg": (2.0, 1.5),
    "fhc_memory_pressure_watermark_pct_cfg": (85, 75),
    "fhc_text_min_mean_tokens_cfg": (4.0, 2.0),
    "fhc_text_min_unique_ratio_cfg": (0.95, 0.85),
    "fhc_text_respect_explicit_cat_dtype_cfg": (True, False),
    # --- RecurrentConfig deep (only when recurrent_model)
    "recurrent_input_mode_cfg": ("hybrid", "sequence_only", "tabular_only"),
    "recurrent_num_workers_cfg": (0, 2),
    # =====================================================================
    # 2026-05-23 iter180 -- DEPTH-4 booster sub-params + FHC persistence
    # + multilabel list-typed depth-3 fields. Booster axes work as pairs:
    # the boosting_type/tree_method/bootstrap_type/grow_policy gate
    # (depth-3) UNLOCKS the depth-4 sub-knob. Sub-knob axes default to
    # the library default so they're no-ops unless the gate is active.
    # =====================================================================
    # LGB boosting_type + depth-4 sub-params
    "lgb_boosting_type_cfg": ("gbdt", "dart", "goss"),
    "lgb_dart_drop_rate_cfg": (0.1, 0.3),         # only when boosting_type='dart'
    "lgb_goss_top_rate_cfg": (0.2, 0.4),          # only when boosting_type='goss'
    # XGB tree_method + depth-4 sub-params
    "xgb_tree_method_cfg": ("auto", "hist"),
    "xgb_hist_max_bin_cfg": (256, 64),            # only when tree_method='hist'
    # CB bootstrap_type + grow_policy + depth-4 sub-params
    "cb_bootstrap_type_cfg": ("Bayesian", "Bernoulli", "MVS"),
    "cb_bayesian_bagging_temperature_cfg": (1.0, 5.0),  # only when bootstrap_type='Bayesian'
    "cb_bernoulli_subsample_cfg": (0.8, 0.5),     # only when bootstrap_type='Bernoulli'
    "cb_grow_policy_cfg": ("SymmetricTree", "Lossguide"),
    "cb_lossguide_max_leaves_cfg": (31, 63),      # only when grow_policy='Lossguide'
    # FHC.cache.persistence (depth-4 because it gates disk-tier sub-fields)
    "fhc_cache_persistence_cfg": ("auto", "off", "read_write"),
    # MultilabelDispatchConfig depth-4 list-typed fields (never set pre-iter180)
    "multilabel_per_label_thresholds_cfg": (None, "uniform_0.4"),
    "multilabel_chain_seeds_cfg": (None, "explicit"),
    # F1 (fuzz_blind_spots_F1_F2_F5_F6_F7) -- enable_crash_reporting is a suite-level kwarg consumed directly by train_mlframe_models_suite. Canonicalises to False on non-Windows hosts (crash_reporting is a Windows-specific Faulthandler dump hook there).
    "enable_crash_reporting_cfg": (False, True),
    # =====================================================================
    # 2026-05-26 iter291 -- new-functionality coverage (258 commits over the
    # last 2 days landed substantive new features; the axes below pull each
    # config-flag-flippable surface into fuzz space so the new code paths
    # get cross-axis interaction coverage.
    # =====================================================================
    # TrainingSplitConfig.bucket_stratify (default flipped True 2026-05-25):
    # regression-target stratified split into quantile buckets. False
    # exercises the legacy random-split path that the new default replaced.
    # Canonicalised to True for non-regression primaries (no-op gate).
    "bucket_stratify_cfg": (True, False),
    # TrainingSplitConfig.composite_cardinality_cap (default 200): caps the
    # cardinality of group keys / strata used by the split. Pinning to a
    # tight value (50) forces the cap to actually bite on combos with
    # high-cardinality cat columns; the default (200) leaves headroom.
    "composite_cardinality_cap_cfg": (200, 50),
    # ReportingConfig.honest_estimator_diagnostics (default True 2026-05-25):
    # post-fit honest-diagnostics aggregator (target-distribution mismatch,
    # train-val drift, leakage probe summary). False skips the entire pass
    # so the aggregator overhead is fuzzed alongside its enabled-default.
    "honest_estimator_diagnostics_cfg": (True, False),
    # CompositeTargetDiscoveryConfig.cross_target_ensemble_strategy
    # (default "nnls_stack" per _composite_target_discovery_config.py:641):
    # how the per-composite-component predictions are combined. Untested
    # combinations through fuzz: "mean" (uniform avg), "oof_weighted"
    # (gain-over-baseline weighting), "linear_stack" (OLS). Canon collapses
    # to "nnls_stack" when composite_discovery_enabled_cfg is False.
    "cross_target_ensemble_strategy_cfg": (
        "nnls_stack", "mean", "oof_weighted", "linear_stack",
    ),
    # feature_engineering.basic.add_cyclical_date_features: 2026-05-25 added
    # Kaggle-style cyclical sin/cos transforms (period 7/12/24/31/365). When
    # True the runner threads ``add_cyclical_date_features=True`` so the
    # date-decomposition produces 2K cols (sin+cos per period) on top of the
    # legacy day/weekday/month integer cols. Canon collapses to False when
    # with_datetime_col is False (no datetime column to transform).
    "add_cyclical_date_features_cfg": (False, True),
    # feature_engineering.basic extended-date features (Kaggle-style: quarter,
    # day_of_year, week_of_year, is_weekend, is_month_start, etc.). Same
    # canonicalisation as add_cyclical_date_features_cfg.
    "add_extended_date_features_cfg": (False, True),
    # NNLS stacking-aware blend (AP7+AP8 2026-05-25): when False the simple-
    # blend path uses uniform weights; when True (default) NNLS weights from
    # the stacking surrogate drive the simple-blend mix too. Untested False
    # path matters because it's the fallback when NNLS solver fails. Canon
    # collapses to True when use_ensembles is False.
    "use_nnls_weights_in_blends_cfg": (True, False),
    # Generic prediction-envelope clip phase (4e579b4d 2026-05-26): controls
    # the ``MLFRAME_DISABLE_PREDICTION_ENVELOPE_CLIP`` env var. Default ON
    # (env unset). False sets env=1 for the suite run, exercising the
    # unclamped path so future regression-collapse incidents surface in
    # fuzz. Canon collapses to True for non-regression primaries (clip is
    # regression-only). When clip is False, the suite is allowed to emit
    # extreme out-of-envelope predictions; tests must still pass.
    "enable_prediction_envelope_clip_cfg": (True, False),
    # =====================================================================
    # 2026-05-27 iter332 -- audit-driven new-functionality coverage. The
    # explore-agent audit identified ~30 high-value config fields with new
    # code paths but no fuzz exposure. Top-15 most impactful axes below;
    # remaining 15 (and ~35 medium-value) defer until budget permits.
    # =====================================================================
    # EnsemblingConfig (entire class previously unfuzzed -- no
    # ``ensembling_config`` kwarg on suite, knobs were env-var-only).
    # force_legacy toggles the pre-streaming materialised-aggregation path
    # (allocates (M, N, K) tensors); streaming Welford is the new default.
    # Canon to False when use_ensembles is off (no ensemble path runs).
    "ensembling_force_legacy_cfg": (False, True),
    # quantile_budget_bytes: skip quantile-bucket aggregation when
    # M*N*K*8 > budget. 500 MB default; tight budget (10 MB) forces the
    # fallback-with-warn branch. Canon to default when ensembling is off.
    "ensembling_quantile_budget_bytes_cfg": (500 * 1024 * 1024, 10 * 1024 * 1024),
    # flag_degenerate_conf_subset: prepend [DEGENERATE] marker when
    # confidence-filtered subset's class balance collapses; binary-only.
    "ensembling_flag_degenerate_conf_subset_cfg": (True, False),
    # MLP extreme-AR + group-aware protection cluster (2026-05-26 fix wave).
    # ``mlp_extreme_ar_group_aware_skip``: skip MLP entirely when
    # extreme-AR target detected on group-aware split. False forces MLP
    # to fit through; True is the new default safety gate.
    # Canon to default when "mlp" not in models.
    "mlp_extreme_ar_group_aware_skip_cfg": (False, True),
    # ``mlp_extreme_ar_threshold``: AR autocorrelation threshold above which
    # the skip+weight_decay-bump phases fire. (0.99, 0.50) widens the bite.
    "mlp_extreme_ar_threshold_cfg": (0.99, 0.50),
    # ``mlp_drop_per_group_constants``: drop per-group aggregate features
    # before MLP fit. Brand new branch; True exercises the drop sweep.
    "mlp_drop_per_group_constants_cfg": (False, True),
    # CompositeTargetDiscoveryConfig new flags (2026-05-25/26).
    # always_build_ct_ensemble_for_raw: True default ensures CT_ENSEMBLE
    # gate runs even on raw-only targets; False = legacy regression path.
    "composite_always_build_ct_ensemble_for_raw_cfg": (True, False),
    # ct_ensemble_dummy_floor_enabled: drop ensemble components below
    # strongest-dummy RMSE. False = legacy unfiltered pool.
    "composite_ct_ensemble_dummy_floor_enabled_cfg": (True, False),
    # extreme_ar_group_aware_skip on composite discovery side: short-circuit
    # when AR+group-aware combo would be unstable.
    "composite_extreme_ar_group_aware_skip_cfg": (True, False),
    # oof_holdout_source: how the OOF holdout is carved. ``train_tail`` is
    # the pre-2026-05-25 default; ``external_val`` is the new default that
    # closes a stacking-leak path.
    "composite_oof_holdout_source_cfg": ("external_val", "train_tail"),
    # stacking_aware_gate_enabled: NEW measure-first NNLS gate that
    # validates the ensemble plan before committing CV resources.
    "composite_stacking_aware_gate_enabled_cfg": (False, True),
    # use_baseline_diagnostics_hint: NEW hint-injection path that reuses
    # BD's quick-model verdicts to seed composite scoring.
    "composite_use_baseline_diagnostics_hint_cfg": (True, False),
    # FeatureSelectionConfig pre-screen hardening flags.
    # pre_screen_unsupervised: gates the unsupervised variance / null-fraction
    # column-drop sweep that runs before the main FS estimator.
    "fs_pre_screen_unsupervised_cfg": (True, False),
    # pre_screen_variance_threshold: drop columns with variance below this.
    # 0.0 = no drop; 0.01 exercises the drop branch on near-constant cols.
    "fs_pre_screen_variance_threshold_cfg": (0.0, 0.01),
    # BaselineDiagnosticsConfig.init_score_top_k: how many top BD models
    # contribute init scores for downstream boosters. K=1 is the
    # one-winner branch; K>=2 exercises the OLS-combined branch.
    "baseline_init_score_top_k_cfg": (1, 2),
    # =====================================================================
    # 2026-05-27 iter350 -- audit batch 2. Remaining ~12 high-value axes
    # from the audit-agent report.
    # =====================================================================
    # TrainingBehaviorConfig.use_ap12_calibrated_probs_in_ensemble (default
    # True 2026-05-25): route AP12-calibrated probs into simple blends.
    # False = legacy uncalibrated path. Canon to default when ensembling off.
    "use_ap12_calibrated_probs_in_ensemble_cfg": (True, False),
    # TrainingBehaviorConfig.mlp_extreme_ar_weight_decay_factor: when the
    # extreme-AR + group-aware gate fires, multiply weight_decay by this
    # factor to dampen the MLP into a safe regime. 100x is the default;
    # 1.0 = no bump (legacy behaviour pre-2026-05-26). Canon to default
    # when MLP not in scope OR the AR-skip gate is OFF (already-skipped).
    "mlp_extreme_ar_weight_decay_factor_cfg": (100.0, 1.0),
    # TrainingBehaviorConfig.feature_drift_auto_apply_neural_overrides:
    # 2026-05-26 brand-new override-auto-apply gate. True path entirely
    # uncovered; toggles whether the drift report's neural recommendations
    # get auto-applied to subsequent fits.
    "feature_drift_auto_apply_neural_overrides_cfg": (False, True),
    # TrainingBehaviorConfig.target_temporal_audit_column: None = phase
    # disabled (default); "ts_col" exercises the temporal-audit phase
    # (regime-divergence chart + warn) when the synthetic combo emits a
    # datetime column. Canon to None when with_datetime_col is False.
    "target_temporal_audit_column_cfg": (None, "ts_col"),
    # CompositeTargetDiscoveryConfig.lag_predict_failsafe_tolerance: how
    # far above the lag-predict floor the discovery survivors must stay.
    # (0.10, 0.50) covers the 2026-05-25 calibration regime + the pre-fix
    # loose tolerance that prod logs showed was too permissive.
    "composite_lag_predict_failsafe_tolerance_cfg": (0.10, 0.50),
    # CompositeTargetDiscoveryConfig.extreme_ar_threshold: AR detection
    # threshold for the composite-side extreme-AR skip gate (separate from
    # the MLP-side gate). (0.99, 0.95) widens the gate's bite.
    "composite_extreme_ar_threshold_cfg": (0.99, 0.95),
    # CompositeTargetDiscoveryConfig.ct_ensemble_dummy_floor_tolerance:
    # slack above the strongest-dummy RMSE that components must beat to
    # survive the floor. 0.0 = strict floor; 0.10 = 10pct slack.
    "composite_ct_ensemble_dummy_floor_tolerance_cfg": (0.0, 0.10),
    # CompositeTargetDiscoveryConfig.oof_holdout_frac: fraction of train
    # carved out as OOF holdout for stacking. (0.2, 0.0) tests the
    # pre-2026-05-25 (no-holdout, leak-risk) regime against the new
    # default.
    "composite_oof_holdout_frac_cfg": (0.2, 0.0),
    # CompositeTargetDiscoveryConfig.top_m_after_tiny: how many discovery
    # specs survive the tiny-model screening pass. Bumped 3 -> 10 on
    # 2026-05-25; (10, 3) tests both regimes against the ensemble.
    "composite_top_m_after_tiny_cfg": (10, 3),
    # PreprocessingExtensionsConfig.tfidf_keep_sparse: True (default)
    # keeps the per-column TF-IDF output as scipy.sparse; False forces
    # the legacy ``.toarray()`` path. False blows up RAM but matters
    # for downstream consumers that can't handle sparse.
    "prep_ext_tfidf_keep_sparse_cfg": (True, False),
    # RecurrentConfig.use_attention: True (default) uses attention pooling
    # vs last-hidden (False); flips encoder semantics. Canon collapsed
    # when recurrent_model_cfg is None.
    "recurrent_use_attention_cfg": (True, False),
    # LearningToRankConfig.xgb_objective: "rank:ndcg" (default) vs
    # "rank:map"; flips XGBoost ranking head + autodetect fallback path
    # for y.max() > 1. Canon to default when target_type != LTR.
    "ltr_xgb_objective_cfg": ("rank:ndcg", "rank:map"),
    # BaselineDiagnosticsConfig.init_score_apply_to_target_types: which
    # target families get init-score injection. Default tuple includes
    # regression; toggling to broaden / narrow exercises the logit-init
    # branch for binary.
    "baseline_init_score_apply_target_types_cfg": (
        "regression_only", "regression_and_binary",
    ),
    # =====================================================================
    # 2026-05-28 audit-pass-4 SAFE-subset (W4): 8 axes pulled from
    # D:/Temp/AUDIT_PASS_4_DONE.md (slice_stable_es_* family deferred to a
    # separate batch pending the SliceStableESConfig refactor).
    # All defaults source-verified -- drift corrections noted in
    # FUZZ_AXES_W4_IMPL_DONE.json. Canon-only (no downstream consumer
    # wiring) following the ensembling_degenerate_class_ratio_cfg pattern
    # from the W2 batch (48cc7b7e).
    # =====================================================================
    # CalibrationConfig.policy_auto_pick (default True at
    # src/mlframe/calibration/policy.py:464). Auto-pick the best calibrator
    # by OOF ECE with bootstrap CI tiebreak. Gate: target_type in
    # {binary_classification, multilabel_classification}.
    "calibration_policy_auto_pick_cfg": (True, False),
    # CalibrationConfig.n_bootstrap (default 1000 at policy.py:467 via
    # DEFAULT_N_BOOTSTRAP=1000). Bootstrap resample count for the OOF
    # ECE CI. 100 exercises the fast-tiebreak branch with wider CIs.
    # Gate: classification target AND calibration_policy_auto_pick_cfg=True.
    "calibration_n_bootstrap_cfg": (1000, 100),
    # CalibrationConfig.candidates (default None at policy.py:469 -- None
    # expands to the full CANDIDATE_NAMES=("Sigmoid","Isotonic","Beta","Spline")
    # palette). Restricted ("Sigmoid","Isotonic") drops the optional-dep
    # Beta+Spline branch entirely (faster + reproducible on hosts without
    # betacal / ml_insights). Gate: classification target AND
    # calibration_policy_auto_pick_cfg=True.
    "calibration_candidates_cfg": (None, ("Sigmoid", "Isotonic")),
    # TrainingBehaviorConfig.pipeline_cache_ram_budget_fraction (default 0.4
    # at _model_configs.py:641). Fraction of TOTAL host RAM used as the
    # PipelineCache byte budget. 0.1 exercises the heavy-pressure path
    # where the cache evicts more aggressively. Gate: any (the cache is
    # always live in the suite).
    "pipeline_cache_ram_budget_fraction_cfg": (0.4, 0.1),
    # ReportingConfig.compute_trainset_metrics (default False at
    # _reporting_configs.py:96). True exercises the per-split train-set
    # metric computation path (otherwise the train-set metrics block is
    # silently skipped). Gate: any (every suite emits a report).
    "reporting_compute_trainset_metrics_cfg": (False, True),
    # ReportingConfig.mase_seasonality (default 1 at _reporting_configs.py:140;
    # integer NOT None). Hyndman-Koehler 2006 MASE seasonality used for
    # naive-MAE scaling; 12 exercises the monthly->yearly seasonal path
    # vs. the default simple-naive (lag-1). Gate: target_type=="regression"
    # (MASE is a regression-only metric).
    "reporting_mase_seasonality_cfg": (1, 12),
    # RecurrentConfig.use_stratified_sampler (default True at
    # src/mlframe/training/neural/_recurrent_config.py:90). Weighted
    # sampling for imbalanced data; False exercises the uniform-sampler
    # path. Gate: recurrent_model_cfg in ("lstm","gru","transformer","rnn")
    # AND target_type in classification (stratified sampler is
    # classification-only).
    "recurrent_use_stratified_sampler_cfg": (True, False),
    # TrainingBehaviorConfig.model_file_hash_suffix (default True at
    # _model_configs.py:547; bool NOT str|None). Append a per-model
    # input-schema fingerprint to model filenames. False restores the
    # pre-2026-04-21 naming scheme. Gate: any (the hash-suffix decision
    # fires on every model save).
    "behavior_model_file_hash_suffix_cfg": (True, False),
    # =====================================================================
    # 2026-05-30 audit-pass-6 (W6) -- 15 axes from commits landed since
    # fcc47d04 wave 5. All defaults SOURCE-verified at this commit:
    #   - SliceStableESConfig at src/mlframe/training/_training_runtime_configs.py:42-95
    #   - MRMR Wave 7/8/9 ctor args at filters/mrmr.py:224-302, 589
    #   - CompositeTargetDiscoveryConfig.cv_selector_mode at
    #     _composite_target_discovery_config.py:117
    # Drift notes: audit said slice_stable_es source default "random" and
    # aggregate default "t_lcb"; SOURCE says source="temporal" and
    # aggregate="mean". Pairs use the SOURCE default first.
    # =====================================================================
    # Slice-stable ES master toggle (HIGH; entire algorithm branch).
    "slice_stable_es_enabled_cfg": (False, True),
    # Slice-stable ES aggregator: pure-mean (legacy bit-identical) vs t-LCB
    # parametric lower-confidence-bound. Source default "mean". Source-verified
    # at _training_runtime_configs.py:89.
    "slice_stable_es_aggregate_cfg": ("mean", "t_lcb"),
    # Slice-stable ES shard source. Source default "temporal" (verified
    # at :78); "random" exercises the random-shard branch. Two of the
    # four shard-builder branches (random / temporal / fairness / both).
    "slice_stable_es_source_cfg": ("temporal", "random"),
    # Slice-stable ES Pareto-aware best_iter post-hoc selection
    # (_training_runtime_configs.py:98). Default False.
    "slice_stable_es_pareto_best_iter_selection_cfg": (False, True),
    # Slice-stable ES diagnostic-only path: register K eval-sets + log
    # trace WITHOUT changing stop decisions. _training_runtime_configs.py:76.
    "slice_stable_es_diagnostic_only_cfg": (False, True),
    # MRMR Wave 7 -- per-feature discretisation strategy. Source default
    # "mdlp" (Fayyad-Irani per-feature MDLP), alternative "quantile"
    # restores the pre-2026-05-29 fixed quantile binning behaviour.
    # filters/mrmr.py:224.
    "mrmr_nbins_strategy_cfg": ("mdlp", "quantile"),
    # MRMR Wave 8 F13 -- Chao-Shen entropy bias correction. filters/mrmr.py:229.
    # 3 algorithmic branches; pair exercises default vs CS.
    "mrmr_mi_correction_cfg": ("none", "chao_shen"),
    # MRMR Wave 8 A1 -- JMIM redundancy aggregator (Bennasar 2015) vs
    # Fleuret CMIM (None = legacy). filters/mrmr.py:234.
    "mrmr_redundancy_aggregator_cfg": (None, "jmim"),
    # MRMR Wave 8 A3 -- BUR unique-relevance bonus (Gao 2022). 0.0 = off,
    # 0.5 activates the additive bonus path. filters/mrmr.py:238.
    "mrmr_bur_lambda_cfg": (0.0, 0.5),
    # MRMR Wave 8 C8 -- permutation-null stopping criterion (Yu-Principe
    # 2019). Replaces the threshold gate with a perm-null test.
    # filters/mrmr.py:244.
    "mrmr_cmi_perm_stop_cfg": (False, True),
    # MRMR Wave 8 E11/E12 -- Cluster Stability Selection (Faletto-Bien
    # 2022). filters/mrmr.py:259. Default "classic" Meinshausen-Buhlmann.
    "mrmr_stability_selection_method_cfg": ("classic", "cluster"),
    # MRMR commit 4840bbe7 -- Symmetric Uncertainty (Witten-Frank-Hall
    # 2011) replaces raw MI to remove cardinality bias. filters/mrmr.py:276.
    "mrmr_mi_normalization_cfg": ("none", "su"),
    # MRMR Wave 9 -- Dynamic Cluster Discovery master enable flag.
    # Organic in-greedy-loop cluster discovery via MI/SU distance.
    # filters/mrmr.py:589.
    "mrmr_dcd_enable_cfg": (False, True),
    # 2026-05-30 audit-pass-7 #2: MRMR.baseline_npermutations
    # (mrmr.py:309, default 2). After evaluation.py commit b0e0ea4f the
    # baseline gate uses max_failed=_bnp with min_nonzero_confidence=0.0,
    # turning baseline_npermutations into the "unanimity-quorum" size for
    # rejecting candidates. pair_2 = default permissive regime;
    # pair_8 = strict regime where genuine XOR is borderline.
    "mrmr_baseline_npermutations_cfg": (2, 8),
    # 2026-05-30 audit-pass-7 #3: per_feature_edges low_card_cap kwarg
    # (_adaptive_nbins.py:511, default 32). Threaded into MRMR via
    # nbins_strategy_kwargs={"low_card_cap": ...}. pair_2 disables the
    # midpoint-bypass for all but binary columns (every ordinal/small-
    # categorical column then runs the configured method_resolved);
    # pair_32 keeps the default bypass.
    "mrmr_low_card_cap_cfg": (2, 32),
    # 2026-05-30 audit-pass-7 #4: per_feature_edges collapsed_fallback_nbins
    # kwarg (_adaptive_nbins.py:586, default 5). Only fires when the
    # supervised binning method (mdlp / fayyad_irani / optimal_joint / mah)
    # collapses a column to a single bin -- then the fallback uses this
    # nbins for an unsupervised re-bin so synergy MI on tuples containing
    # the collapsed column is not identically zero. pair_3 stresses sparser
    # fallback joint cardinality; pair_10 stresses denser.
    "mrmr_collapsed_fallback_nbins_cfg": (3, 10),
    # CV-selector mode (HIGH; mean vs Student-t LCB). Gate:
    # composite_discovery_enabled_cfg=True. Source default "mean".
    # _composite_target_discovery_config.py:117.
    "cv_selector_mode_cfg": ("mean", "t_lcb"),
    # TrainingBehaviorConfig.auto_wrap_partial_fit_es (S27 close-out).
    # Now a real ctor param at _model_configs.py (default True). True forces
    # OFF the PartialFitESWrapper auto-wrap at
    # _trainer_train_and_evaluate.py:551; pair = (False=default-leave-on,
    # True=force-off). Wired via inversion:
    # auto_wrap_partial_fit_es = not auto_wrap_partial_fit_es_force_off_cfg.
    "auto_wrap_partial_fit_es_force_off_cfg": (False, True),
    # =====================================================================
    # 2026-05-30 audit-pass-6 LOW-tier deferred batch (W6 LOW). 28 axes
    # from D:/Temp/AUDIT_PASS_6_DONE.md (S27 now landed above the divider as
    # a real ctor param at TrainingBehaviorConfig.auto_wrap_partial_fit_es).
    # Defaults SOURCE-verified at HEAD:
    #   - ShapProxiedFS Stage-A/Refine/Reval/Threading (S1-S18) against
    #     src/mlframe/feature_selection/shap_proxied_fs.py:41-117.
    #   - Curve-shape ES coeff/min_iters (S25/S26) against
    #     src/mlframe/training/_model_configs.py:506-507.
    #   - MRMR Wave 8 A2/C9/D10/F14 (S32/S34/S35/S37) against
    #     src/mlframe/feature_selection/filters/mrmr.py:241,249,252,265.
    #   - CV-selector alpha/confidence/quantile/persist_fold (S41-S44)
    #     against _composite_target_discovery_config.py:127-130.
    # =====================================================================
    # ShapProxiedFS Stage-A prefilter knobs (S1-S8). All gate on
    # use_shap_proxied_fs=True. Flow through
    # build_shap_proxied_fs_kwargs_from_flat -> ShapProxiedFS.__init__.
    # Pairs verified against shap_proxied_fs.py:79-87 source defaults.
    "shap_proxied_prefilter_stage1_keep_cfg": (None, 200),
    "shap_proxied_prefilter_univariate_batch_size_cfg": (None, 256),
    "shap_proxied_shap_prefilter_enabled_cfg": (True, False),
    "shap_proxied_shap_prefilter_safety_factor_cfg": (4, 8),
    "shap_proxied_shap_prefilter_min_features_cfg": (40, 80),
    "shap_proxied_shap_aware_stage1_keep_cfg": (True, False),
    # 2026-05-31 audit-pass-14 F14-2: extended (2, 4) -> (2, 4, 8). Source
    # default flipped 8 -> 2 in iter76 (shap_proxied_fs.py:249); the legacy
    # cushion=8 branch is now untested in default config, so we keep the
    # third value pinned for fuzz coverage of the pre-iter76 calibration.
    "shap_proxied_shap_aware_stage1_cushion_cfg": (2, 4, 8),
    "shap_proxied_shap_aware_stage1_floor_cfg": (200, 500),
    # ShapProxiedFS Refine UCB knobs (S9-S12). Gate on
    # use_shap_proxied_fs=True AND shap_proxied_within_cluster_refine_cfg=True.
    # Source defaults at shap_proxied_fs.py:96-99.
    "shap_proxied_refine_ucb_enabled_cfg": (True, False),
    "shap_proxied_refine_ucb_min_eval_size_cfg": (None, 8),
    "shap_proxied_refine_ucb_slack_cfg": (None, 0.0),
    "shap_proxied_refine_ucb_stdev_multiplier_cfg": (1.0, 0.5),
    # ShapProxiedFS Revalidation knobs (S13-S17). Gate on
    # use_shap_proxied_fs=True AND shap_proxied_revalidate_cfg=True.
    # Source defaults at shap_proxied_fs.py:100-104.
    "shap_proxied_revalidation_n_estimators_cfg": (100, None),
    "shap_proxied_revalidation_ucb_enabled_cfg": (True, False),
    "shap_proxied_revalidation_ucb_min_eval_size_cfg": (None, 3),
    "shap_proxied_revalidation_ucb_slack_cfg": (None, 0.0),
    "shap_proxied_revalidation_ucb_stdev_multiplier_cfg": (None, 1.0),
    # ShapProxiedFS Threading (S18). Gate on use_shap_proxied_fs=True.
    # Source default at shap_proxied_fs.py:113.
    "shap_proxied_inner_n_jobs_cap_cfg": (False, True),
    # MRMR Wave 8 LOW scalars (S32, S34, S35, S37). Gate on
    # use_mrmr_fs=True. Source defaults at filters/mrmr.py:241,249,252,265.
    "mrmr_relaxmrmr_alpha_cfg": (0.0, 0.1),
    "mrmr_uaed_auto_size_cfg": (False, True),
    "mrmr_cpt_test_cfg": (False, True),
    "mrmr_pid_synergy_bonus_cfg": (0.0, 0.1),
    # CV-selector LOW knobs (S41-S44). Gate on
    # composite_discovery_enabled_cfg=True. Source defaults at
    # _composite_target_discovery_config.py:127-130.
    "cv_selector_alpha_cfg": (1.0, 1.5),
    "cv_selector_confidence_cfg": (0.9, 0.99),
    "cv_selector_quantile_level_cfg": (0.9, 0.95),
    "cv_persist_fold_scores_cfg": (False, True),
    # =====================================================================
    # 2026-05-31 audit-pass-8 HIGH (#1-#4). Four production knobs that landed
    # in the 27814687..HEAD diff with no fuzz coverage. Defaults source-
    # verified at HEAD (see dataclass field comments).
    # =====================================================================
    # #1 MRMR cardinality-bias pre-screen + Miller-Madow correction toggle.
    # Source default True (filters/mrmr.py:334). Gates the high-card
    # pre-screen + MM-bias subtraction at the selection gate. Canonicalises
    # to True (source default) when use_mrmr_fs=False so the dedup pass
    # collapses non-MRMR combos onto a single canonical baseline.
    "mrmr_cardinality_bias_correction_cfg": (True, False),
    # #2 MRMR diminishing-returns stopping rule: stop greedy selection once
    # current gain drops below this fraction of the first-selected gain.
    # Source default 0.05 (filters/mrmr.py:326). 0.0 disables (legacy
    # absolute-floor-only path). Canonicalises to 0.05 when use_mrmr_fs=False.
    "mrmr_min_relevance_gain_relative_to_first_cfg": (0.05, 0.0),
    # #3 MLP sklearn-canonical random_state seed. Source default None
    # (training/neural/base.py:217). 42 seeds torch+numpy+python random+
    # Lightning DataLoader workers before any random op fires. Canonicalises
    # to None when neither 'mlp' in models nor recurrent_model_cfg is set.
    "mlp_random_state_cfg": (None, 42),
    # #4 MLP sklearn-canonical class_weight for imbalance handling. Source
    # default None (training/neural/base.py:218). "balanced" routes through
    # sklearn.utils.class_weight.compute_sample_weight and multiplies into
    # any caller-supplied sample_weight. Canonicalises to None outside the
    # compound gate (MLP active AND binary/multiclass classification AND
    # rare_5pct/rare_1pct imbalance).
    "mlp_class_weight_cfg": (None, "balanced"),
    # =====================================================================
    # 2026-05-31 audit-pass-8 MED + LOW->MED (#5/#7/#8/#9/#10). Five
    # production knobs that landed in the 27814687..HEAD diff with no fuzz
    # coverage. Defaults source-verified at HEAD (see dataclass field
    # comments).
    # =====================================================================
    # #5 ShapProxiedFS adaptive_prescreen_by_stability toggle. Source default
    # False (shap_proxied_fs.py:208). True activates the per-fold phi-mean
    # matrix + stability scoring + adaptive brute-force-cap narrowing.
    # Canonicalises to False when use_shap_proxied_fs=False.
    "shap_proxied_adaptive_prescreen_by_stability_cfg": (False, True),
    # #7 MLP use_layernorm default-flip (False since 2026-05-30 commit
    # a1ee8adb at flat.py:205). True exercises the post-StandardScaler /
    # embedding-output regime where LN is appropriate. Canonicalises to
    # False outside the compound gate ('mlp' in models AND target_type
    # == "regression").
    "mlp_use_layernorm_cfg": (False, True),
    # #8 MLP L1 penalty alpha. 0.0 = library default (no-op);
    # 0.001 fires the new BN/LN/GN-excluded L1 branch at
    # _flat_torch_module.py:272-301. Canonicalises to 0.0 outside the
    # 'mlp' in models gate.
    "mlp_l1_alpha_cfg": (0.0, 0.001),
    # #9 MLP all-zero-weight-batch WARN reachability. False = legacy fuzz
    # (no synthetic zero-weight injection); True = frame-builder injects a
    # contiguous N-row block whose recency weights collapse to ~0 so at
    # least one MLP training batch sees weight_sum < 1e-12. Canonicalises
    # to False outside the compound gate ('mlp' in models AND
    # weight_schemas != ("uniform",)).
    "mlp_inject_zero_sample_weight_batch_cfg": (False, True),
    # #10 XOR synergy pair injection for the new fleuret-mode conditional-
    # MI gate (feature_selection/filters/evaluation.py:596). False = legacy
    # fuzz frames; True = frame-builder emits a guaranteed XOR-synergy pair
    # (two binary cols whose XOR predicts y at high MI). Canonicalises to
    # False outside the compound gate (use_mrmr_fs=True AND
    # mrmr_interactions_max_order_cfg >= 2).
    "inject_xor_synergy_pair_cfg": (False, True),
    # =====================================================================
    # 2026-05-31 audit-pass-9 (W9). 8 new fuzz axes (5 HIGH + 3 MED) covering
    # MLP + MRMR + target-type surfaces added since 6c83a714 with no fuzz
    # coverage. Source-verified line numbers:
    #   #1 _flat_torch_module.py:499 (AdamW betas setdefault)
    #   #2 base.py:266-267 (use_ema/ema_params), :767-816 (SWA/EMA mutex)
    #   #3 base.py:268, :897-907 (label_smoothing multiclass-only)
    #   #4 base.py:269-270, :878-884 (focal_loss_gamma binary-only)
    #   #5 flat.py:208, :465-474 (use_residual + spectral_norm warning)
    #   #6 flat.py:209-210, :398-408 (numerical_embedding=plr + kwargs)
    #   #7 mrmr.py:656-665 (fe_hybrid_orth_enable + pair_enable)
    #   #8 _configs_base.py:126 (TargetTypes.MULTI_TARGET_REGRESSION)
    # =====================================================================
    # #1 AdamW betas: (0.9, 0.95) is the new realmlp-td-tuned default at
    # _flat_torch_module.py:499; PyTorch's legacy (0.9, 0.999) is the
    # second variant so the pre-flip ablation surface still gets fuzz
    # coverage. Canon collapses to (0.9, 0.95) outside ('mlp' in models
    # AND optimizer in {AdamW, Adam}); the fuzz runner uses the default
    # AdamW optimizer so the gate reduces to 'mlp' in models.
    "mlp_adamw_betas_cfg": ((0.9, 0.95), (0.9, 0.999)),
    # #2 use_ema: PytorchLightningEstimator(use_ema=...) -- new in F-28
    # (base.py:266-267 + callback wiring at :777-816). Default False;
    # True wires Lightning's WeightAveraging callback with EMA averaging
    # function + falls back to SWA-as-EMA shim on older Lightning. Canon
    # collapses to False outside 'mlp' in models. Mutual-exclusion with
    # use_swa is enforced at base.py:767 (raises ValueError); a separate
    # canon pin collapses use_ema=True to False whenever use_swa=True so
    # we never enumerate the ValueError-only combo.
    "mlp_use_ema_cfg": (False, True),
    # #3 label_smoothing: PytorchLightningEstimator(label_smoothing=...)
    # at base.py:268, gated to multiclass at :897-907 (replaces the
    # caller's CrossEntropyLoss with label_smoothing=eps). Default 0.0;
    # 0.1 fires the substitution path. Canon collapses to 0.0 outside
    # ('mlp' in models AND target_type == "multiclass_classification").
    "mlp_label_smoothing_cfg": (0.0, 0.1),
    # #4 focal_loss_gamma: PytorchLightningEstimator(focal_loss_gamma=...)
    # at base.py:269-270, gated to binary at :878-884 (substitutes
    # BCEWithLogitsLoss with sigmoid_focal_loss). Default None (BCE
    # unchanged); 2.0 fires the substitution. Canon collapses to None
    # outside ('mlp' in models AND target_type == "binary_classification");
    # paired with imbalance_ratio in {rare_5pct, rare_1pct} so the focal-
    # loss target (class imbalance) is actually present when the path
    # fires. Outside that compound gate the axis canonicalises to None.
    "mlp_focal_loss_gamma_cfg": (None, 2.0),
    # #5 use_residual: generate_mlp(use_residual=...) at flat.py:208 wraps
    # each Linear in a residual block; default False. Canon collapses to
    # False outside ('mlp' in models) AND ALSO collapses to False whenever
    # the default spectral_norm=True holds -- at flat.py:472-478 the
    # combination produces only a WARN and the residual path silently
    # treats the skip projection as non-Lipschitz; the meaningful arch
    # change happens only with spectral_norm=False. The fuzz suite doesn't
    # expose spectral_norm as an axis (library default True), so today the
    # canon collapses use_residual to False unconditionally outside the
    # MLP gate; once a spectral_norm axis is added the canon will be
    # refined to follow it.
    "mlp_use_residual_cfg": (False, True),
    "mlp_use_learnable_cat_embeddings_cfg": (True, False),
    "mlp_categorical_embed_dim_cfg": (None, 4, 16),
    # #6 numerical_embedding: generate_mlp(numerical_embedding=...) at
    # flat.py:209-210 + branch at :398-408. None = no embedding (default);
    # "plr" = PeriodicLinearEmbedding (multiplies input dim by
    # 2*n_frequencies + maybe 1). Canon collapses to None outside 'mlp'
    # in models. The kwargs-quartet (embed_dim, n_frequencies, sigma,
    # include_raw) is wired as a single literal axis -- "paper_default"
    # leaves the PLR module ctor at its NeurIPS-2024 RealMLP defaults,
    # "include_raw_false" overrides ``include_raw=False`` so the raw
    # numeric column is dropped from the embedded output (exercises the
    # narrower output-dim path).
    "mlp_numerical_embedding_cfg": (None, "plr"),
    "mlp_numerical_embedding_kwargs_cfg": ("paper_default", "include_raw_false"),
    # #7 mrmr_fe_hybrid_orth: MRMR(fe_hybrid_orth_enable=...) at
    # mrmr.py:656. Master switch defaults False (legacy byte-identical);
    # True enables the univariate orth + cross-basis pair pipeline (new
    # EngineeredRecipe kinds "orth_univariate" / "orth_pair_cross").
    # Sub-axis pair_enable defaults True (mrmr.py:664) once the master is
    # on; False disables the bilinear pair stage but keeps the univariate.
    # Canon: master collapses to False outside use_mrmr_fs; pair_enable
    # collapses to True outside (use_mrmr_fs AND master==True).
    "mrmr_fe_hybrid_orth_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_pair_enable_cfg": (False, True),
    # 2026-05-31 audit-pass-10 (W10) axes. Each extends iter615's master-on
    # hybrid-orth axis with the per-stage tunables that were exposed in
    # 8e385854 but never wired into the fuzz harness; plus a brand-new
    # optimizer axis covering the MuonAdamWHybrid path (F-33). Defaults
    # source-verified at HEAD:
    #   #1 mlp_optimizer_cfg = "adamw"
    #      (training/neural/_flat_torch_module.py:86 -- default falls back
    #      to torch.optim.AdamW when caller does not override). The Muon
    #      alternative wires MuonAdamWHybrid via mlp_kwargs["model_params"]
    #      ["optimizer"] = MuonAdamWHybrid (per _muon_optimizer.py:20 wiring
    #      contract). Canon collapses to "adamw" outside 'mlp' in models.
    #   #2 mrmr_fe_hybrid_orth_degrees_cfg = (2, 3)
    #      (feature_selection/filters/mrmr.py:657). Pair: (2, 3) vs (2,).
    #      Canon collapses to (2, 3) outside (use_mrmr_fs AND master==True).
    #   #3 mrmr_fe_hybrid_orth_basis_cfg = "auto"
    #      (feature_selection/filters/mrmr.py:658). Pair: "auto" vs "hermite".
    #      Canon collapses to "auto" outside (use_mrmr_fs AND master==True).
    #   #4 mrmr_fe_hybrid_orth_top_k_cfg = 5
    #      (feature_selection/filters/mrmr.py:663). Pair: 5 vs 1 (tie-break
    #      stress). Canon collapses to 5 outside (use_mrmr_fs AND master==True).
    #   #5 BLOCKED -- qcut hidden constants (n_unique<=32 threshold + q=10)
    #      live at _mrmr_fit_impl.py:276,281 as HARDCODED magic numbers, NOT
    #      MRMR ctor params. No fuzz axis can wire them today; the wiring
    #      requires a prior MRMR.__init__ promotion (e.g.
    #      ``fe_hybrid_orth_discrete_y_uniques_threshold`` +
    #      ``fe_hybrid_orth_qcut_bins`` ctor kwargs). See TODO marker below.
    #   #6 mrmr_fe_hybrid_orth_pair_max_degree_cfg = 2
    #      (feature_selection/filters/mrmr.py:665). Pair: 2 vs 3. Canon
    #      collapses to 2 outside (use_mrmr_fs AND master==True AND
    #      pair_enable==True) -- compound gate matches the pair-stage gate.
    "mlp_optimizer_cfg": ("adamw", "muon_hybrid"),
    "mrmr_fe_hybrid_orth_degrees_cfg": ((2, 3), (2,)),
    "mrmr_fe_hybrid_orth_basis_cfg": ("auto", "hermite"),
    "mrmr_fe_hybrid_orth_top_k_cfg": (5, 1),
    "mrmr_fe_hybrid_orth_pair_max_degree_cfg": (2, 3),
    # TODO(audit-pass-10 #5 -- ctor-promotion gap): the hybrid MI scorer at
    # ``_mrmr_fit_impl.py:276,281`` hardcodes ``n_unique<=32`` and
    # ``pd.qcut(q=10)`` as the discrete-vs-continuous y bucketing constants.
    # Neither is exposed via ``MRMR.__init__`` today, so the fuzz harness
    # cannot wire them as axes without a prior promotion step. Promote both
    # to ctor params (suggested names ``fe_hybrid_orth_discrete_y_uniques_threshold``
    # and ``fe_hybrid_orth_qcut_bins``) and then add the matching
    # ``mrmr_qcut_unique_threshold_cfg`` (e.g. (32, 16)) / ``mrmr_qcut_q_cfg``
    # (e.g. (10, 5)) axes here. Until then, Layer 24 Scenario D pins R^2>=0.85
    # at the implicit q=10/threshold=32 calibration and there is no fuzz
    # discrimination across alternate values.
    # =====================================================================
    # 2026-05-31 audit-pass-12 (W12). 12 fuzz axes (5 HIGH + 5 MED + 2 LOW)
    # covering the post-e8d11293 production knob surface (22 commits, F-34
    # MTR dispatch + 8 MRMR FE layers + MRMR/ShapProxiedFS artifact-reuse
    # pipeline). Source defaults verified at HEAD; see dataclass field
    # comments for line citations.
    # =====================================================================
    # Group A -- F-34 MTR suite-side dispatch.
    # A1 CompositeTargetDiscoveryConfig.multilabel_strategy (validator at
    # _composite_target_discovery_config.py:940 accepts {"per_target",
    # "skip", "multi_target_regression"}; field default "per_target" at :773).
    # Gate: target_type in {multilabel_classification, multi_target_regression};
    # canon-collapse to "per_target" otherwise (the field is consumed only
    # on those target types).
    "composite_target_multilabel_strategy_cfg": (
        "per_target", "multi_target_regression",
    ),
    # A2 cross-target ensemble enablement marker. The early-return WARN at
    # _phase_composite_post_xt_ensemble._build_cross_target_ensemble_for_target
    # fires when target_type == TargetTypes.MULTI_TARGET_REGRESSION (D2 in
    # commit d48245de). Existing fuzz never co-varies this with the MTR
    # target_type. New axis is canon-only (the WARN gate is suite-internal,
    # no top-level kwarg today); pair (True, False) exercises both branches
    # AND the canon dedup pins True on combos where the CT ensemble would
    # be skipped anyway.
    "enable_ct_ensemble_cfg": (True, False),
    # A3 MTR metric-registry coverage marker (canon-only). The new entries
    # in metrics_registry._register_builtin_multi_target_regression
    # (rmse_macro/_micro/_max, mae_macro/_max, r2_macro/_min) are reachable
    # only when target_type=multi_target_regression. Canon collapses to
    # None outside the MTR target so dedup absorbs non-MTR combos that
    # could not consume the metric value.
    "mtr_eval_metric_cfg": (None, "rmse_macro"),
    # Group B -- MRMR FE layer master switches (all default-OFF per
    # filters/mrmr.py; legacy bit-identical at False).
    # B1 Layer 33 K-fold target encoding (mrmr.py:705 fe_kfold_te_enable).
    # Gate: use_mrmr_fs=True AND combo has categorical column.
    "mrmr_fe_kfold_te_enable_cfg": (False, True),
    # B2 Layer 37 missingness-aware FE three sub-master switches.
    # mrmr.py:749/751/752 (indicator / count / pattern). Compound gate:
    # use_mrmr_fs=True AND (inject_inf_nan OR inject_all_nan_col OR
    # cat columns with null_fraction_cats > 0 -- any source of NaNs in
    # the frame).
    "mrmr_fe_missingness_indicator_enable_cfg": (False, True),
    "mrmr_fe_missingness_count_enable_cfg": (False, True),
    "mrmr_fe_missingness_pattern_enable_cfg": (False, True),
    # B3 Layer 34 count / freq / cat-num residual FE. Single 4-way axis
    # maps to mrmr.py:723/725/727 (fe_count_encoding_enable /
    # fe_frequency_encoding_enable / fe_cat_num_interaction_enable). Gate:
    # use_mrmr_fs=True AND categorical column present.
    "mrmr_fe_cat_aux_enable_cfg": ("off", "count", "freq", "interaction"),
    # B4 Layer 32 spline + Fourier extra bases (mrmr.py:676 default ()).
    # Compound gate: use_mrmr_fs=True AND mrmr_fe_hybrid_orth_enable_cfg=True
    # (extra-basis stage runs after the polynomial stages at the hybrid
    # entry point; no-op when master is off).
    "mrmr_fe_hybrid_orth_extra_bases_cfg": ((), ("spline",), ("fourier",)),
    # B5 Layer 38 ratio + grouped-delta + lagged-diff FE. Single 4-way axis
    # maps to mrmr.py:769/772/774/777 master switches. Gate: use_mrmr_fs=True.
    # "grouped_delta" requires fe_grouped_delta_group_col (None today in
    # fuzz frame builder); "lagged_diff" requires fe_lagged_diff_time_col
    # (None today). Both kinds canon-collapse to "off" unless the frame
    # builder emits a candidate group_col / time_col.
    "mrmr_fe_ratio_delta_diff_cfg": (
        "off", "ratio", "grouped_delta", "lagged_diff",
    ),
    # B6 Layer 26 generic MI-greedy FE (mrmr.py:691 fe_mi_greedy_enable).
    # Sibling to fe_hybrid_orth; gate is use_mrmr_fs=True only (no compound
    # gate needed).
    "mrmr_fe_mi_greedy_enable_cfg": (False, True),
    # Group C -- MRMR + ShapProxiedFS artifact-reuse pipeline.
    # C1 master switch for the cross-selector artifact handoff. "off" keeps
    # MRMR.retain_artifacts=False AND does NOT pass precomputed=... to
    # ShapProxiedFS (legacy). "on" sets MRMR.retain_artifacts=True (mrmr.py:787)
    # AND threads MRMR.export_artifacts() into ShapProxiedFS(precomputed=...)
    # at shap_proxied_fs.py:258. Compound gate: use_mrmr_fs=True AND
    # use_shap_proxied_fs=True (both selectors must be in the chain for
    # the handoff to fire).
    "mrmr_shap_proxy_artifact_reuse_cfg": ("off", "on"),
    # C2 align_precomputed_to_X branch selector. The 4 branches at
    # _shap_proxy_precomputed.align_precomputed_to_X are:
    #   "exact":       names == X.columns in order (line 168)
    #   "permuted":    same set, different order (line 180)
    #   "subset":      X.columns is a subset of feature_names (line 180)
    #   "mismatched":  consumer / producer disagree; returns (None, report)
    #                  with WARN-and-fall-back (line 216)
    # Gate: mrmr_shap_proxy_artifact_reuse_cfg == "on" (otherwise no
    # precomputed dict is passed).
    "mrmr_shap_proxy_align_mode_cfg": (
        "exact", "permuted", "subset", "mismatched",
    ),
    # TODO(audit-pass-12 D1 -- kernel_tuning_cache verification): the
    # Layer 30 bulk corrcoef dedup (commit 77478957) + Layer 31 numba
    # batched MI scorer (commit b6c3ab0d) are auto-dispatched with no
    # user-facing toggle exposed on the MRMR ctor. Tuning-cache keys live
    # at ``mlframe.feature_selection.filters.batch_pair_mi_gpu`` (Wave 23
    # P1) + ``hermite_fe.py:616`` (Wave 23 P2) + ``gpu.py:478`` (joint-
    # hist dispatch) + ``discretization.py:746`` (per-host cache). No new
    # MRMR ctor kwarg surfaces in the e8d11293..HEAD diff -- the cache
    # is consulted internally and the dispatcher is deterministic given
    # a fixed host. No fuzz axis can be wired without source-side ctor
    # promotion (e.g. ``MRMR(disable_kernel_tuning_cache=True)``). See
    # FUZZ_AXES_W12_IMPL_DONE.md for the recommended source-side
    # promotion if reproducibility-without-cache becomes a fuzz target.
    # =====================================================================
    # 2026-05-31 audit-pass-14 (W14). 6 fuzz axes (2 HIGH default-flip +
    # 3 MED new auto-mode/feature + 1 LOW shape invariant) covering the
    # 8b581eea..34578dab iter69-76 + Layers 44-54 diff. Defaults
    # SOURCE-verified at HEAD against
    # src/mlframe/feature_selection/shap_proxied_fs.py:249, :258 and
    # src/mlframe/feature_selection/filters/mrmr.py:621-655, :845-847,
    # :947-950. See AUDIT_PASS_14_DONE.md for the per-finding citations.
    # F14-6 (fe_provenance_ shape invariant) is implemented as a sensor
    # test in test_fuzz_combo_cross_axis_W11C.py rather than a fuzz axis
    # (no opt-out switch exposed in MRMR.__init__).
    # =====================================================================
    # F14-1 [HIGH default-flip] ShapProxiedFS.cluster_backend
    # (shap_proxied_fs.py:258, default "auto" since iter75). "auto" routes
    # to SU at width<=cluster_su_auto_max_features (default 2000) else
    # Pearson; "pearson" is the legacy regime that fuzz used to exercise
    # by default. Gate: use_shap_proxied_fs=True (the toggle is unread
    # otherwise). Canon-collapse to "auto" outside the gate.
    "shap_proxied_cluster_backend_cfg": ("auto", "su", "pearson"),
    # F14-3 [MED new feature] MRMR partial_fit streaming ctor knobs
    # (mrmr.py:845-847, Layer 53). The partial_fit() public API is itself
    # a separate code path that the fuzz suite does NOT invoke today;
    # these ctor params shape the future partial_fit behaviour and are
    # wired here as coverage-marker axes so the (param != default) ctor
    # branches still receive pairwise enumeration. Gate: use_mrmr_fs=True;
    # canon-collapses to the source defaults outside the gate.
    "mrmr_partial_fit_decay_cfg": (0.0, 0.3),
    "mrmr_partial_fit_min_recompute_cfg": (100, 50),
    "mrmr_partial_fit_window_cfg": (None, 500),
    # F14-4 [MED new auto-mode] MRMR.dcd_tau_cluster (mrmr.py:621, Layer 47).
    # Type pin dropped: now accepts ``'auto'`` to enable bimodal SU valley
    # detection, falling back to 0.7 on unimodal / degenerate. The
    # _dcd_tau_auto.calibrate_tau path is unreached without this axis.
    # Gate: use_mrmr_fs=True AND mrmr_dcd_enable_cfg=True; canon to 0.7
    # outside that compound gate.
    "mrmr_dcd_tau_cluster_cfg": (0.7, "auto"),
    # F14-5 [MED new auto-mode] MRMR.dcd_distance (mrmr.py:622, Layer 46).
    # "auto" runs SU + VI per pair and returns max; "su" is the legacy
    # default. Gate: use_mrmr_fs=True AND mrmr_dcd_enable_cfg=True; canon
    # to "su" outside the gate.
    "mrmr_dcd_distance_cfg": ("su", "auto"),
    # F14-5 [MED Layer 44 aggregator enrichment] MRMR.dcd_swap_method
    # (mrmr.py:655, _VALID_DCD_SWAP_METHODS expanded at :947-950 to add
    # ``pca_pc2``, ``median_z``, ``signed_max_abs``, ``signed_l2_sum``).
    # "auto" picks per pair; the explicit values pin a single new method
    # for fuzz repro. Gate: use_mrmr_fs=True AND mrmr_dcd_enable_cfg=True;
    # canon to "auto" outside the gate.
    "mrmr_dcd_swap_method_cfg": (
        "auto", "mean_z", "pca_pc2", "median_z", "signed_max_abs",
    ),
    # F14-2 [HIGH default-flip] -- the existing
    # ``shap_proxied_shap_aware_stage1_cushion_cfg`` pair (2, 4) declared
    # above (~line 1198) is extended with the legacy value 8 to keep
    # fuzz coverage of the pre-iter76 calibration. The source default
    # flipped 8 -> 2 in iter76 (shap_proxied_fs.py:249) so the cushion=8
    # branch is now untested in default config. The pair is extended in
    # place rather than redeclared here; see the existing entry at the
    # Stage-A LOW-tier block.
    # =====================================================================
    # iter639 audit-pass-15. Layers 63-85 + F-62..F-72 newly exposed
    # config knobs that had no fuzz coverage. Defaults SOURCE-verified at
    # HEAD against src/mlframe/feature_selection/filters/mrmr.py and
    # src/mlframe/training/neural/_flat_torch_module.py /
    # src/mlframe/training/neural/flat.py.
    # =====================================================================
    # MRMR Layers 63-76, 85 -- hybrid-orth scorer family.
    # H1 fe_hybrid_orth_default_scorer (mrmr.py:1092, default "plug_in").
    # Layer 85 routes ALL univariate FE scoring through one of 8 estimators
    # (plug_in / ksg / copula / dcor / hsic / jmim / tc / cmim); each is a
    # distinct numerical code path with separate failure modes, and plug_in
    # hides the other 7. Pair (plug_in, ksg, copula) covers the histogram-
    # based default + the two highest-leverage alternates (kNN MI + rank-
    # invariant copula MI). Gate: use_mrmr_fs=True AND
    # mrmr_fe_hybrid_orth_enable_cfg=True; canon-collapse to "plug_in"
    # outside that compound gate.
    # audit-pass-17: + "auto_oracle" (L100, Param-Oracle-driven scorer pick).
    "mrmr_fe_hybrid_orth_default_scorer_cfg": ("plug_in", "ksg", "copula", "auto_oracle"),
    # H2 fe_hybrid_orth_meta_enable (mrmr.py:1068, default False). Layer 76
    # meta-scorer signal-fingerprint auto-select drives a per-call dispatcher
    # that picks one of the 8 scorers from data stats; mis-fingerprinting
    # on weird-cat / inject_inf_nan combos is a leakage hazard. Gate:
    # use_mrmr_fs=True AND mrmr_fe_hybrid_orth_enable_cfg=True; canon to
    # False outside.
    "mrmr_fe_hybrid_orth_meta_enable_cfg": (False, True),
    # H3 fe_hybrid_orth_bootstrap_enable (mrmr.py:878, default False).
    # Layer 62 bootstrap-stable MI resamples rows during scoring; behavioural
    # divergence from non-bootstrapped scoring surfaces NaN-propagation +
    # RNG-mutation regressions. Gate: use_mrmr_fs=True AND
    # mrmr_fe_hybrid_orth_enable_cfg=True; canon to False outside.
    "mrmr_fe_hybrid_orth_bootstrap_enable_cfg": (False, True),
    # H4 fe_hybrid_orth_three_gate_enable (mrmr.py:897, default False).
    # Layer 63 three-gate K-fold OOF CMI introduces a CV split inside FE;
    # interacts with shuffle_val_cfg / wholeday_splitting_cfg in ways the
    # existing axes do not probe. Gate: use_mrmr_fs=True AND
    # mrmr_fe_hybrid_orth_enable_cfg=True; canon to False outside.
    "mrmr_fe_hybrid_orth_three_gate_enable_cfg": (False, True),
    # Neural F-62..F-72 -- MLP training options.
    # N1 use_sam (training/neural/_flat_torch_module.py:48, default False).
    # F-63 SAM optimizer wrapper changes training_step into a double-forward;
    # interacts with class_weight + sample_weight in ways the plain AdamW
    # path doesn't. Gate: 'mlp' in models; canon to False outside.
    "mlp_use_sam_cfg": (False, True),
    # N2 use_lookahead (training/neural/_flat_torch_module.py:43, default
    # False). F-62 Lookahead has 3 fix commits (F-A/F-B/F-D) for slow-weight
    # commit + state_dict round-trip; recent fix density signals it needs
    # broader combo coverage. Gate: 'mlp' in models; canon to False outside.
    "mlp_use_lookahead_cfg": (False, True),
    # N3 use_mixup (training/neural/_flat_torch_module.py:46, default False).
    # F-68/F-69/F-70 Mixup mutates labels in training_step; multilabel +
    # label_smoothing + focal_loss + multi_target combinations are the
    # silent-correctness landmines. Gate: 'mlp' in models; canon to False
    # outside.
    "mlp_use_mixup_cfg": (False, True),
    # N4 spectral_norm_output_only (training/neural/flat.py:224, default
    # False). F-72 cheap output-only SN wraps just the final Linear;
    # distinct code path from full-SN and exercises the bounded-output +
    # Lipschitz interaction on classifier vs regressor. Gate: 'mlp' in
    # models; canon to False outside.
    "mlp_spectral_norm_output_only_cfg": (False, True),
    # iter642 audit-pass-15 batch 2. The 6 remaining MRMR hybrid-orth
    # sub-features the W15 audit identified as MED-leverage; each enables
    # a distinct numerical path that the master-on axis alone does not
    # cover. Defaults source-verified at HEAD. All canon-collapse to False
    # outside (use_mrmr_fs AND mrmr_fe_hybrid_orth_enable_cfg).
    # H5 fe_hybrid_orth_ensemble_enable (mrmr.py:1044). Layer 69 rank-
    # fusion combines >=3 scorer rankings via mean_rank/borda;
    # combinatorial blow-up risk on rare_imbalance + multilabel.
    "mrmr_fe_hybrid_orth_ensemble_enable_cfg": (False, True),
    # H6 fe_hybrid_orth_lasso_enable (mrmr.py:784). Layer 81 Lasso
    # pre-selection inserts an sklearn LinearModel inside the FE loop;
    # degenerate-col + label-leak + rank-deficient combos are exactly
    # where Lasso silently blows up.
    "mrmr_fe_hybrid_orth_lasso_enable_cfg": (False, True),
    # H7 fe_hybrid_orth_elasticnet_enable (mrmr.py:800). Layer 82
    # ElasticNet alt path with separate l1_ratio; sibling to Lasso but
    # different solver/penalty arithmetic.
    "mrmr_fe_hybrid_orth_elasticnet_enable_cfg": (False, True),
    # H8 fe_hybrid_orth_adaptive_arity_enable (mrmr.py:749). Layer 78
    # picks pair/triplet/quadruplet arity per column; one knob exercises
    # the 3 multi-arity assemblies that triplet_enable / quadruplet_enable
    # axes only test in isolation.
    "mrmr_fe_hybrid_orth_adaptive_arity_enable_cfg": (False, True),
    # H9 fe_hybrid_orth_diff_basis_enable (mrmr.py:845). Layer 59 emits
    # Hermite-difference basis features only when source-source
    # correlation > threshold; exercises the unique col-pair branching.
    "mrmr_fe_hybrid_orth_diff_basis_enable_cfg": (False, True),
    # H10 fe_semi_supervised_enable (mrmr.py:767). Layer 80 fits orth-poly
    # bases on unlabeled-pool X; thread-local pool plumbing + leakage-by-
    # construction claim deserve fuzz coverage.
    "mrmr_fe_semi_supervised_enable_cfg": (False, True),
    # =====================================================================
    # audit-pass-16: MRMR Layers 87-91 (NVIDIA #1-4 + two-tier IT gates).
    # 8 new master-enable / mode-selector axes; defaults source-verified at
    # HEAD against MRMR.__init__. All gate on use_mrmr_fs=True; canon-collapse
    # to source defaults outside their gate so dedup absorbs phantom variation.
    # =====================================================================
    # L87 grouped multi-stat aggregator (mrmr.py:1255). Master switch for the
    # grouped-agg recipe stage (auto-detect group cols + CMI/uplift gate).
    "mrmr_fe_grouped_agg_enable_cfg": (False, True),
    # L88 per-group quantile FE (mrmr.py:1268) + target-aware mode (mrmr.py:1270).
    # target_aware toggles the OOF-fit supervised MDLP-edge path -- a distinct
    # leakage-sensitive branch vs unsupervised quantiles.
    "mrmr_fe_grouped_quantile_enable_cfg": (False, True),
    "mrmr_fe_grouped_quantile_target_aware_cfg": (False, True),
    # L89 cat-cat synergy cross (mrmr.py:1285). Interaction-information-filtered
    # cat-pair cross + cardinality-routed TE/raw-code path.
    "mrmr_fe_cat_pair_enable_cfg": (False, True),
    # L90 numeric decomposition (mrmr.py:1300) + digits emitter (mrmr.py:1302).
    # Empty digits tuple disables the digit_extract emitter (rounding-only path)
    # vs the default that exercises both emitters -- a distinct branch.
    "mrmr_fe_numeric_decompose_enable_cfg": (False, True),
    "mrmr_fe_numeric_decompose_digits_cfg": ((0, 1, 2), ()),
    # L91 two-tier IT gates. Tier-1 raw-floor MI pruning wires into all four
    # *_with_recipes wrappers; Tier-2 greedy cross-mechanism CMI dedup pass
    # after the L27 Spearman dedup.
    # audit-pass-17: L97 (cfe9640b) flipped fe_local_mi_gate default to True
    # (mrmr.py:1267). Track the source default first in the tuple so the
    # default-config run exercises the prod gate state; (True, False) still
    # covers the OFF variant.
    "mrmr_fe_local_mi_gate_cfg": (True, False),
    "mrmr_fe_unified_second_pass_gate_cfg": (False, True),
    # =====================================================================
    # audit-pass-17: MRMR Param-Oracle / fe_auto + FE families L92-104.
    # Defaults source-verified at HEAD against MRMR.__init__. All gate on
    # use_mrmr_fs=True; canon-collapse to source defaults outside the gate.
    # =====================================================================
    # L99 Meta FE-recommender (mrmr.py:1478). fe_auto=True turns ~50 FE flags
    # ON via the rule recommender for the fit then restores them. The single
    # biggest behavioural surface; the per-flag axes are dead while it's on.
    "mrmr_fe_auto_cfg": (False, True),
    # L92 temporal leak-safe grouped aggregation (mrmr.py:1426).
    "mrmr_fe_temporal_agg_enable_cfg": (False, True),
    # L93 multi-column composite group-key aggregation (mrmr.py:1296).
    "mrmr_fe_composite_group_agg_enable_cfg": (False, True),
    # L95 periodic/modular decompose (mrmr.py:1375) + per-group distribution
    # distance (mrmr.py:1387) -- two distinct generators.
    "mrmr_fe_modular_enable_cfg": (False, True),
    "mrmr_fe_group_distance_enable_cfg": (False, True),
    # L104 rare-category (mrmr.py:1403) + conditional-residual (mrmr.py:1407)
    # FE families.
    "mrmr_fe_rare_category_enable_cfg": (False, True),
    "mrmr_fe_conditional_residual_enable_cfg": (False, True),
    # =====================================================================
    # 2026-06-13 -- coverage refresh for the embedding-passthrough + the five
    # default-ON / one default-OFF MRMR FE families that landed across the
    # ~140-commit window and had no fuzz axis. All gate on use_mrmr_fs=True.
    # The five default-ON families list their source default FIRST (so the
    # default-config run keeps exercising the prod state) and add the OFF
    # branch that nothing previously fuzzed; the gradient-interaction seeder
    # is default-OFF so its tuple lists False first then the un-fuzzed ON.
    # =====================================================================
    # embedding/free-text passthrough through MRMR (commits b896abbf / 3eb0a564).
    # embedding_passthrough=True (source default, mrmr/_mrmr_class.py:2404) routes
    # detected embedding (pl.List) + free-text columns AROUND the selector core;
    # False forces MRMR to try to consume them, exercising the consume/coerce path
    # the passthrough was built to avoid. The two detect-* sub-knobs only bite when
    # the master is on and the frame actually carries an embedding / text column.
    # Compound gate (canonical_key): use_mrmr_fs=True AND (embedding_col_count>0 OR
    # text_col_count>0) -- otherwise there is nothing to passthrough and the axis
    # canon-collapses to the source default.
    "mrmr_embedding_passthrough_cfg": (True, False),
    "mrmr_embedding_passthrough_detect_embeddings_cfg": (True, False),
    "mrmr_embedding_passthrough_detect_text_cfg": (True, False),
    # MRMR FE families added in the recent window, default-ON in MRMR.__init__ --
    # the OFF branch was never fuzzed. hinge_basis (change-point / kink basis,
    # mrmr/_mrmr_class.py:1306), conditional_dispersion (per-bin spread features,
    # :2282), wavelet_basis (multi-scale leg features, :2307), stability_vote
    # (k-fold selection-stability voting, :454), sufficient_summary_early_stop
    # (H(y)-relative residual stop that halts the FE search, :512). All gate on
    # use_mrmr_fs=True; canon-collapse to the source default (True) outside.
    "mrmr_fe_hinge_enable_cfg": (True, False),
    "mrmr_fe_conditional_dispersion_enable_cfg": (True, False),
    "mrmr_fe_wavelet_enable_cfg": (True, False),
    "mrmr_fe_stability_vote_enable_cfg": (True, False),
    "mrmr_fe_sufficient_summary_early_stop_cfg": (True, False),
    # Gradient-interaction seeder (mrmr/_mrmr_class.py:1537, default-OFF, bench-
    # rejected 2026-06-10 so it stays opt-in). The ON branch -- a GBM-gradient
    # co-occurrence interaction seeder -- has no fuzz exposure. Gate use_mrmr_fs
    # AND interactions_max_order>=2 (the seeder feeds the interaction stage);
    # canon to False outside.
    "mrmr_fe_gradient_interaction_enable_cfg": (False, True),
    # MRMR FE-family + escalation + hybrid-orth scorer master toggles. Each pair is (MRMR.__init__ default, flipped). All gate on use_mrmr_fs=True; canon-collapse to the source default outside so dedup absorbs no-op variation when MRMR is off.
    "mrmr_fe_rung_schedule_enable_cfg": (True, False),
    "mrmr_fe_auto_escalation_enable_cfg": (True, False),
    "mrmr_fe_escalation_underdelivery_enable_cfg": (True, False),
    "mrmr_fe_synergy_prevalence_rescue_enable_cfg": (True, False),
    "mrmr_fe_pair_prewarp_enable_cfg": (True, False),
    "mrmr_fe_univariate_basis_enable_cfg": (True, False),
    "mrmr_fe_univariate_fourier_enable_cfg": (True, False),
    "mrmr_fe_hybrid_orth_triplet_enable_cfg": (True, False),
    "mrmr_fe_hybrid_orth_quadruplet_enable_cfg": (True, False),
    "mrmr_fe_binned_numeric_agg_enable_cfg": (True, False),
    "mrmr_fe_discrete_structural_operators_enable_cfg": (True, False),
    "mrmr_fe_pairwise_modular_enable_cfg": (True, False),
    "mrmr_fe_integer_lattice_enable_cfg": (True, False),
    "mrmr_fe_row_argmax_enable_cfg": (True, False),
    "mrmr_fe_conditional_gate_enable_cfg": (True, False),
    "mrmr_fe_escalation_feedforward_enable_cfg": (False, True),
    "mrmr_fe_gate_med_enable_cfg": (False, True),
    "mrmr_fe_pair_perm_null_admission_enable_cfg": (False, True),
    "mrmr_fe_ii_routing_enable_cfg": (False, True),
    "mrmr_fe_gbm_seeder_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_adaptive_degree_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_conditional_routing_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_cluster_basis_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_ksg_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_copula_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_dcor_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_hsic_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_jmim_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_tc_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_cmim_enable_cfg": (False, True),
    "mrmr_fe_hybrid_orth_auto_scorer_enable_cfg": (False, True),
    "mrmr_fe_mi_greedy_cmi_enable_cfg": (False, True),
    "mrmr_fe_cat_triple_enable_cfg": (False, True),
    "mrmr_fe_rankgauss_enable_cfg": (False, True),
    # FeatureSelectionConfig RFECV first-class lever fields (D-surface, commit 55a31c6c). Gated to rfecv-on in canonical_key.
    # enable_permutation_importance flips RFECV importance_getter to permutation; prescreen='univariate_ht' fires the in-tree
    # FDR univariate pre-filter (no external dep, pandas-only -> auto-skips on polars); swap_top_k arms the SFFS post-convergence swap pass.
    "rfecv_enable_permutation_importance_cfg": (False, True),
    "rfecv_prescreen_cfg": (None, "univariate_ht"),
    "rfecv_swap_top_k_cfg": (None, 2),
    # TrainingSplitConfig time-aware split surface (E2, commits e3c45edb / e935a166 / 1730e5aa). cv_strategy routes the main split
    # (random / forward-walk timeseries / purged-embargo); cv_purge sets the purged embargo gap; conformal_size carves a second holdout.
    "cv_strategy_cfg": ("random", "timeseries", "purged"),
    "cv_purge_cfg": (0, 5),
    "conformal_size_cfg": (None, 0.05),
}
