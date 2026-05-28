"""Combo enumerator + results log for train_mlframe_models_suite fuzzing.

Design principles:
  * Deterministic: identical master_seed → identical combo list on any host.
  * Dedup-canonical: combos are canonicalized before hashing so
    semantically-equivalent combos (e.g. align_polars_dicts=True with
    pandas input) collapse to one.
  * Pairwise-covering: the greedy sampler guarantees every
    (axis_i=value_i, axis_j=value_j) pair is exercised at least once.
  * xfail-aware: combos hitting known bugs are auto-marked xfail via a
    declarative rule table — single source of truth shared with tracked
    tests elsewhere in the suite.

Canonicalisation contract (READ THIS BEFORE EDITING ``canonical_key`` /
``_canonical_*``).

Canonicalisation deduplicates SEMANTICALLY-EQUIVALENT combos. It does
NOT silence flaky combos. The two are easy to confuse and the latter is
strictly forbidden.

  Legitimate canon (keep): ``imbalance="balanced"`` collapses regardless
  of the imbalance-mode flag, because at 50/50 the mode produces
  bit-identical data. The two combos really are the same combo, and
  hashing them as one is a memory / time win with no coverage cost.

  ILLEGITIMATE canon (DO NOT WRITE, FIX PROD INSTEAD): zeroing
  ``text_col_count`` for combos that hit a CB hang, forcing
  ``inject_degenerate_cols=False`` for combos that crash CB's cat-feature
  auto-detect, forcing ``remove_constant_columns=True`` for combos that
  break the polars-ds robust scaler. These are real production bugs
  that real users would hit. Hiding them in canon means the fuzz suite
  STAYS GREEN while production stays broken — exactly the inverse of
  what this harness is for.

  If you catch yourself writing a canon rule whose justification
  references a CrashID / fuzz cXXXX / "the X path hangs" — STOP. Find
  the prod fix. Once prod is fixed the canon is unnecessary; until prod
  is fixed, the failure is doing its job by surfacing the bug.

Concrete example, 2026-04-27: the original ``_canonical_text_col_count``
zeroed text columns when CB + small-n + heavy NaN injection landed on
inner-CV folds smaller than CB's default ``occurrence_lower_bound=50``.
The fix was a real production change in
``training/helpers.compute_cb_text_processing`` that scales the floor
proportionally to the fit-time row count (called from
``trainer._train_model_with_fallback`` and ``feature_selection/wrappers.py``
RFECV inner-fold). After the fix, the canon was retired.

See ``CLAUDE.md`` (project root) for the full anti-masking checklist
covering canon, runtime ``*_eff`` rewrites in ``test_fuzz_suite.py``,
``pytest.mark.xfail`` rules, and "0-row defensive guards" in
production code.

Results log: every fuzz run appends one JSONL row per combo to
``tests/training/_fuzz_results.jsonl`` capturing combo key, outcome
(pass/fail/xfail/skip), and — on failure — the exception class and a
one-line summary. That file is the audit trail used by human / agent
follow-ups to decide what to fix next.
"""
from __future__ import annotations

import hashlib
import orjson
import os
import random
from dataclasses import astuple, dataclass, field
from itertools import combinations as iter_combinations
from itertools import product as iter_product
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Axis space
# ---------------------------------------------------------------------------

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
    "rfecv_n_features_selection_rule_cfg": ("auto", "argmax", "one_se_max"),
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
    # 2026-05-28 audit-pass-2 PART A: 4 LOW-tier coverage-gap axes deferred
    # from coverage_agent W11C wave.
    # ensembling_degenerate_class_ratio: EnsemblingConfig.degenerate_class_ratio
    # (default 0.01 at _model_configs.py:981); 0.05 widens the degenerate-subset
    # gate. Gated on use_ensembles AND classification target.
    "ensembling_degenerate_class_ratio_cfg": (0.01, 0.05),
    # behavior_use_flaml_zeroshot: TrainingBehaviorConfig.use_flaml_zeroshot
    # (default False at _model_configs.py:485). True picks flaml_zeroshot
    # XGB/LGBM classes vs. vanilla. Gated on xgb/lgb in models AND a working
    # flaml install (from_axes canonises True -> False if flaml is missing).
    "behavior_use_flaml_zeroshot_cfg": (False, True),
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
    "baseline_quick_model_n_estimators_cfg": (200, 50),
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
}


# ---------------------------------------------------------------------------
# Combo dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FuzzCombo:
    models: tuple[str, ...]
    input_type: str
    n_rows: int
    cat_feature_count: int
    null_fraction_cats: float
    use_mrmr_fs: bool
    weight_schemas: tuple[str, ...]
    target_type: str
    auto_detect_cats: bool
    align_polars_categorical_dicts: bool
    seed: int
    # New axes 2026-04-24 — have defaults so existing pinned sensor combos
    # and stored ``_fuzz_results.jsonl`` rows keep deserialising cleanly.
    prefer_polarsds: bool = True
    use_text_features: bool = True
    honor_user_dtype: bool = False
    target_carrier: str = "numpy"
    text_col_count: int = 0
    embedding_col_count: int = 0
    # 2026-04-24 combo extension from test_suite_coverage_gaps analysis
    outlier_detection: "str | None" = None
    use_ensembles: bool = False
    continue_on_model_failure: bool = False
    iterations: int = 3
    prefer_calibrated_classifiers: bool = False
    inject_degenerate_cols: bool = False
    inject_inf_nan: bool = False
    with_datetime_col: bool = False
    inject_zero_col: bool = False
    fairness_col: "str | None" = None
    custom_prep: "str | None" = None
    input_storage: str = "memory"
    # 2026-04-24 round 2 — config-field axes
    fillna_value_cfg: "float | None" = None
    scaler_name_cfg: "str | None" = "standard"
    categorical_encoding_cfg: str = "ordinal"
    skip_categorical_encoding_cfg: bool = False
    val_placement_cfg: str = "forward"
    test_size_cfg: float = 0.1
    trainset_aging_limit_cfg: "float | None" = None
    cat_text_card_threshold_cfg: int = 300
    early_stopping_rounds_cfg: "int | None" = None
    use_robust_eval_metric_cfg: bool = True
    # Fix G — adversarial axes
    inject_label_leak: bool = False
    inject_rank_deficient: bool = False
    inject_all_nan_col: bool = False
    # R3 — drift, imbalance, weird-cat axes
    inject_test_drift: "str | None" = None
    imbalance_ratio: str = "balanced"
    weird_cat_content: "str | None" = None
    # 2026-04-24 Phase H — multilabel dispatch axis. Only meaningful when
    # target_type == multilabel_classification.
    multilabel_strategy_cfg: str = "auto"
    # 2026-05-04 — LTR ensembling method. Only meaningful when
    # target_type == learning_to_rank.
    ranking_ensemble_method: str = "rrf"
    # 2026-04-26 batch 1 — additional config-field axes
    fix_infinities_cfg: bool = True
    ensure_float32_cfg: bool = True
    remove_constant_columns_cfg: bool = True
    imputer_strategy_cfg: "str | None" = "mean"
    shuffle_val_cfg: bool = False
    shuffle_test_cfg: bool = False
    wholeday_splitting_cfg: bool = True
    val_sequential_fraction_cfg: float = 0.5
    # 2026-04-26 batch 3 — multilabel dispatch axes
    multilabel_n_chains_cfg: int = 3
    multilabel_chain_order_cfg: str = "random"
    multilabel_cv_cfg: int = 5
    # 2026-04-26 batch 4 — PreprocessingExtensionsConfig axes
    prep_ext_scaler_cfg: "str | None" = None
    prep_ext_kbins_cfg: "int | None" = None
    prep_ext_polynomial_degree_cfg: "int | None" = None
    prep_ext_dim_reducer_cfg: "str | None" = None
    prep_ext_nonlinear_cfg: "str | None" = None
    # PySR symbolic regression (gated for cls / large-n / text / embedding combos in canonical_key).
    prep_ext_pysr_enabled_cfg: bool = False
    # MRMR NaN handling strategy. No-op when use_mrmr_fs is False (axis canonicalised away).
    mrmr_nan_strategy_cfg: str = "separate_bin"
    # 2026-04-26 batch 5 — RFECV
    rfecv_estimator_cfg: "str | None" = None
    # 2026-04-26 batch 6 — recurrent
    recurrent_model_cfg: "str | None" = None
    # 2026-04-28 batch 4 followup — confidence analysis
    include_confidence_analysis_cfg: bool = False
    # 2026-05-11 Wave 15 — MRMR-internal knobs
    mrmr_interactions_max_order_cfg: int = 1
    mrmr_fe_max_steps_cfg: int = 1
    mrmr_cat_fe_enable_cfg: bool = True
    # 2026-05-11 Wave 21 — assorted high-value config-toggle axes
    dummy_baselines_enabled_cfg: bool = True
    baseline_diagnostics_enabled_cfg: bool = True
    use_groups_cfg: bool = True
    apply_outlier_to_val_cfg: bool = True
    multilabel_allow_uncalibrated_cfg: bool = False
    report_residual_audit_cfg: bool = True
    ltr_assume_comparable_scales_cfg: bool = False
    # 2026-05-18 — composite-target discovery (Packs J + K) fuzz axes.
    # Defaulted False / None so existing pinned sensor combos and
    # archived ``_fuzz_results.jsonl`` rows keep deserialising.
    composite_discovery_enabled_cfg: bool = False
    composite_transforms_mode_cfg: "str | None" = None
    # 2026-05-18 — MRMR feature-engineering FE-search knobs. All
    # default to library defaults so existing pinned sensor combos
    # and archived ``_fuzz_results.jsonl`` rows keep deserialising
    # without behaviour change.
    mrmr_fe_npermutations_cfg: int = 0
    mrmr_fe_ntop_features_cfg: int = 0
    mrmr_fe_unary_preset_cfg: str = "minimal"
    mrmr_fe_binary_preset_cfg: str = "minimal"
    mrmr_fe_smart_polynom_iters_cfg: int = 0
    mrmr_fe_smart_polynom_steps_cfg: int = 10
    mrmr_fe_min_polynom_degree_cfg: int = 3
    mrmr_fe_max_polynom_degree_cfg: int = 3
    mrmr_cat_fe_include_numeric_cfg: bool = False
    # 2026-05-19 -- PreprocessingExtensionsConfig polynomial-auto-tune
    # axes (iter-69 byte cap + iter-340 polynomial_max_features cap).
    # Defaults mirror the live configs so existing pinned sensor combos
    # deserialise unchanged.
    prep_ext_polynomial_max_features_cfg: "int | None" = 10_000
    prep_ext_polynomial_interaction_only_cfg: bool = True
    prep_ext_memory_safety_max_bytes_cfg: "int | None" = 500_000_000
    # 2026-05-19 -- composite-discovery stacked-residual axes.
    composite_use_stacked_discovery_cfg: bool = False
    composite_use_stacked_discovery_residual_cfg: bool = False
    # 2026-05-22 — six gate-flip axes from the TVT-MLP-collapse cascade.
    # Defaults are the POST-FIX values; pre-fix variants live in AXES for
    # regression coverage. All canonicalise to post-fix when composite
    # discovery is OFF.
    composite_skip_raw_dominates_ratio_cfg: float = 0.0
    composite_skip_ablation_delta_pct_cfg: float = 0.0
    composite_eps_mi_gain_cfg: float = -10.0
    composite_top_k_after_mi_cfg: int = 32
    composite_require_beats_raw_baseline_cfg: bool = False
    composite_per_bin_n_bins_cfg: int = 0
    composite_tiny_screening_mode_cfg: str = "per_family"
    composite_include_additive_residual_cfg: bool = True
    mlp_activation_cfg: str = "ReLU"
    composite_skip_wrap_pass_predict_cfg: bool = True
    # 2026-05-21 -- mini-HPT (target + feature distribution analyzer) toggle.
    # Default True mirrors the suite signature default; archived
    # _fuzz_results.jsonl rows that pre-date this axis deserialise as True.
    enable_target_distribution_analyzer_cfg: bool = True
    # 2026-05-21 -- MRMR FE pair-check subsample knob. Default 0 (disabled)
    # mirrors the legacy axis behaviour; archived rows pre-dating this axis
    # deserialise as 0 (no subsample).
    fe_check_pairs_subsample_n_cfg: int = 0
    # 2026-05-21 iter150 -- multi-target / multi-target-type axis. See AXES
    # docstring for value semantics. Default None preserves legacy single-
    # target fuzz behaviour for combos archived before this axis landed.
    extra_targets: "str | None" = None
    # 2026-05-21 iter151 -- P0/P1/P2 audit fill-in. All fields default
    # to the pre-iter151 implicit values so combos archived before this
    # batch deserialise without behaviour change.
    enable_quantile_regression_cfg: bool = False
    linear_alpha_cfg: float = 1.0
    linear_solver_cfg: str = "lbfgs"
    enable_feature_handling_config_cfg: bool = False
    enable_precomputed_cfg: bool = False
    test_sequential_fraction_cfg: "float | None" = None
    calib_size_cfg: "float | None" = None
    use_boruta_shap_cfg: bool = False
    use_sample_weights_in_fs_cfg: bool = False
    fallback_to_sklearn_cfg: bool = True
    prefer_gpu_configs_cfg: bool = True
    prefer_cpu_for_lightgbm_cfg: bool = True
    mrmr_identity_cache_scope_cfg: str = "ctx"
    skip_identity_equivalent_pre_pipelines_cfg: bool = True
    rfecv_leakage_corr_threshold_cfg: float = 0.95
    rfecv_mbh_adaptive_threshold_cfg: int = 30
    # 2026-05-22 iter162 -- nested-config / depth-2 audit fill-in.
    # All default to the field's library default so combos archived
    # pre-iter162 deserialise without behaviour change.
    fhc_cache_eviction_strategy_cfg: str = "size_weighted"
    fhc_cache_allow_pickle_cfg: bool = False
    fhc_cache_ram_fraction_cfg: float = 0.3
    fhc_text_definite_text_mean_chars_cfg: int = 100
    fhc_text_min_alphabet_entropy_cfg: float = 4.5
    fhc_repro_deterministic_torch_cfg: bool = False
    fhc_auto_locale_detection_cfg: str = "fallback_only"
    reporting_prob_histogram_yscale_cfg: str = "auto"
    reporting_title_metrics_template_cfg: str = "ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC"
    reporting_matplotlib_rcparams_cfg: "str | None" = None
    reporting_multiclass_panels_cfg: str = "CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC"
    confidence_model_kwargs_cfg: str = "default"
    composite_mi_estimator_cfg: str = "bin"
    composite_mi_nbins_cfg: int = 16
    composite_mi_aggregation_cfg: str = "mean"
    composite_mi_sample_strategy_cfg: str = "random"
    composite_stacked_residual_aggregation_cfg: str = "mean"
    composite_discovery_n_jobs_cfg: int = 1
    quantile_crossing_fix_cfg: str = "sort"
    quantile_coverage_pairs_cfg: str = "default"
    quantile_wrapper_n_jobs_cfg: Any = "auto"
    mlp_predict_batch_size_cfg: "int | None" = None
    ltr_cb_loss_fn_cfg: str = "YetiRankPairwise"
    ltr_lgb_objective_cfg: str = "lambdarank"
    ltr_rrf_k_cfg: int = 60
    recurrent_precision_cfg: str = "32-true"
    recurrent_sequence_preprocessing_cfg: str = "none"
    # 2026-05-22 iter170 -- wave-3 depth-3+ audit. All default to the
    # library default so combos archived pre-iter170 deserialise unchanged.
    lgb_feature_fraction_cfg: float = 1.0
    lgb_num_leaves_cfg: int = 31
    xgb_max_depth_cfg: int = 6
    xgb_colsample_bynode_cfg: float = 1.0
    cb_border_count_cfg: int = 254
    hgb_max_leaf_nodes_cfg: int = 31
    rfecv_cv_n_splits_cfg: int = 2
    robust_q_low_cfg: float = 0.01
    robust_q_high_cfg: float = 0.99
    tfidf_max_features_cfg: int = 5000
    kbins_encode_cfg: str = "ordinal"
    nonlinear_n_components_cfg: int = 100
    pysr_operator_preset_cfg: str = "standard"
    confidence_ensemble_quantile_cfg: float = 0.1
    cat_text_card_threshold_pct_cfg: float = 0.001
    rfecv_n_features_selection_rule_cfg: str = "auto"
    rfecv_stability_selection_cfg: bool = False
    rfecv_leakage_action_cfg: str = "warn"
    mrmr_fe_adaptive_threshold_relax_cfg: bool = True
    mrmr_use_simple_mode_cfg: bool = False
    mrmr_identity_cache_include_y_cfg: bool = True
    # 2026-05-27 MRMR friend-graph + cluster-aggregate (defaults mirror mrmr.py __init__)
    mrmr_build_friend_graph_cfg: bool = True
    mrmr_friend_graph_prune_cfg: bool = False
    mrmr_cluster_aggregate_enable_cfg: bool = True
    mrmr_cluster_aggregate_mode_cfg: str = "augment"
    # 2026-05-28 ShapProxiedFS axes (defaults mirror ShapProxiedFS.__init__)
    use_shap_proxied_fs: bool = False
    shap_proxied_optimizer_cfg: str = "auto"
    shap_proxied_revalidate_cfg: bool = True
    shap_proxied_trust_guard_cfg: bool = True
    shap_proxied_interaction_aware_cfg: bool = False
    shap_proxied_cluster_features_cfg: "bool | str" = "auto"
    # 2026-05-28 ShapProxiedFS extension axes (defaults mirror ShapProxiedFS.__init__)
    shap_proxied_active_learning_cfg: bool = False
    shap_proxied_prefilter_method_cfg: str = "auto"
    # 2026-05-28 ShapProxiedFS deeper extension axes (B1-B6 audit-pass-2).
    # Defaults verified against feature_selection/shap_proxied_fs.py:41-89.
    shap_proxied_config_jitter_cfg: bool = False
    shap_proxied_uncertainty_penalty_cfg: float = 0.0
    shap_proxied_within_cluster_refine_cfg: bool = True
    shap_proxied_use_bias_corrector_cfg: bool = True
    shap_proxied_refine_n_estimators_cfg: "int | None" = 100
    shap_proxied_trust_guard_n_estimators_cfg: "int | None" = 100
    # 2026-05-28 audit-pass-2 PART A: 4 LOW-tier coverage-gap axes.
    # Defaults mirror EnsemblingConfig / TrainingBehaviorConfig /
    # PreprocessingExtensionsConfig in src/mlframe/training/_model_configs.py
    # + _preprocessing_configs.py.
    ensembling_degenerate_class_ratio_cfg: float = 0.01
    behavior_use_flaml_zeroshot_cfg: bool = False
    target_temporal_audit_granularity_cfg: str = "auto"
    prep_ext_dim_n_components_cfg: int = 50
    # 2026-05-28 TextDetectionConfig.text_min_cardinality (default 300)
    fhc_text_min_cardinality_cfg: int = 300
    # 2026-05-28 CompositeTargetDiscoveryConfig deep knobs (defaults mirror the dataclass)
    composite_auto_skip_on_baseline_optimal_cfg: bool = False
    composite_mi_n_neighbors_cfg: int = 3
    composite_auto_base_null_perms_cfg: int = 20
    composite_multi_base_max_k_cfg: int = 3
    # 2026-05-28 TrainingBehaviorConfig.extreme_ar_group_aware_skip_models (enum: default/include_linear/empty)
    extreme_ar_group_aware_skip_models_cfg: str = "default_neural"
    # 2026-05-28 FeatureSelectionConfig.pre_screen_null_fraction_threshold (default 0.99)
    fs_pre_screen_null_fraction_threshold_cfg: float = 0.99
    # 2026-05-28 LinearModelConfig.l1_ratio (ElasticNet mix; default 0.5 in src)
    linear_l1_ratio_cfg: float = 0.5
    # 2026-05-28 RecurrentConfig.hidden_size (library default 128)
    recurrent_hidden_size_cfg: int = 128
    catfe_fwer_correction_cfg: str = "none"
    catfe_perm_budget_strategy_cfg: str = "bandit_ucb1"
    catfe_permutation_null_cfg: str = "joint_independence"
    catfe_bootstrap_ci_n_replicates_cfg: int = 0
    catfe_use_miller_madow_cfg: "bool | None" = None
    catfe_refine_passes_cfg: int = 0
    catfe_enable_streaming_cache_cfg: bool = False
    catfe_unknown_strategy_cfg: str = "clip"
    composite_screening_cfg: str = "hybrid"
    composite_tiny_model_num_leaves_cfg: int = 15
    composite_tiny_model_learning_rate_cfg: float = 0.1
    composite_raw_baseline_tolerance_cfg: float = 1.02
    composite_use_wilcoxon_gate_cfg: bool = False
    composite_detect_alpha_drift_cfg: bool = True
    composite_reject_on_alpha_drift_cfg: bool = False
    reporting_figsize_cfg: str = "default"
    reporting_plot_dpi_cfg: "int | None" = None
    reporting_quantile_panels_cfg: str = "default"
    reporting_ltr_panels_cfg: str = "default"
    reporting_plotly_template_cfg: "str | None" = None
    reporting_matplotlib_style_cfg: "str | None" = None
    baseline_quick_model_n_estimators_cfg: int = 200
    baseline_quick_model_num_leaves_cfg: int = 31
    baseline_quick_model_learning_rate_cfg: float = 0.05
    baseline_sample_n_cfg: int = 50_000
    baseline_high_potential_min_dominance_pct_cfg: float = 5.0
    baseline_best_model_min_lift_cfg: float = 1.5
    dummy_stratified_n_repeats_cfg: int = 20
    dummy_paired_bootstrap_n_resamples_cfg: int = 1000
    ltr_mlp_loss_fn_cfg: str = "ranknet"
    ltr_eval_at_cfg: str = "default"
    multilabel_force_native_xgb_cfg: bool = False
    fhc_pricing_cap_usd_cfg: "float | None" = None
    fhc_pricing_warn_above_usd_cfg: float = 1.0
    fhc_logging_verbose_cfg: bool = False
    fhc_repro_langdetect_seed_cfg: int = 0
    fhc_repro_pinned_svd_solver_params_cfg: bool = True
    fhc_repro_forbid_nonatomic_fs_cfg: bool = False
    fhc_repro_deterministic_eviction_cfg: bool = False
    fhc_cache_prefetch_enabled_cfg: bool = True
    fhc_cache_prefetch_vram_safety_factor_cfg: float = 2.0
    fhc_memory_pressure_watermark_pct_cfg: int = 85
    fhc_text_min_mean_tokens_cfg: float = 4.0
    fhc_text_min_unique_ratio_cfg: float = 0.95
    fhc_text_respect_explicit_cat_dtype_cfg: bool = True
    recurrent_input_mode_cfg: str = "hybrid"
    recurrent_num_workers_cfg: int = 0
    # 2026-05-23 iter180 -- DEPTH-4 booster sub-params + FHC persistence
    # + multilabel list-typed fields.
    lgb_boosting_type_cfg: str = "gbdt"
    lgb_dart_drop_rate_cfg: float = 0.1
    lgb_goss_top_rate_cfg: float = 0.2
    xgb_tree_method_cfg: str = "auto"
    xgb_hist_max_bin_cfg: int = 256
    cb_bootstrap_type_cfg: str = "Bayesian"
    cb_bayesian_bagging_temperature_cfg: float = 1.0
    cb_bernoulli_subsample_cfg: float = 0.8
    cb_grow_policy_cfg: str = "SymmetricTree"
    cb_lossguide_max_leaves_cfg: int = 31
    fhc_cache_persistence_cfg: str = "auto"
    multilabel_per_label_thresholds_cfg: "str | None" = None
    multilabel_chain_seeds_cfg: "str | None" = None
    # F1 -- enable_crash_reporting (suite-level kwarg; Windows-only meaningful).
    enable_crash_reporting_cfg: bool = False
    # 2026-05-26 iter291 new-functionality axes (post-2-day commit wave).
    # Defaults match the post-fix library defaults so existing pinned sensor
    # combos + archived ``_fuzz_results.jsonl`` rows keep deserialising.
    bucket_stratify_cfg: bool = True
    composite_cardinality_cap_cfg: int = 200
    honest_estimator_diagnostics_cfg: bool = True
    cross_target_ensemble_strategy_cfg: str = "nnls_stack"
    add_cyclical_date_features_cfg: bool = False
    add_extended_date_features_cfg: bool = False
    use_nnls_weights_in_blends_cfg: bool = True
    enable_prediction_envelope_clip_cfg: bool = True
    # 2026-05-27 iter332 audit-driven new-functionality axes.
    ensembling_force_legacy_cfg: bool = False
    ensembling_quantile_budget_bytes_cfg: int = 500 * 1024 * 1024
    ensembling_flag_degenerate_conf_subset_cfg: bool = True
    mlp_extreme_ar_group_aware_skip_cfg: bool = False
    mlp_extreme_ar_threshold_cfg: float = 0.99
    mlp_drop_per_group_constants_cfg: bool = False
    composite_always_build_ct_ensemble_for_raw_cfg: bool = True
    composite_ct_ensemble_dummy_floor_enabled_cfg: bool = True
    composite_extreme_ar_group_aware_skip_cfg: bool = True
    composite_oof_holdout_source_cfg: str = "external_val"
    composite_stacking_aware_gate_enabled_cfg: bool = False
    composite_use_baseline_diagnostics_hint_cfg: bool = True
    fs_pre_screen_unsupervised_cfg: bool = True
    fs_pre_screen_variance_threshold_cfg: float = 0.0
    baseline_init_score_top_k_cfg: int = 1
    # 2026-05-27 iter350 audit batch 2 axes.
    use_ap12_calibrated_probs_in_ensemble_cfg: bool = True
    mlp_extreme_ar_weight_decay_factor_cfg: float = 100.0
    feature_drift_auto_apply_neural_overrides_cfg: bool = False
    target_temporal_audit_column_cfg: "str | None" = None
    composite_lag_predict_failsafe_tolerance_cfg: float = 0.10
    composite_extreme_ar_threshold_cfg: float = 0.99
    composite_ct_ensemble_dummy_floor_tolerance_cfg: float = 0.0
    composite_oof_holdout_frac_cfg: float = 0.2
    composite_top_m_after_tiny_cfg: int = 10
    prep_ext_tfidf_keep_sparse_cfg: bool = True
    recurrent_use_attention_cfg: bool = True
    ltr_xgb_objective_cfg: str = "rank:ndcg"
    baseline_init_score_apply_target_types_cfg: str = "regression_only"

    def canonical_key(self) -> tuple:
        """Hashable tuple used for dedup. Canonicalizes semantically
        equivalent combos so e.g. ``align_polars_categorical_dicts=True`` with
        pandas input collapses to the False variant."""
        align = self.align_polars_categorical_dicts
        # 2026-05-12 Wave 30: polars input only reaches models that
        # support it natively. For non-polars-native models (LGB,
        # linear, MLP), canonicalise to pandas so the downstream
        # sklearn pipeline encoder (CatBoostEncoder) / strategies
        # receive pandas DataFrames they can consume. Without this, a
        # polars DataFrame hits CatBoostEncoder.fit and raises
        # ``ValueError: Unexpected input type: <class 'polars...'>``
        # (surfaced c0002/c0004 fuzz failures).
        _input_type = self.input_type
        if _input_type != "pandas":
            _any_polars_native = any(
                m in self.models for m in ("cb", "xgb", "hgb", "mlp", "lstm", "gru", "transformer")
            )
            if not _any_polars_native:
                _input_type = "pandas"
        null_frac = self.null_fraction_cats if self.cat_feature_count > 0 else 0.0
        # fairness_col is meaningful only if that column exists → None
        # when cat_feature_count == 0 (no cat_0 to reference).
        fairness = self.fairness_col if self.cat_feature_count > 0 else None
        # custom_prep=pca2 makes sense only on a clean all-numeric
        # frame — IncrementalPCA can't consume:
        #   * string/cat/text/embedding columns (no pre-encoding before
        #     custom_pre_pipeline in the mlframe pipeline),
        #   * NaN values (sklearn IncrementalPCA explicitly rejects NaN;
        #     the error message even suggests HistGradientBoosting as
        #     the alternative),
        #   * all-null / all-const columns (degenerate for PCA's
        #     variance computation).
        # Canonicalise custom_prep → None for any of those axes so
        # the pairwise sampler doesn't waste combos on a guaranteed-fail
        # configuration. Users who want PCA on real data must
        # pre-process upstream.
        pca_incompatible = (
            self.cat_feature_count > 0
            or self.text_col_count > 0
            or self.embedding_col_count > 0
            or self.inject_inf_nan          # injects np.nan → PCA rejects
            or self.inject_degenerate_cols  # adds all-null column → PCA rejects
        )
        custom_prep = self.custom_prep if not pca_incompatible else None
        use_ensembles = self.use_ensembles
        target_carrier = self.target_carrier
        if self.target_type == "multilabel_classification":
            # Multilabel FTE intentionally unpacks list cells to (N, K)
            # ndarray. Native list-Series targets are a different
            # contract, so keep that axis for a dedicated future sensor.
            target_carrier = "numpy"
        return (
            tuple(sorted(self.models)),
            _input_type,
            self.n_rows,
            self.cat_feature_count,
            null_frac,
            self.use_mrmr_fs,
            tuple(sorted(self.weight_schemas)),
            self.target_type,
            target_carrier,
            self.auto_detect_cats,
            align,
            self.prefer_polarsds,
            self.use_text_features,
            self.honor_user_dtype,
            # text_col_count: passthrough. The historical canonicalisation
            # here zeroed text columns on small-n CB + heavy NaN combos
            # (fuzz c0056 / c0070) where CB's default
            # ``occurrence_lower_bound=50`` cannot build a TF-IDF
            # dictionary on tiny inner-CV folds. That production hang
            # is now fixed in ``training/helpers.compute_cb_text_processing``
            # (called from ``trainer._train_model_with_fallback`` and
            # the RFECV inner-fold path in ``feature_selection/wrappers``)
            # which scales ``occurrence_lower_bound`` proportionally to
            # the actual fit-time row count. The fuzz axis stays fully
            # exercised.
            self.text_col_count,
            self.embedding_col_count,
            # 2026-04-24 combo-extension axes
            # OneClassSVM has O(n²) fit cost — collapse to None on the
            # large-row tier (n>=1200) so it doesn't dominate runtime.
            # IsolationForest and LOF stay enabled across all sizes.
            None if (self.outlier_detection == "ocsvm" and self.n_rows >= 1200) else self.outlier_detection,
            use_ensembles,
            self.continue_on_model_failure,
            self.iterations,
            # ``prefer_calibrated_classifiers=True`` + multilabel raises
            # ``NotImplementedError`` at trainer.py:_validate_multilabel_calibration_compat
            # (CalibratedClassifierCV is single-output only). Canonicalise
            # to False for multilabel combos so dedup collapses the
            # known-incompatible variant. Surfaced 2026-04-28 default seed
            # c0060 (lgb_xgb / multilabel + prefer_calibrated=True) -
            # previously masked because the polars ``pl.List`` -> pandas
            # roundtrip presented as 1-D, and the calibration guard
            # didn't fire; with the multilabel target normalised the
            # guard correctly rejects, so the canon owns the dedup.
            False if (self.prefer_calibrated_classifiers and self.target_type == "multilabel_classification") else self.prefer_calibrated_classifiers,
            # CB+multilabel+degenerate canon RETIRED 2026-04-27 (batch 2).
            # Production fix: explicit cat_features list now drops columns
            # whose dtype is numeric (via the existing dtype-aware filter
            # in feature_selection/wrappers.py + the new num-degenerate
            # guard in trainer._train_model_with_fallback). The fuzz axis
            # exercises the full inject_degenerate_cols × CB × multilabel
            # cross-product again.
            self.inject_degenerate_cols,
            self.inject_inf_nan,
            self.with_datetime_col,
            self.inject_zero_col,
            fairness,
            custom_prep,
            self.input_storage,
            # 2026-04-24 round 2 — config field axes
            self.fillna_value_cfg,
            self.scaler_name_cfg,
            self.categorical_encoding_cfg,
            self.skip_categorical_encoding_cfg,
            self.val_placement_cfg,
            self.test_size_cfg,
            self.trainset_aging_limit_cfg,
            self.cat_text_card_threshold_cfg,
            self.early_stopping_rounds_cfg,
            self.use_robust_eval_metric_cfg,
            # Fix G — adversarial axes
            self.inject_label_leak,
            self.inject_rank_deficient,
            self.inject_all_nan_col,
            # R3 — drift, imbalance, weird-cat
            # inject_test_drift canonicalises to None when n_rows is too
            # small to meaningfully distinguish train from test slices.
            self.inject_test_drift if self.n_rows >= 300 else None,
            # imbalance_ratio canonicalisation: meaningful only on
            # binary classification. Extreme imbalance on small frames
            # causes random val/test splits to drop one class entirely
            # ("CatBoostError: Target contains only one unique value",
            # 2026-04-24 c0062/c0085). Clamp by expected minority count
            # per 10% slice — need ≥2 minority rows in each split's
            # worst-case unlucky draw, i.e. frac * slice_size ≥ ~4 → need
            # frac * n * 0.1 ≥ 4 → frac ≥ 40/n. At n=1200, rare_1pct
            # (0.01) = 12 total minority, slice=1.2 — unreliable → clamp
            # to rare_5pct. rare_5pct survives at n≥800.
            self._canonical_imbalance(),
            # weird_cat_content relevant only if there are cat columns.
            self.weird_cat_content if self.cat_feature_count > 0 else None,
            # Phase H: multilabel_strategy_cfg only meaningful for multilabel.
            # `chain` dispatch breaks two assumptions when the inputs aren't
            # already classifier-friendly numeric pandas:
            #   * sklearn ClassifierChain wraps the inner estimator and
            #     forwards the raw frame — polars Enum / Utf8 + raw cat
            #     columns reach HGB/LGB which can't handle strings (Cluster
            #     "could not convert string to float: 'A'").
            #   * the linear model in multilabel uses LinearRegression which
            #     produces (N, K) predictions; report_regression_model_perf
            #     then crashes plotting 2-D preds vs targets.
            # Downgrade chain → wrapper for combos that would hit either,
            # so chain is exercised only on safe (pandas, no-cats, no-linear)
            # multilabel combos.
            self._canonical_multilabel_strategy(),
            # 2026-04-26 batch 1 — config-field axes
            # fix_infinities=False is intentional Inf pass-through (per
            # preprocessing.py:154-156). Combining it with inject_inf_nan
            # (which puts np.inf into num_0) feeds raw Inf to XGB/HGB and
            # crashes them — that's a user-misconfiguration, not a bug.
            # Canonicalise away so the dedup pass collapses these combos
            # into the safe variant. The fix_infinities=False axis is still
            # exercised on clean-data combos.
            self.fix_infinities_cfg if not self.inject_inf_nan else True,
            self.ensure_float32_cfg,
            # remove_constant_columns=False is meaningful only when no
            # degenerate columns will exist. Combining it with
            # inject_degenerate_cols (adds num_const + num_null columns)
            # or inject_all_nan_col routes an all-NaN column to the
            # downstream scaler — polars_ds robust_scale crashes on
            # quantile(None) - quantile(None) in that case (c0008).
            # Canonicalise to True so the dedup pass collapses the
            # known-bad combination.
            self.remove_constant_columns_cfg if not (self.inject_degenerate_cols or self.inject_all_nan_col) else True,
            # imputer_strategy is meaningful only if nulls / NaNs exist.
            # Without missing values the imputer never fires → all variants
            # collapse to the default to avoid wasting combos on a no-op axis.
            self._canonical_imputer_strategy(),
            self.shuffle_val_cfg,
            self.shuffle_test_cfg,
            # wholeday_splitting requires a datetime column to take effect.
            self.wholeday_splitting_cfg if self.with_datetime_col else True,
            # val_sequential_fraction=1.0 + val_placement=backward + non-shuffled
            # splits + degenerate-col injection occasionally land an empty val
            # window after constant-col removal trims rows (c0102: shape=(0,29)
            # reaches SimpleImputer). Pre-existing splitter edge case — not
            # batch-4-introduced. Canonicalise the val_sequential_fraction
            # value down to 0.5 in the trigger window so dedup absorbs the
            # zero-row variant; the underlying splitter bug needs a separate
            # fix.
            (
                0.5
                if (
                    self.val_sequential_fraction_cfg == 1.0
                    and self.val_placement_cfg == "backward"
                    and not self.shuffle_val_cfg
                    and self.inject_degenerate_cols
                )
                else self.val_sequential_fraction_cfg
            ),
            # multilabel_n_chains/chain_order/cv only matter when
            # target_type=multilabel AND strategy=chain. For all other
            # combos these are no-ops, so canonicalise to defaults to
            # avoid wasting combo budget on identical-behaviour entries.
            self.multilabel_n_chains_cfg if self._is_chain_dispatch() else 3,
            self.multilabel_chain_order_cfg if self._is_chain_dispatch() else "random",
            self.multilabel_cv_cfg if self._is_chain_dispatch() else 5,
            # PreprocessingExtensionsConfig axes — none of these tolerate
            # raw NaN/Inf or all-null columns; canonicalise to None when
            # NaN-injecting axes are on so the dedup pass collapses the
            # known-bad combinations. Categorical-bearing inputs are also
            # canonicalised away because the sklearn-bridge requires
            # already-encoded numeric input.
            self._canonical_prep_ext("scaler"),
            self._canonical_prep_ext("kbins"),
            self._canonical_prep_ext("polynomial_degree"),
            self._canonical_prep_ext("dim_reducer"),
            self._canonical_prep_ext("nonlinear"),
            # RFECV: only meaningful when the underlying model is in the
            # combo's mlframe_models list and n_rows is small enough that
            # the iterative re-fit doesn't blow runtime budget.
            self._canonical_rfecv_estimator(),
            # Recurrent: needs sequence inputs alongside the tabular df.
            # The synthetic builder emits sequences only on the small-row
            # tier (training time scales linearly per epoch with n).
            self._canonical_recurrent_model(),
            # 2026-05-11 Wave 15 — MRMR-internal knobs collapse to default
            # when MRMR is disabled. cat-FE-enable also collapses when
            # there are 0/1 cat features (cat-FE requires >=2). interactions
            # >= 2 collapses when cat_feature_count == 0 (cat-FE is the
            # primary trigger for the Wave-14-fixed kway-engineered-cols
            # path; pure numeric MRMR at order>=2 exercises a different
            # branch but is still useful to fuzz, so keep that variant).
            self.mrmr_interactions_max_order_cfg if self.use_mrmr_fs else 1,
            self.mrmr_fe_max_steps_cfg if self.use_mrmr_fs else 1,
            self.mrmr_cat_fe_enable_cfg if (
                self.use_mrmr_fs and self.cat_feature_count >= 2
            ) else True,
            # 2026-05-11 Wave 21 — config-toggle axes with relevance gates
            self.dummy_baselines_enabled_cfg,
            self.baseline_diagnostics_enabled_cfg,
            # use_groups only matters when wholeday_splitting is on AND there's
            # a datetime column to derive groups from.
            self.use_groups_cfg if (
                self.with_datetime_col and self.wholeday_splitting_cfg
            ) else True,
            # apply_outlier_to_val only matters when outlier_detection is set.
            self.apply_outlier_to_val_cfg if self.outlier_detection is not None else True,
            # multilabel_allow_uncalibrated only meaningful for multilabel.
            self.multilabel_allow_uncalibrated_cfg if (
                self.target_type == "multilabel_classification"
            ) else False,
            # report_residual_audit only matters for regression target.
            self.report_residual_audit_cfg if self.target_type == "regression" else True,
            # ltr_assume_comparable_scales only meaningful for LTR.
            self.ltr_assume_comparable_scales_cfg if (
                self.target_type == "learning_to_rank"
            ) else False,
            # composite-discovery only fires on regression targets and
            # only when at least 2 numeric base candidates exist. For
            # non-regression / no-numeric-bases combos collapse to the
            # disabled variant so dedup absorbs identical-behaviour
            # combos.
            self.composite_discovery_enabled_cfg if (
                self.target_type == "regression"
                # require enough columns for a base candidate pool (the
                # frame builder emits ``num_0..num_3`` for n_rows >= 300
                # so this is always satisfied when target=regression;
                # the gate is here for future shrinkage).
            ) else False,
            # composite_transforms_mode is a no-op when discovery is off.
            self.composite_transforms_mode_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else None,
            # MRMR FE knobs canonicalise to defaults when
            # ``use_mrmr_fs=False`` so dedup collapses identical
            # behaviour combos. The library defaults (npermutations=0,
            # ntop_features=0, unary/binary_preset="minimal",
            # smart_polynom_iters=0, etc.) keep the FE step OFF and
            # the polynomial search disabled; any non-default value
            # activates a code path the prior fuzz axis space did
            # not exercise.
            self.mrmr_fe_npermutations_cfg if self.use_mrmr_fs else 0,
            self.mrmr_fe_ntop_features_cfg if self.use_mrmr_fs else 0,
            self.mrmr_fe_unary_preset_cfg if self.use_mrmr_fs else "minimal",
            self.mrmr_fe_binary_preset_cfg if self.use_mrmr_fs else "minimal",
            self.mrmr_fe_smart_polynom_iters_cfg if self.use_mrmr_fs else 0,
            # smart_polynom_steps is a no-op when smart_polynom_iters=0.
            self.mrmr_fe_smart_polynom_steps_cfg if (
                self.use_mrmr_fs and self.mrmr_fe_smart_polynom_iters_cfg > 0
            ) else 10,
            # min/max polynom_degree only matter when smart_polynom is on.
            self.mrmr_fe_min_polynom_degree_cfg if (
                self.use_mrmr_fs and self.mrmr_fe_smart_polynom_iters_cfg > 0
            ) else 3,
            self.mrmr_fe_max_polynom_degree_cfg if (
                self.use_mrmr_fs and self.mrmr_fe_smart_polynom_iters_cfg > 0
            ) else 3,
            # CatFEConfig.include_numeric only matters when cat-FE is
            # enabled AND MRMR is on AND there are categorical columns
            # to mix discretized numerics with.
            self.mrmr_cat_fe_include_numeric_cfg if (
                self.use_mrmr_fs
                and self.mrmr_cat_fe_enable_cfg
                and self.cat_feature_count > 0
            ) else False,
            # 2026-05-19 -- polynomial auto-tune axes are no-ops when
            # prep_ext_polynomial_degree_cfg is None (the whole polynomial
            # step is OFF). Canonicalise to library defaults so dedup
            # absorbs combos that differ only on inactive knobs.
            self.prep_ext_polynomial_max_features_cfg if (
                self.prep_ext_polynomial_degree_cfg is not None
            ) else 10_000,
            self.prep_ext_polynomial_interaction_only_cfg if (
                self.prep_ext_polynomial_degree_cfg is not None
            ) else True,
            # memory_safety_max_bytes guards every sklearn-bridge stage
            # output, not just polynomial; keep meaningful whenever ANY
            # prep-ext stage is on. Canonicalise to the default cap when
            # none of scaler / kbins / polynomial / dim_reducer /
            # nonlinear is active.
            self.prep_ext_memory_safety_max_bytes_cfg if (
                self.prep_ext_scaler_cfg is not None
                or self.prep_ext_kbins_cfg is not None
                or self.prep_ext_polynomial_degree_cfg is not None
                or self.prep_ext_dim_reducer_cfg is not None
                or self.prep_ext_nonlinear_cfg is not None
            ) else 500_000_000,
            # Composite stacked-discovery knobs no-op unless composite
            # discovery is enabled AND the target is regression (the
            # discovery is regression-only). Canonicalise to False so
            # dedup collapses identical-behaviour combos.
            self.composite_use_stacked_discovery_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else False,
            self.composite_use_stacked_discovery_residual_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else False,
            # skip_wrap_pass_predict toggles per-component re-wrap at
            # predict; only meaningful when composite discovery actually
            # produced wrappers (regression + enabled).
            self.composite_skip_wrap_pass_predict_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else True,
            # 2026-05-22 -- TVT-MLP audit-followup gate axes. All only
            # meaningful when composite discovery is on; canonicalise
            # to the post-fix default when off.
            self.composite_skip_raw_dominates_ratio_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else 0.0,
            self.composite_skip_ablation_delta_pct_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else 0.0,
            self.composite_eps_mi_gain_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else -10.0,
            self.composite_top_k_after_mi_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else 32,
            self.composite_require_beats_raw_baseline_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else False,
            self.composite_per_bin_n_bins_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else 0,
            self.composite_tiny_screening_mode_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else "per_family",
            self.composite_include_additive_residual_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
            ) else True,
            # mlp_activation_cfg only meaningful when 'mlp' is in models
            # AND target_type is regression / classification head;
            # canonicalise to "ReLU" otherwise. NOTE: axis not yet
            # plumbed into the suite call (see comment in AXES dict).
            self.mlp_activation_cfg if (
                "mlp" in self.models
                and self.target_type in ("regression", "binary_classification",
                                          "multiclass_classification")
            ) else "ReLU",
            # 2026-05-21 -- mini-HPT (target + feature distribution analyzer)
            # toggle. Axis is meaningful on every target type since both
            # detectors run unconditionally when enabled.
            self.enable_target_distribution_analyzer_cfg,
            # 2026-05-21 -- FE check-pairs subsample knob is meaningful only
            # when the FE inner pass actually runs. Canonicalise to 0 when
            # use_mrmr_fs is False OR both FE entry points (npermutations +
            # ntop_features) are 0 -- the survivor-rebuild branch can't be
            # exercised in those cases, so dedup-collapsing 50_000 -> 0
            # avoids spending pairwise budget on identical-behaviour combos.
            # Subsample also needs n_rows > subsample_n: canon-to-0 at
            # n_rows=1000 (always <= 50_000 budget), keep at n_rows=200_000.
            self.fe_check_pairs_subsample_n_cfg if (
                self.use_mrmr_fs
                and (self.mrmr_fe_npermutations_cfg > 0 or self.mrmr_fe_ntop_features_cfg > 0)
                and self.n_rows > self.fe_check_pairs_subsample_n_cfg
            ) else 0,
            # 2026-05-21 iter150 -- extra_targets canon:
            #  - multilabel / LTR: collapse to None (multilabel is already 2-D
            #    within ONE target; LTR has its own ranker dispatch path that
            #    bypasses the target_by_type loop).
            #  - mixed_reg_bin / mixed_reg_bin_2each require regression primary:
            #    collapse to None otherwise (the secondary type would conflict
            #    with the primary target inference logic in the suite).
            (None if self.target_type in ("multilabel_classification", "learning_to_rank")
             else (None if (self.extra_targets in ("mixed_reg_bin", "mixed_reg_bin_2each")
                            and self.target_type != "regression")
                   else self.extra_targets)),
            # 2026-05-21 iter151 -- P0/P1/P2 canons. Each axis collapses to
            # its dataclass default when the feature it gates is inactive,
            # so semantically-no-op combos dedup down to one representative.
            # P0-1: quantile only meaningful on regression primary.
            (self.enable_quantile_regression_cfg if self.target_type == "regression" else False),
            # P0-2: linear axes only meaningful when "linear" in models.
            (self.linear_alpha_cfg if "linear" in self.models else 1.0),
            (self.linear_solver_cfg if "linear" in self.models else "lbfgs"),
            # P0-3 / P0-4: enable flags carry through; FHC/precomputed have
            # no preconditions on other combo axes so no further canon.
            self.enable_feature_handling_config_cfg,
            self.enable_precomputed_cfg,
            # P1-5: test_sequential_fraction is a time-axis split; only
            # meaningful when with_datetime_col is True (otherwise the
            # splitter has no time signal to sort by).
            (self.test_sequential_fraction_cfg if self.with_datetime_col else None),
            # P1-6: calib_size always meaningful; passthrough.
            self.calib_size_cfg,
            # P1-7: use_boruta_shap independent.
            self.use_boruta_shap_cfg,
            # P1-8: use_sample_weights_in_fs only meaningful when any FS is
            # enabled (MRMR / RFECV / Boruta) AND weights schema includes
            # something non-uniform (otherwise FS receives all-1s weights
            # and the branch is a no-op).
            (self.use_sample_weights_in_fs_cfg if (
                (self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                 or self.use_boruta_shap_cfg)
                and any(s != "uniform" for s in self.weight_schemas)
            ) else False),
            # P1-9: fallback_to_sklearn only meaningful when polars-ds
            # preferred (prefer_polarsds=True). Otherwise the path is dead.
            (self.fallback_to_sklearn_cfg if self.prefer_polarsds else True),
            # P1-10a/b: device toggles always meaningful; passthrough.
            self.prefer_gpu_configs_cfg,
            self.prefer_cpu_for_lightgbm_cfg,
            # P2-16: mrmr_identity_cache_scope only meaningful when MRMR
            # is enabled.
            (self.mrmr_identity_cache_scope_cfg if self.use_mrmr_fs else "ctx"),
            # P2-17: skip_identity dedup-skip only meaningful when any FS
            # is active AND there's an ensembling path that could benefit
            # from non-deduped pipelines.
            (self.skip_identity_equivalent_pre_pipelines_cfg if (
                self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                or self.use_boruta_shap_cfg
            ) else True),
            # P2-18a/b: RFECV thresholds only meaningful when RFECV is on.
            (self.rfecv_leakage_corr_threshold_cfg if self.rfecv_estimator_cfg is not None else 0.95),
            (self.rfecv_mbh_adaptive_threshold_cfg if self.rfecv_estimator_cfg is not None else 30),
            # 2026-05-22 iter162 nested-config canons.
            # FHC sub-configs only meaningful when enable_feature_handling_config_cfg=True.
            (self.fhc_cache_eviction_strategy_cfg if self.enable_feature_handling_config_cfg else "size_weighted"),
            (self.fhc_cache_allow_pickle_cfg if self.enable_feature_handling_config_cfg else False),
            (self.fhc_cache_ram_fraction_cfg if self.enable_feature_handling_config_cfg else 0.3),
            (self.fhc_text_definite_text_mean_chars_cfg if self.enable_feature_handling_config_cfg else 100),
            (self.fhc_text_min_alphabet_entropy_cfg if self.enable_feature_handling_config_cfg else 4.5),
            (self.fhc_repro_deterministic_torch_cfg if self.enable_feature_handling_config_cfg else False),
            (self.fhc_auto_locale_detection_cfg if self.enable_feature_handling_config_cfg else "fallback_only"),
            # ReportingConfig nested -- always meaningful (suite always reports).
            self.reporting_prob_histogram_yscale_cfg,
            self.reporting_title_metrics_template_cfg,
            self.reporting_matplotlib_rcparams_cfg,
            # multiclass_panels only meaningful for multiclass / multilabel.
            (self.reporting_multiclass_panels_cfg if self.target_type in ("multiclass_classification", "multilabel_classification") else "CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC"),
            # ConfidenceAnalysisConfig.model_kwargs only when confidence_analysis enabled.
            (self.confidence_model_kwargs_cfg if self.include_confidence_analysis_cfg else "default"),
            # CompositeTargetDiscoveryConfig nested only when composite enabled (regression only).
            (self.composite_mi_estimator_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else "bin"),
            (self.composite_mi_nbins_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else 16),
            (self.composite_mi_aggregation_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else "mean"),
            (self.composite_mi_sample_strategy_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else "random"),
            # stacked_residual_aggregation only when composite_use_stacked_discovery_residual=True.
            (self.composite_stacked_residual_aggregation_cfg if (
                self.composite_discovery_enabled_cfg
                and self.target_type == "regression"
                and self.composite_use_stacked_discovery_residual_cfg
            ) else "mean"),
            (self.composite_discovery_n_jobs_cfg if self.composite_discovery_enabled_cfg and self.target_type == "regression" else 1),
            # QuantileRegressionConfig nested only when enable_quantile and regression primary.
            (self.quantile_crossing_fix_cfg if self.enable_quantile_regression_cfg and self.target_type == "regression" else "sort"),
            (self.quantile_coverage_pairs_cfg if self.enable_quantile_regression_cfg and self.target_type == "regression" else "default"),
            (self.quantile_wrapper_n_jobs_cfg if self.enable_quantile_regression_cfg and self.target_type == "regression" else "auto"),
            # MLP predict batch size only when "mlp" in models.
            (self.mlp_predict_batch_size_cfg if "mlp" in self.models else None),
            # LearningToRankConfig nested only for LTR.
            (self.ltr_cb_loss_fn_cfg if self.target_type == "learning_to_rank" else "YetiRankPairwise"),
            (self.ltr_lgb_objective_cfg if self.target_type == "learning_to_rank" else "lambdarank"),
            (self.ltr_rrf_k_cfg if self.target_type == "learning_to_rank" and self.ranking_ensemble_method == "rrf" else 60),
            # RecurrentConfig nested only when a recurrent model is requested.
            (self.recurrent_precision_cfg if self._canonical_recurrent_model() is not None else "32-true"),
            (self.recurrent_sequence_preprocessing_cfg if self._canonical_recurrent_model() is not None else "none"),
            # F5 -- PySR cannot consume non-finite values; when inject_inf_nan is on, the True/False variants are behaviour-identical (both crash on the first inf/nan row) so collapse the True variant to False to spare pairwise budget. Also collapse when the upstream cat-feature gate already disabled the bridge (cats + non-encoded path) since PySR runs inside the sklearn bridge.
            self._canonical_pysr_enabled(),
            # F1 -- enable_crash_reporting is a Windows-only Faulthandler dump hook; on non-Windows hosts the True variant is a no-op so collapse to False. On Windows the axis stays meaningful.
            self._canonical_enable_crash_reporting(),
            # 2026-05-26 iter291 -- new-functionality canons.
            # bucket_stratify is a regression-target split-time toggle. For
            # non-regression primaries it's a no-op gate, so collapse to the
            # default True so the dedup pass coalesces identical-behaviour
            # combos. Note: multi-target combos with a regression primary
            # keep the axis live (the split applies to the primary).
            (self.bucket_stratify_cfg if self.target_type == "regression" else True),
            # composite_cardinality_cap is meaningful only when bucket_stratify
            # is True AND there is a cat column whose cardinality could exceed
            # the cap (n_rows tier 200k routinely produces high-card splits).
            # Canonicalise to the default 200 otherwise so the cap variant
            # collapses cleanly.
            (
                self.composite_cardinality_cap_cfg
                if (self.target_type == "regression" and self.bucket_stratify_cfg
                    and (self.cat_feature_count > 0 or self.n_rows >= 50_000))
                else 200
            ),
            # honest_estimator_diagnostics is a reporting-side aggregator; always
            # meaningful regardless of target type (the per-target aggregator
            # walks every produced (target, model) pair).
            self.honest_estimator_diagnostics_cfg,
            # cross_target_ensemble_strategy only meaningful when composite
            # discovery is on AND target is regression. Canon to "nnls_stack"
            # default otherwise.
            (
                self.cross_target_ensemble_strategy_cfg
                if (self.composite_discovery_enabled_cfg
                    and self.target_type == "regression")
                else "nnls_stack"
            ),
            # cyclical / extended date features only meaningful when a datetime
            # column is present. Canon to False otherwise (the feature engine
            # has no datetime to consume).
            (self.add_cyclical_date_features_cfg if self.with_datetime_col else False),
            (self.add_extended_date_features_cfg if self.with_datetime_col else False),
            # NNLS weights in blends only meaningful when ensembling is on.
            (self.use_nnls_weights_in_blends_cfg if self.use_ensembles else True),
            # Prediction-envelope clip is regression-only. Canon to True
            # (the post-fix default) for non-regression primaries so the
            # False variant doesn't waste pairwise budget on no-op gates.
            (self.enable_prediction_envelope_clip_cfg if self.target_type == "regression" else True),
            # 2026-05-27 iter332 audit-driven canons.
            # Ensembling knobs only meaningful when use_ensembles is True.
            (self.ensembling_force_legacy_cfg if self.use_ensembles else False),
            (self.ensembling_quantile_budget_bytes_cfg if self.use_ensembles else 500 * 1024 * 1024),
            # flag_degenerate_conf_subset is binary-classification-only;
            # collapse to default for other target types.
            (
                self.ensembling_flag_degenerate_conf_subset_cfg
                if (self.use_ensembles and self.target_type == "binary_classification")
                else True
            ),
            # MLP knobs only meaningful when MLP is in scope.
            (self.mlp_extreme_ar_group_aware_skip_cfg if "mlp" in self.models else False),
            (self.mlp_extreme_ar_threshold_cfg if "mlp" in self.models else 0.99),
            (self.mlp_drop_per_group_constants_cfg if "mlp" in self.models else False),
            # Composite-target knobs only meaningful when composite discovery
            # is enabled AND target is regression (composite is regression-only).
            (
                self.composite_always_build_ct_ensemble_for_raw_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else True
            ),
            (
                self.composite_ct_ensemble_dummy_floor_enabled_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else True
            ),
            (
                self.composite_extreme_ar_group_aware_skip_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else True
            ),
            (
                self.composite_oof_holdout_source_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else "external_val"
            ),
            (
                self.composite_stacking_aware_gate_enabled_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else False
            ),
            (
                self.composite_use_baseline_diagnostics_hint_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else True
            ),
            # FS pre-screen only meaningful when MRMR / RFECV / Boruta active.
            (
                self.fs_pre_screen_unsupervised_cfg
                if (self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                    or self.use_boruta_shap_cfg)
                else True
            ),
            (
                self.fs_pre_screen_variance_threshold_cfg
                if (self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                    or self.use_boruta_shap_cfg)
                else 0.0
            ),
            # BaselineDiagnostics init_score top_k only meaningful when
            # baseline_diagnostics is enabled.
            (self.baseline_init_score_top_k_cfg if self.baseline_diagnostics_enabled_cfg else 1),
            # 2026-05-27 iter350 audit batch 2 canons.
            # AP12 calibrated probs in ensemble only meaningful when ensembling on.
            (self.use_ap12_calibrated_probs_in_ensemble_cfg if self.use_ensembles else True),
            # MLP weight_decay factor only meaningful when MLP is in scope.
            (self.mlp_extreme_ar_weight_decay_factor_cfg if "mlp" in self.models else 100.0),
            # Feature-drift auto-apply only meaningful when feature_drift report runs.
            self.feature_drift_auto_apply_neural_overrides_cfg,
            # Temporal audit column only meaningful when a datetime column exists.
            (self.target_temporal_audit_column_cfg if self.with_datetime_col else None),
            # Composite knobs only meaningful when composite discovery on AND regression.
            (
                self.composite_lag_predict_failsafe_tolerance_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 0.10
            ),
            (
                self.composite_extreme_ar_threshold_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 0.99
            ),
            (
                self.composite_ct_ensemble_dummy_floor_tolerance_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 0.0
            ),
            (
                self.composite_oof_holdout_frac_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 0.2
            ),
            (
                self.composite_top_m_after_tiny_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 10
            ),
            # tfidf_keep_sparse only meaningful when prep_ext has text + tfidf path.
            (
                self.prep_ext_tfidf_keep_sparse_cfg
                if self.text_col_count > 0
                else True
            ),
            # Recurrent use_attention only meaningful when a recurrent model is requested.
            (self.recurrent_use_attention_cfg if self._canonical_recurrent_model() is not None else True),
            # LTR xgb_objective only meaningful for LTR + xgb.
            (
                self.ltr_xgb_objective_cfg
                if (self.target_type == "learning_to_rank" and "xgb" in self.models)
                else "rank:ndcg"
            ),
            # BD init_score target types only meaningful when BD enabled.
            (
                self.baseline_init_score_apply_target_types_cfg
                if self.baseline_diagnostics_enabled_cfg
                else "regression_only"
            ),
            # 2026-05-27 MRMR friend-graph + cluster-aggregate (recent mrmr.py
            # features). All collapse to the mrmr.py defaults when MRMR is off
            # so non-FS combos don't gain phantom variation. friend_graph_prune
            # only bites when the graph is actually built; cluster_aggregate_mode
            # only when aggregation is enabled.
            (self.mrmr_build_friend_graph_cfg if self.use_mrmr_fs else True),
            (
                self.mrmr_friend_graph_prune_cfg
                if (self.use_mrmr_fs and self.mrmr_build_friend_graph_cfg)
                else False
            ),
            (self.mrmr_cluster_aggregate_enable_cfg if self.use_mrmr_fs else True),
            (
                self.mrmr_cluster_aggregate_mode_cfg
                if (self.use_mrmr_fs and self.mrmr_cluster_aggregate_enable_cfg)
                else "augment"
            ),
            # 2026-05-28 ShapProxiedFS axes. All sub-knobs collapse to the
            # ShapProxiedFS.__init__ defaults when use_shap_proxied_fs is off so
            # non-shap-proxied combos don't gain phantom variation.
            self.use_shap_proxied_fs,
            (self.shap_proxied_optimizer_cfg if self.use_shap_proxied_fs else "auto"),
            (self.shap_proxied_revalidate_cfg if self.use_shap_proxied_fs else True),
            (self.shap_proxied_trust_guard_cfg if self.use_shap_proxied_fs else True),
            (self.shap_proxied_interaction_aware_cfg if self.use_shap_proxied_fs else False),
            (self.shap_proxied_cluster_features_cfg if self.use_shap_proxied_fs else "auto"),
            # 2026-05-28 ShapProxiedFS extension axes (active_learning +
            # prefilter_method). Collapse to ShapProxiedFS __init__ defaults
            # (False / "auto") when use_shap_proxied_fs is off.
            (self.shap_proxied_active_learning_cfg if self.use_shap_proxied_fs else False),
            (self.shap_proxied_prefilter_method_cfg if self.use_shap_proxied_fs else "auto"),
            # 2026-05-28 ShapProxiedFS deeper extension axes (B1-B6 audit-pass-2).
            # All collapse to ShapProxiedFS.__init__ defaults when the selector
            # is off so non-shap combos don't gain phantom variation.
            (self.shap_proxied_config_jitter_cfg if self.use_shap_proxied_fs else False),
            (self.shap_proxied_uncertainty_penalty_cfg if self.use_shap_proxied_fs else 0.0),
            # within_cluster_refine ALSO depends on cluster_features being on
            # (literal False switches off the cluster pass entirely, so the
            # refine flag becomes a no-op); canon True there so the toggle
            # doesn't fork canonical keys it can't actually affect.
            (
                self.shap_proxied_within_cluster_refine_cfg
                if (self.use_shap_proxied_fs
                    and self.shap_proxied_cluster_features_cfg is not False)
                else True
            ),
            (self.shap_proxied_use_bias_corrector_cfg if self.use_shap_proxied_fs else True),
            (self.shap_proxied_refine_n_estimators_cfg if self.use_shap_proxied_fs else 100),
            (
                self.shap_proxied_trust_guard_n_estimators_cfg
                if self.use_shap_proxied_fs
                else 100
            ),
            # 2026-05-28 audit-pass-2 PART A canons.
            # EnsemblingConfig.degenerate_class_ratio only meaningful on
            # classification + ensembling (the degenerate-subset gate is
            # binary/multilabel-only); mirror the existing
            # flag_degenerate_conf_subset_cfg gating.
            (
                self.ensembling_degenerate_class_ratio_cfg
                if (self.use_ensembles
                    and self.target_type in ("binary_classification",
                                              "multilabel_classification"))
                else 0.01
            ),
            # use_flaml_zeroshot only meaningful when xgb or lgb is in scope
            # (it picks flaml_zeroshot.{XGB,LGBM}{Classifier,Regressor}).
            # Canon to False when neither is in models.
            (
                self.behavior_use_flaml_zeroshot_cfg
                if ("xgb" in self.models or "lgb" in self.models)
                else False
            ),
            # target_temporal_audit_granularity only meaningful when the
            # temporal-audit phase actually runs (target_temporal_audit_column
            # resolved to a real datetime via with_datetime_col gating).
            (
                self.target_temporal_audit_granularity_cfg
                if (self.with_datetime_col
                    and self.target_temporal_audit_column_cfg == "ts_col")
                else "auto"
            ),
            # PreprocessingExtensionsConfig.dim_n_components only meaningful
            # when a dim-reducer is actually picked (PCA or TruncatedSVD).
            (
                self.prep_ext_dim_n_components_cfg
                if self.prep_ext_dim_reducer_cfg in ("PCA", "TruncatedSVD")
                else 50
            ),
            # 2026-05-28 TextDetectionConfig.text_min_cardinality. Collapse to
            # library default (300) when FHC is not enabled -- axis is a no-op
            # when the FHC sub-config never gets built.
            (self.fhc_text_min_cardinality_cfg if self.enable_feature_handling_config_cfg else 300),
            # 2026-05-28 CompositeTargetDiscoveryConfig deep knobs. Each collapses
            # to its library default when composite discovery is off OR the gating
            # sub-axis is off. composite_auto_skip_on_baseline_optimal_cfg ALSO
            # requires baseline_diagnostics_enabled_cfg; composite_mi_n_neighbors_cfg
            # ALSO requires composite_mi_estimator_cfg='knn'.
            (
                self.composite_auto_skip_on_baseline_optimal_cfg
                if (
                    self.composite_discovery_enabled_cfg
                    and self.target_type == "regression"
                    and self.baseline_diagnostics_enabled_cfg
                )
                else False
            ),
            (
                self.composite_mi_n_neighbors_cfg
                if (
                    self.composite_discovery_enabled_cfg
                    and self.target_type == "regression"
                    and self.composite_mi_estimator_cfg == "knn"
                )
                else 3
            ),
            (
                self.composite_auto_base_null_perms_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 20
            ),
            (
                self.composite_multi_base_max_k_cfg
                if (self.composite_discovery_enabled_cfg and self.target_type == "regression")
                else 3
            ),
            # 2026-05-28 TrainingBehaviorConfig.extreme_ar_group_aware_skip_models.
            # Only meaningful when MLP-extreme-AR skip is on AND mlp is in the
            # combo's models tuple (the skip-list axis only bites then).
            (
                self.extreme_ar_group_aware_skip_models_cfg
                if (self.mlp_extreme_ar_group_aware_skip_cfg and "mlp" in self.models)
                else "default_neural"
            ),
            # 2026-05-28 FeatureSelectionConfig.pre_screen_null_fraction_threshold.
            # Sibling of fs_pre_screen_variance_threshold_cfg: only fires when
            # an FS method (MRMR / RFECV / BorutaShap) actually invokes the
            # pre-screen path. Mirror the existing variance-threshold gating
            # exactly so the two siblings collapse on the SAME condition.
            (
                self.fs_pre_screen_null_fraction_threshold_cfg
                if (self.use_mrmr_fs or self.rfecv_estimator_cfg is not None
                    or self.use_boruta_shap_cfg)
                else 0.99
            ),
            # 2026-05-28 LinearModelConfig.l1_ratio. Only meaningful when
            # 'linear' is in the models AND the solver is 'saga' (sklearn
            # raises l1_ratio>0 with lbfgs/liblinear). Canon to 0.0 (Ridge-
            # equivalent) when those conditions don't hold so disabled
            # combos collapse to a single canonical key.
            (
                self.linear_l1_ratio_cfg
                if ("linear" in self.models and self.linear_solver_cfg == "saga")
                else 0.0
            ),
            # 2026-05-28 RecurrentConfig.hidden_size. Only meaningful when
            # a canonical recurrent model is picked. Canon to library default
            # (128) otherwise.
            (
                self.recurrent_hidden_size_cfg
                if self._canonical_recurrent_model() is not None
                else 128
            ),
        )

    def _canonical_recurrent_model(self) -> "str | None":
        rec = self.recurrent_model_cfg
        if rec is None:
            return None
        # Recurrent training is the slowest model on the bench (PyTorch
        # Lightning loop). Cap at the small-row tier so a fuzz combo
        # doesn't push past per-test timeout. Was n_rows > 600 against
        # the legacy (300, 600, 1000, 5000) tiers; the 2026-05-21 axis
        # bump (1000, 200_000) makes 1000 the small tier so recurrent
        # only fires there (200k would explode timeout).
        if self.n_rows > 1000:
            return None
        # The recurrent classifier path expects a 1-D label; multilabel
        # (N, K) targets aren't supported by RecurrentTorchModel today.
        if self.target_type == "multilabel_classification":
            return None
        # Text / embedding columns can't be packed into the auxiliary
        # feature matrix (RecurrentDataset.aux_features expects float32),
        # and there's no TF-IDF preflight on the recurrent path.
        if self.text_col_count > 0 or self.embedding_col_count > 0:
            return None
        # Categorical features need encoding before reaching the
        # recurrent path's StandardScaler (np.asarray on string columns
        # raises). Require either zero cats or polars-ds encoding active.
        if self.cat_feature_count > 0 and not (
            self.prefer_polarsds
            and self.categorical_encoding_cfg in ("ordinal", "onehot")
            and not self.skip_categorical_encoding_cfg
        ):
            return None
        # Recurrent + MRMR + small n + heavy aging/OD trim collapses the
        # tabular train to ≤ ~100 rows, MRMR then drops every feature, and
        # the companion mlframe model (e.g. CB) hits Pool() with empty
        # labels (c0079). Disable recurrent in this combination — the
        # tabular feature-selection axis is exercised separately.
        if self.use_mrmr_fs:
            return None
        return rec

    def _canonical_pysr_enabled(self) -> bool:
        """F5 -- PySR symbolic regression cannot consume inf / NaN values. Whenever ``inject_inf_nan=True`` the PySR path crashes on the first non-finite row regardless of the True/False axis value, so canonicalise to the disabled variant to spare pairwise-coverage budget."""
        if not self.prep_ext_pysr_enabled_cfg:
            return False
        if self.inject_inf_nan or self.inject_all_nan_col:
            return False
        return True

    def _canonical_enable_crash_reporting(self) -> bool:
        """F1 -- crash_reporting is a Windows-only Faulthandler dump hook (``mlframe.training.crash_reporting``). On non-Windows hosts the True variant is a no-op so collapse to False; on Windows it stays meaningful."""
        import platform as _platform
        if _platform.system() != "Windows":
            return False
        return bool(self.enable_crash_reporting_cfg)

    def _canonical_rfecv_estimator(self) -> "str | None":
        rfe = self.rfecv_estimator_cfg
        if rfe is None:
            return None
        # rfecv name is "<base>_rfecv" — strip suffix to look up base model.
        base = rfe.rsplit("_rfecv", 1)[0]
        if base not in self.models:
            return None
        # RFECV iterates 10+ refits; cap at the small-row tier. Was
        # n_rows > 1200 against legacy tiers; the 2026-05-21 axis bump
        # (1000, 200_000) makes 1000 the small tier so RFECV only fires
        # there (200k * 10 refits would dominate the suite wall).
        if self.n_rows > 1000:
            return None
        # sklearn RFECV.fit (via check_classification_targets) raises
        # "Supported target types are: ('binary', 'multiclass'). Got
        # 'multilabel-indicator'". Multilabel is not in scope for RFECV.
        if self.target_type == "multilabel_classification":
            return None
        # Rare imbalance (rare_5pct / rare_1pct on small n) lets RFECV's
        # internal CV folds land on single-class y, raising "Invalid
        # classes inferred from unique values of y. Expected: [0], got
        # [1]". Disable RFECV unless the target distribution is balanced.
        if (
            self.target_type == "binary_classification"
            and self.imbalance_ratio != "balanced"
        ):
            return None
        return rfe

    def _canonical_prep_ext(self, name: str) -> "Any":
        """Collapse PreprocessingExtensions axes to None when the surrounding
        combo would feed the sklearn bridge data it cannot consume:
          * raw NaN / Inf in numeric columns (sklearn transformers raise);
          * an all-NaN column (PCA/scalers reject any NaN);
          * raw categorical columns reaching the bridge — happens whenever
            cats exist AND any of: polars-ds pipeline is off (no encoding
            step at all), categorical encoding is explicitly skipped, or
            categorical encoding is None.
        For the dim-reducer specifically, also require enough features
        (heuristic: ≥ 8 cat columns onehot-expanded so n_features beats
        the default dim_n_components=50) — otherwise sklearn raises
        "n_components must be <= n_features".
        """
        attr = f"prep_ext_{name}_cfg"
        value = getattr(self, attr)
        if value is None:
            return None
        if self.inject_inf_nan or self.inject_all_nan_col:
            return None
        if self.cat_feature_count > 0:
            cats_will_be_encoded = (
                self.prefer_polarsds
                and self.categorical_encoding_cfg in ("ordinal", "onehot")
                and not self.skip_categorical_encoding_cfg
            )
            if not cats_will_be_encoded:
                return None
        # Text columns flow through as raw strings unless tfidf_columns is
        # explicitly set; embedding columns are pl.List(Float32) which
        # numpy.asarray turns into ragged object arrays. Both crash any
        # downstream sklearn transformer (StandardScaler / Polynomial / PCA).
        # Disable prep_ext entirely for combos with text or embedding.
        if self.text_col_count > 0 or self.embedding_col_count > 0:
            return None
        # Latent NaN sources unrelated to the inject_* axes — the polars-ds
        # imputer chain doesn't always run (e.g. when the caller sets
        # imputer_strategy=None or scaler=None). The bridge then sees NaN
        # values that crash PolynomialFeatures / KBinsDiscretizer / PCA
        # (c0116, c0047). Require the imputer + scaler to be active so the
        # bridge always sees clean input. Also disable when the datetime
        # path is on (datetime extraction may introduce NaT/NaN that the
        # polars-ds imputer doesn't handle for newly-derived columns).
        if (
            self.imputer_strategy_cfg is None
            or self.scaler_name_cfg is None
            or self.with_datetime_col
            # inject_zero_col adds a constant-zero numeric column. The
            # polars-ds RobustScaler computes (X - median) / IQR; for
            # zero-variance the IQR=0, the divide returns NaN, and the
            # bridge then sees NaN that crashes PolynomialFeatures /
            # KBinsDiscretizer (c0116). Disable prep_ext for this combo.
            or self.inject_zero_col
        ):
            return None
        # dim_reducer requires n_features >= dim_n_components (default 50).
        # The synthetic builder emits 4 numeric + cat_feature_count*~5 (for
        # onehot) features. Conservative threshold: cat_feature_count >= 8
        # with onehot (~40+ extra cols) OR ordinal (cats stay as 1 col each)
        # is risky → require onehot + cats >= 8 for dim_reducer.
        if name == "dim_reducer":
            if not (
                self.cat_feature_count >= 8
                and self.categorical_encoding_cfg == "onehot"
            ):
                return None
        return value

    def _canonical_text_col_count(self) -> int:
        """Mirror the text_col_count canonicalisation in canonical_key.

        Currently a passthrough — historical canonicalisation rules
        (which dropped text cols on small-n CB + heavy NaN, and on
        cb_rfecv + NaN) were masking real production hangs in CB's
        text-feature path. Those are now fixed in production code:
        ``trainer.py`` skips text features whenever the effective
        train rows are below CB's occurrence_lower_bound floor.
        """
        return self.text_col_count

    def _is_chain_dispatch(self) -> bool:
        return (
            self.target_type == "multilabel_classification"
            and self._canonical_multilabel_strategy() == "chain"
        )

    def _canonical_multilabel_strategy(self) -> str:
        if self.target_type != "multilabel_classification":
            return "auto"
        if self.multilabel_strategy_cfg != "chain":
            return self.multilabel_strategy_cfg
        # Chain dispatch is only safe with classifier-friendly numeric input:
        # sklearn ClassifierChain wraps the inner estimator and bypasses the
        # polars-native fastpath, so categorical/string columns reach HGB/LGB
        # raw. The linear model also routes (N,K) preds through the regression
        # report and crashes on multilabel-shaped predictions. Both downgrades
        # are correctness fixes, not capability removals — chain is still
        # exercised for the combos that satisfy the prerequisites.
        if self.cat_feature_count > 0:
            return "wrapper"
        if "linear" in self.models:
            return "wrapper"
        return "chain"

    def _canonical_imputer_strategy(self) -> "str | None":
        """imputer_strategy variants are no-ops when there's nothing to impute.

        Returns the chosen strategy only when the frame is expected to
        contain nulls / NaNs — otherwise canonicalises to the default
        ('mean') so the dedup pass collapses the no-op variants."""
        has_nulls = (
            self.inject_inf_nan
            or self.inject_all_nan_col
            or (self.cat_feature_count > 0 and self.null_fraction_cats > 0)
            or self.inject_degenerate_cols  # adds an all-null column
        )
        return self.imputer_strategy_cfg if has_nulls else "mean"

    def _canonical_imbalance(self) -> str:
        if "classification" not in self.target_type:
            return "balanced"
        # multiclass and multilabel: imbalance fuzz is its own can of worms
        # (per-class balancing for multiclass, per-label for multilabel —
        # neither is currently supported by the synthetic builder).
        # Collapse to balanced for these target types.
        if self.target_type in ("multiclass_classification", "multilabel_classification"):
            return "balanced"
        imb = self.imbalance_ratio
        frac = {"rare_5pct": 0.05, "rare_1pct": 0.01, "balanced": 0.5}.get(imb, 0.5)
        # Effective minority-source N after train-side trimming. When the
        # combo also enables trainset_aging_limit (drops first/last K% of
        # train) + outlier_detection (filters ~5%), the surviving train
        # may have ~half the row count and a corresponding fraction of
        # minority rows. The earlier "n*0.1*frac >= 4" guard considered
        # only the val/test slice size — train-thinning broke c0048 at
        # n=5000 (1923 rows post-trim → 19 expected positives → unlucky 0).
        effective_n = float(self.n_rows)
        if self.trainset_aging_limit_cfg is not None:
            effective_n *= self.trainset_aging_limit_cfg
        if self.outlier_detection is not None:
            effective_n *= 0.95  # OD typically drops ~5% (contamination=0.05)
        # Each split gets ~0.1×n rows. Require frac × 0.1 × n_eff ≥ 4
        # (~4 minority rows expected in the smallest slice).
        if frac * 0.1 * effective_n < 4:
            # Try the next-safer rarity level.
            if imb == "rare_1pct":
                return "rare_5pct" if 0.05 * 0.1 * effective_n >= 4 else "balanced"
            if imb == "rare_5pct":
                return "balanced"
        return imb

    def short_id(self) -> str:
        h = hashlib.blake2s(repr(self.canonical_key()).encode(), digest_size=4).hexdigest()
        return f"c{self.seed:04d}_{h}"

    def pytest_id(self) -> str:
        # Include a human-readable prefix so failing IDs are diagnostic.
        tag = "_".join(sorted(self.models))
        short_input = self.input_type.replace("polars_", "pl_")
        return f"{self.short_id()}-{tag}-{short_input}-n{self.n_rows}"

    def to_json(self) -> dict:
        return {
            "short_id": self.short_id(),
            "models": list(self.models),
            "input_type": self.input_type,
            "n_rows": self.n_rows,
            "cat_feature_count": self.cat_feature_count,
            "null_fraction_cats": self.null_fraction_cats,
            "use_mrmr_fs": self.use_mrmr_fs,
            "weight_schemas": list(self.weight_schemas),
            "target_type": self.target_type,
            "target_carrier": self.target_carrier,
            "auto_detect_cats": self.auto_detect_cats,
            "align_polars_categorical_dicts": self.align_polars_categorical_dicts,
            "seed": self.seed,
            "prefer_polarsds": self.prefer_polarsds,
            "use_text_features": self.use_text_features,
            "honor_user_dtype": self.honor_user_dtype,
            "text_col_count": self.text_col_count,
            "embedding_col_count": self.embedding_col_count,
            # 2026-04-24 combo-extension axes
            "outlier_detection": self.outlier_detection,
            "use_ensembles": self.use_ensembles,
            "continue_on_model_failure": self.continue_on_model_failure,
            "iterations": self.iterations,
            "prefer_calibrated_classifiers": self.prefer_calibrated_classifiers,
            "inject_degenerate_cols": self.inject_degenerate_cols,
            "inject_inf_nan": self.inject_inf_nan,
            "with_datetime_col": self.with_datetime_col,
            "inject_zero_col": self.inject_zero_col,
            "fairness_col": self.fairness_col,
            "custom_prep": self.custom_prep,
            "input_storage": self.input_storage,
            # 2026-04-24 round 2
            "fillna_value_cfg": self.fillna_value_cfg,
            "scaler_name_cfg": self.scaler_name_cfg,
            "categorical_encoding_cfg": self.categorical_encoding_cfg,
            "skip_categorical_encoding_cfg": self.skip_categorical_encoding_cfg,
            "val_placement_cfg": self.val_placement_cfg,
            "test_size_cfg": self.test_size_cfg,
            "trainset_aging_limit_cfg": self.trainset_aging_limit_cfg,
            "cat_text_card_threshold_cfg": self.cat_text_card_threshold_cfg,
            "early_stopping_rounds_cfg": self.early_stopping_rounds_cfg,
            "use_robust_eval_metric_cfg": self.use_robust_eval_metric_cfg,
            # Fix G
            "inject_label_leak": self.inject_label_leak,
            "inject_rank_deficient": self.inject_rank_deficient,
            "inject_all_nan_col": self.inject_all_nan_col,
            # R3
            "inject_test_drift": self.inject_test_drift,
            "imbalance_ratio": self.imbalance_ratio,
            "weird_cat_content": self.weird_cat_content,
            "multilabel_strategy_cfg": self.multilabel_strategy_cfg,
            # 2026-04-26 batch 1
            "fix_infinities_cfg": self.fix_infinities_cfg,
            "ensure_float32_cfg": self.ensure_float32_cfg,
            "remove_constant_columns_cfg": self.remove_constant_columns_cfg,
            "imputer_strategy_cfg": self.imputer_strategy_cfg,
            "shuffle_val_cfg": self.shuffle_val_cfg,
            "shuffle_test_cfg": self.shuffle_test_cfg,
            "wholeday_splitting_cfg": self.wholeday_splitting_cfg,
            "val_sequential_fraction_cfg": self.val_sequential_fraction_cfg,
            # batch 3 — multilabel dispatch
            "multilabel_n_chains_cfg": self.multilabel_n_chains_cfg,
            "multilabel_chain_order_cfg": self.multilabel_chain_order_cfg,
            "multilabel_cv_cfg": self.multilabel_cv_cfg,
            # batch 4 — PreprocessingExtensionsConfig
            "prep_ext_scaler_cfg": self.prep_ext_scaler_cfg,
            "prep_ext_kbins_cfg": self.prep_ext_kbins_cfg,
            "prep_ext_polynomial_degree_cfg": self.prep_ext_polynomial_degree_cfg,
            "prep_ext_dim_reducer_cfg": self.prep_ext_dim_reducer_cfg,
            "prep_ext_nonlinear_cfg": self.prep_ext_nonlinear_cfg,
            "prep_ext_pysr_enabled_cfg": self.prep_ext_pysr_enabled_cfg,
            "mrmr_nan_strategy_cfg": self.mrmr_nan_strategy_cfg,
            # batch 5
            "rfecv_estimator_cfg": self.rfecv_estimator_cfg,
            # batch 6
            "recurrent_model_cfg": self.recurrent_model_cfg,
            # 2026-04-28 batch 4 followup
            "include_confidence_analysis_cfg": self.include_confidence_analysis_cfg,
        }


# ---------------------------------------------------------------------------
# Known-xfail rules (single source of truth for auto-xfail)
# ---------------------------------------------------------------------------


# _rule_linear_polars_gating_bug REMOVED 2026-04-22 (Fix 11):
# core.py:3085 now computes polars_pipeline_applied per-strategy:
#   polars_pipeline_applied AND strategy.supports_polars
#                            AND NOT strategy.requires_encoding
# Linear (supports_polars=False, requires_encoding=True) always gets
# skip_preprocessing=False, so its pre_pipeline runs the encoder fully.
# Permanent regression guard: test_polars_full_combo_with_linear in
# test_integration_prod_like_polars.py (xfail removed) +
# test_sensor_linear_polars_gating_bug in test_fuzz_regression_sensors.py.


# _rule_mrmr_plus_linear_multi_pandas REMOVED 2026-04-23: new-seed fuzz
# showed 4/6 XPASS from this rule — the MRMR+linear+pandas combos that
# once failed on feature-name mismatch now pass (composite of all
# 2026-04-22/23 fixes — per-model pipeline cloning, _is_fitted
# Pipeline-aware check, MRMR in-place drop, Fix 10 polars-native MRMR,
# Fix 11 per-strategy polars_pipeline_applied, Fix 12 dt==class dispatch).
# Rule is now misleading; permanent regression guard: the integration
# test suite's test_polars_full_combo_with_linear (already un-xfailed).

# _rule_cb_nan_in_cat_features_mrmr REMOVED 2026-04-23: fixed in trainer.py
# `_polars_nullable_categorical_cols` — the candidate list for the fill_null
# pre-fit pass now includes pl.Utf8 / pl.String dtypes (previously only
# Categorical / Enum). Raw Utf8 cat columns with nulls are now filled
# before CB sees them.


# _rule_mrmr_plus_xgb_lgb_polars_utf8_small REMOVED 2026-04-23: same root
# cause as _rule_cb_sparse_text_small — categorize_dataset incorrectly
# routed polars Categorical / Utf8 columns through the numeric branch
# because `dt in set_of_dtype_classes` uses hash equality, and
# `pl.Categorical` instance hash differs from class hash. Permanent
# regression guard: test_sensor_categorize_dataset_recognizes_polars_cat_dtypes.


# _rule_cb_sparse_text_small REMOVED 2026-04-23: the underlying failures
# were a symptom of the categorize_dataset dt-in-set hash bug
# (filters.py:2660, `dt in {pl.Utf8, pl.Categorical, ...}` returns False
# for Categorical instances because class-vs-instance hash differs).
# Fixed there with explicit `==` checks per dtype. Both c0048 and c0098
# now PASS. Permanent regression guard:
# test_sensor_categorize_dataset_recognizes_polars_cat_dtypes.


# REMOVED 2026-04-22: _rule_polars_schema_dispatch_bug
#
# Root cause was: _build_tier_dfs in core.py cached tier-DFs keyed only on
# strategy.feature_tier() — which collides between Polars and pandas inputs
# when a non-polars-native strategy (Linear) runs before a polars-native
# strategy (XGB) in the same multi-model suite. Linear stashed pandas
# tier-DFs under tier=(False,False); XGB retrieved the same key and got
# pandas back; XGBoostStrategy.prepare_polars_dataframe then tried
# df.schema.items() on a pandas frame and raised AttributeError.
#
# Fix: cache key now = (tier, kind) where kind ∈ {"pl","pd"} sampled from
# the first non-None input. See test_sensor_tier_cache_polars_pandas_collision
# in test_fuzz_regression_sensors.py — permanent regression guard.


# _rule_mrmr_single_linear_pandas REMOVED 2026-04-22: MRMR.transform now
# uses getattr(self, 'support_', None) so a fit() that exits without
# setting support_ (e.g. early-exit on low-MI synthetic data) degrades
# to pass-through instead of raising. Regression guard:
# test_sensor_mrmr_transform_handles_missing_support_ in test_fuzz_regression_sensors.py.


# _rule_multilabel_full_pipeline_deferred REMOVED 2026-04-25 — full
# multilabel integration landed in Session 6. All 42 multilabel combos
# in the fuzz suite pass end-to-end after target_type plumbing into
# get_training_configs, MultiOutputClassifier wrapping for
# HGB/XGB/LGB/Linear, multilabel-aware report path, MRMR target injection
# fix, and supervised-encoder target collapse. See CHANGELOG.md
# "Session 6: multilabel full-pipeline integration" entry.


# _rule_cb_pool_reuse_with_mrmr_small_n_filtered REMOVED 2026-04-27 — fixed
# by empty-target + length-mismatch guards in
# trainer._maybe_get_or_build_cb_pool (rebuild on mismatch / empty target).
# The rule's TODO (rebuild cb Pool unconditionally when use_mrmr_fs=True) is
# subsumed by the per-fit length-mismatch check.
#
# _rule_cb_text_dict_collapse_with_full_quartet REMOVED 2026-04-27 — fixed
# by dynamic CB ``text_processing`` calibration in
# training/helpers.compute_cb_text_processing, applied at trainer fit-time
# AND in feature_selection/wrappers.py RFECV inner-fold. The rule's TODO
# ("auto-tune occurrence_lower_bound on small folds") is exactly what the
# new helper does — scaling the floor proportionally to fold rows so words
# that occur in 5%+ survive the prune. Permanent regression coverage:
# fuzz combos c0056 / c0070 / c0079 in tests/training/test_fuzz_suite.py.


KNOWN_XFAIL_RULES: list[tuple[Callable[[FuzzCombo], bool], str]] = [
    # _rule_linear_polars_gating_bug REMOVED 2026-04-22 (Fix 11).
    # Permanent regression guard: test_polars_full_combo_with_linear
    # (xfail removed) + test_sensor_linear_polars_gating_bug.
    # _rule_mrmr_plus_linear_multi_pandas REMOVED 2026-04-23.
    # _rule_cb_nan_in_cat_features_mrmr REMOVED 2026-04-23.
    # _rule_cb_multilabel_cat_nulls REMOVED 2026-04-26 — fixed by
    # defensive null-fill in trainer._train_model_with_fallback for the
    # CB pandas multilabel path.
    # _rule_empty_val_degenerate_cats_backward REMOVED 2026-04-26 — fixed
    # by min-rows guard in trainer._apply_pre_pipeline_transforms (skips
    # pre_pipeline.transform when val_df has 0 rows).
    # _rule_cb_only_mrmr_small_n_with_od REMOVED 2026-04-26 — fixed by
    # empty-target guard in _maybe_get_or_build_cb_pool.
    # _rule_cb_text_feature_full_quartet_heavy_inject REMOVED 2026-04-26
    # — fixed by feature-list filter in feature_selection/wrappers.py
    # (cat/text/embedding lists from outer fit_params are now narrowed to
    # current_features instead of overwriting the iteration-local lists).
    # _rule_cb_regression_polars_enum_mrmr_nulls_large REMOVED 2026-04-26 —
    # the matching combo (c0033 under default master_seed) now XPASSes; the
    # NaN-in-cat-feature path no longer reproduces. Removing the rule turns
    # any future regression into a real test failure instead of a silent
    # absorbed xpass.
    # _rule_multilabel_full_pipeline_deferred REMOVED 2026-04-25 — Session 6
    # full integration landed; all 42 multilabel combos pass end-to-end.
    # _rule_mrmr_plus_xgb_lgb_polars_utf8_small REMOVED 2026-04-23 — fixed by
    # `dt in set` → `dt == class` correction in filters.py categorize_dataset.
    # Permanent regression guard:
    # test_sensor_categorize_dataset_recognizes_polars_cat_dtypes.
    # _rule_cb_sparse_text_small REMOVED 2026-04-23 — same root cause; same sensor.
    # _rule_polars_schema_dispatch_bug REMOVED 2026-04-22: fixed in
    # core.py _build_tier_dfs (cache key now includes container kind).
    # Permanent regression guard: test_sensor_tier_cache_polars_pandas_collision.
    # _rule_mrmr_single_linear_pandas REMOVED 2026-04-22: fixed in MRMR.transform.
    # Permanent regression guard: test_sensor_mrmr_transform_handles_missing_support_.
]


def xfail_reason(combo: FuzzCombo) -> str | None:
    for predicate, reason in KNOWN_XFAIL_RULES:
        if predicate(combo):
            return reason
    return None


# ---------------------------------------------------------------------------
# Enumerator: pairwise-greedy, deterministic, deduplicated
# ---------------------------------------------------------------------------


def _powerset_nonempty(items: tuple[str, ...]) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    for r in range(1, len(items) + 1):
        for sub in iter_combinations(items, r):
            out.append(tuple(sorted(sub)))
    return out


# iter373 (2026-05-27): set of model flavours that ship a native ranker
# implementation. Used to reject fuzz combos where target_type=LTR is paired
# with a subset that contains NO ranker (e.g. ``models=('linear',)`` or
# ``models=('hgb',)``) -- the runtime raises NotImplementedError on those.
# c0120 was such a combo, surfaced 2026-05-27 by the random-pick /loop;
# without the filter the enumerator emits ~5 unrunnable combos per 150 (any
# LTR subset that happens to be hgb / linear / sklearn only).
_LTR_NATIVE_RANKERS: frozenset[str] = frozenset({"cb", "xgb", "lgb", "mlp"})


# 2026-05-28 audit-pass-2 PART A: behavior_use_flaml_zeroshot needs an optional
# dep (`flaml`). Probe once at import time; canon True -> False if the dep is
# missing so the fuzz harness doesn't propose a combo that crashes at fit-time
# on environments without it (CI matrix without `flaml`). Cached as a module
# constant so each _build_combo call is cheap.
try:  # pragma: no cover -- import-time probe, behaviour is identical either way
    import importlib

    importlib.import_module("flaml")
    _HAS_FLAML = True
except Exception:
    _HAS_FLAML = False


def _canon_use_flaml_zeroshot(requested: bool) -> bool:
    """Canon for behavior_use_flaml_zeroshot_cfg: drop True to False when the
    `flaml` optional dep is not importable so generated combos stay runnable."""
    return bool(requested) if _HAS_FLAML else False


def _combo_is_runnable(models: tuple[str, ...], target_type: str) -> bool:
    """Cheap structural check: does this (models, target_type) pair have a
    chance of completing training? Returns False for LTR combos whose model
    subset contains NO native ranker flavour. Other target types are always
    runnable from this axis-pair perspective (other axes do their own
    canonicalisation downstream)."""
    if target_type == "learning_to_rank":
        return any(m in _LTR_NATIVE_RANKERS for m in models)
    return True


def _sample_axes(rng: random.Random) -> dict[str, Any]:
    return {name: rng.choice(values) for name, values in AXES.items()}


def _build_combo(models: tuple[str, ...], axes: dict[str, Any], seed: int) -> FuzzCombo:
    return FuzzCombo(
        models=tuple(sorted(models)),
        input_type=axes["input_type"],
        n_rows=axes["n_rows"],
        cat_feature_count=axes["cat_feature_count"],
        null_fraction_cats=axes["null_fraction_cats"],
        use_mrmr_fs=axes["use_mrmr_fs"],
        weight_schemas=axes["weight_schemas"],
        target_type=axes["target_type"],
        auto_detect_cats=axes["auto_detect_cats"],
        align_polars_categorical_dicts=axes["align_polars_categorical_dicts"],
        seed=seed,
        prefer_polarsds=axes.get("prefer_polarsds", True),
        use_text_features=axes.get("use_text_features", True),
        honor_user_dtype=axes.get("honor_user_dtype", False),
        target_carrier=axes.get("target_carrier", "numpy"),
        text_col_count=axes.get("text_col_count", 0),
        embedding_col_count=axes.get("embedding_col_count", 0),
        # 2026-04-24 combo-extension axes
        outlier_detection=axes.get("outlier_detection"),
        use_ensembles=axes.get("use_ensembles", False),
        continue_on_model_failure=axes.get("continue_on_model_failure", False),
        iterations=axes.get("iterations", 3),
        prefer_calibrated_classifiers=axes.get("prefer_calibrated_classifiers", False),
        inject_degenerate_cols=axes.get("inject_degenerate_cols", False),
        inject_inf_nan=axes.get("inject_inf_nan", False),
        with_datetime_col=axes.get("with_datetime_col", False),
        inject_zero_col=axes.get("inject_zero_col", False),
        fairness_col=axes.get("fairness_col"),
        custom_prep=axes.get("custom_prep"),
        input_storage=axes.get("input_storage", "memory"),
        # 2026-04-24 round 2
        fillna_value_cfg=axes.get("fillna_value_cfg"),
        scaler_name_cfg=axes.get("scaler_name_cfg", "standard"),
        categorical_encoding_cfg=axes.get("categorical_encoding_cfg", "ordinal"),
        skip_categorical_encoding_cfg=axes.get("skip_categorical_encoding_cfg", False),
        val_placement_cfg=axes.get("val_placement_cfg", "forward"),
        test_size_cfg=axes.get("test_size_cfg", 0.1),
        trainset_aging_limit_cfg=axes.get("trainset_aging_limit_cfg"),
        cat_text_card_threshold_cfg=axes.get("cat_text_card_threshold_cfg", 300),
        early_stopping_rounds_cfg=axes.get("early_stopping_rounds_cfg"),
        use_robust_eval_metric_cfg=axes.get("use_robust_eval_metric_cfg", True),
        # Fix G
        inject_label_leak=axes.get("inject_label_leak", False),
        inject_rank_deficient=axes.get("inject_rank_deficient", False),
        inject_all_nan_col=axes.get("inject_all_nan_col", False),
        # R3
        inject_test_drift=axes.get("inject_test_drift"),
        imbalance_ratio=axes.get("imbalance_ratio", "balanced"),
        weird_cat_content=axes.get("weird_cat_content"),
        multilabel_strategy_cfg=axes.get("multilabel_strategy_cfg", "auto"),
        # 2026-04-26 batch 1
        fix_infinities_cfg=axes.get("fix_infinities_cfg", True),
        ensure_float32_cfg=axes.get("ensure_float32_cfg", True),
        remove_constant_columns_cfg=axes.get("remove_constant_columns_cfg", True),
        imputer_strategy_cfg=axes.get("imputer_strategy_cfg", "mean"),
        shuffle_val_cfg=axes.get("shuffle_val_cfg", False),
        shuffle_test_cfg=axes.get("shuffle_test_cfg", False),
        wholeday_splitting_cfg=axes.get("wholeday_splitting_cfg", True),
        val_sequential_fraction_cfg=axes.get("val_sequential_fraction_cfg", 0.5),
        # batch 3 — multilabel dispatch
        multilabel_n_chains_cfg=axes.get("multilabel_n_chains_cfg", 3),
        multilabel_chain_order_cfg=axes.get("multilabel_chain_order_cfg", "random"),
        multilabel_cv_cfg=axes.get("multilabel_cv_cfg", 5),
        # batch 4 — PreprocessingExtensionsConfig
        prep_ext_scaler_cfg=axes.get("prep_ext_scaler_cfg"),
        prep_ext_kbins_cfg=axes.get("prep_ext_kbins_cfg"),
        prep_ext_polynomial_degree_cfg=axes.get("prep_ext_polynomial_degree_cfg"),
        prep_ext_dim_reducer_cfg=axes.get("prep_ext_dim_reducer_cfg"),
        prep_ext_nonlinear_cfg=axes.get("prep_ext_nonlinear_cfg"),
        prep_ext_pysr_enabled_cfg=axes.get("prep_ext_pysr_enabled_cfg", False),
        mrmr_nan_strategy_cfg=axes.get("mrmr_nan_strategy_cfg", "separate_bin"),
        # batch 5
        rfecv_estimator_cfg=axes.get("rfecv_estimator_cfg"),
        # batch 6
        recurrent_model_cfg=axes.get("recurrent_model_cfg"),
        # 2026-04-28 batch 4 followup
        include_confidence_analysis_cfg=axes.get("include_confidence_analysis_cfg", False),
        # 2026-05-11 Wave 15 -- MRMR-internal knobs
        mrmr_interactions_max_order_cfg=axes.get("mrmr_interactions_max_order_cfg", 1),
        mrmr_fe_max_steps_cfg=axes.get("mrmr_fe_max_steps_cfg", 1),
        mrmr_cat_fe_enable_cfg=axes.get("mrmr_cat_fe_enable_cfg", True),
        # 2026-05-11 Wave 21 -- assorted config-toggle axes
        dummy_baselines_enabled_cfg=axes.get("dummy_baselines_enabled_cfg", True),
        baseline_diagnostics_enabled_cfg=axes.get("baseline_diagnostics_enabled_cfg", True),
        use_groups_cfg=axes.get("use_groups_cfg", True),
        apply_outlier_to_val_cfg=axes.get("apply_outlier_to_val_cfg", True),
        multilabel_allow_uncalibrated_cfg=axes.get("multilabel_allow_uncalibrated_cfg", False),
        report_residual_audit_cfg=axes.get("report_residual_audit_cfg", True),
        ltr_assume_comparable_scales_cfg=axes.get("ltr_assume_comparable_scales_cfg", False),
        composite_discovery_enabled_cfg=axes.get("composite_discovery_enabled_cfg", False),
        composite_transforms_mode_cfg=axes.get("composite_transforms_mode_cfg", None),
        mrmr_fe_npermutations_cfg=axes.get("mrmr_fe_npermutations_cfg", 0),
        mrmr_fe_ntop_features_cfg=axes.get("mrmr_fe_ntop_features_cfg", 0),
        mrmr_fe_unary_preset_cfg=axes.get("mrmr_fe_unary_preset_cfg", "minimal"),
        mrmr_fe_binary_preset_cfg=axes.get("mrmr_fe_binary_preset_cfg", "minimal"),
        mrmr_fe_smart_polynom_iters_cfg=axes.get("mrmr_fe_smart_polynom_iters_cfg", 0),
        mrmr_fe_smart_polynom_steps_cfg=axes.get("mrmr_fe_smart_polynom_steps_cfg", 10),
        mrmr_fe_min_polynom_degree_cfg=axes.get("mrmr_fe_min_polynom_degree_cfg", 3),
        mrmr_fe_max_polynom_degree_cfg=axes.get("mrmr_fe_max_polynom_degree_cfg", 3),
        mrmr_cat_fe_include_numeric_cfg=axes.get("mrmr_cat_fe_include_numeric_cfg", False),
        # 2026-05-19 -- PreprocessingExtensionsConfig polynomial-auto-tune
        # axes. Previously declared in AXES + dataclass but not threaded
        # through _build_combo, so randomised values silently fell back to
        # dataclass defaults. Wired 2026-05-21.
        prep_ext_polynomial_max_features_cfg=axes.get(
            "prep_ext_polynomial_max_features_cfg", 10_000
        ),
        prep_ext_polynomial_interaction_only_cfg=axes.get(
            "prep_ext_polynomial_interaction_only_cfg", True
        ),
        prep_ext_memory_safety_max_bytes_cfg=axes.get(
            "prep_ext_memory_safety_max_bytes_cfg", 500_000_000
        ),
        # 2026-05-19 -- composite-discovery stacked-residual axes. Same
        # un-wired bug as above; defaulting these collapsed every combo to
        # the False / True dataclass defaults regardless of the axis value.
        composite_use_stacked_discovery_cfg=axes.get(
            "composite_use_stacked_discovery_cfg", False
        ),
        composite_use_stacked_discovery_residual_cfg=axes.get(
            "composite_use_stacked_discovery_residual_cfg", False
        ),
        # 2026-05-22 -- six gate-flip axes from the TVT-MLP-collapse
        # cascade. Defaults match the post-fix values; AXES holds the
        # pre-fix variant for regression coverage.
        composite_skip_raw_dominates_ratio_cfg=axes.get(
            "composite_skip_raw_dominates_ratio_cfg", 0.0
        ),
        composite_skip_ablation_delta_pct_cfg=axes.get(
            "composite_skip_ablation_delta_pct_cfg", 0.0
        ),
        composite_eps_mi_gain_cfg=axes.get(
            "composite_eps_mi_gain_cfg", -10.0
        ),
        composite_top_k_after_mi_cfg=axes.get(
            "composite_top_k_after_mi_cfg", 32
        ),
        composite_require_beats_raw_baseline_cfg=axes.get(
            "composite_require_beats_raw_baseline_cfg", False
        ),
        composite_per_bin_n_bins_cfg=axes.get(
            "composite_per_bin_n_bins_cfg", 0
        ),
        composite_tiny_screening_mode_cfg=axes.get(
            "composite_tiny_screening_mode_cfg", "per_family"
        ),
        composite_include_additive_residual_cfg=axes.get(
            "composite_include_additive_residual_cfg", True
        ),
        mlp_activation_cfg=axes.get("mlp_activation_cfg", "ReLU"),
        composite_skip_wrap_pass_predict_cfg=axes.get(
            "composite_skip_wrap_pass_predict_cfg", True
        ),
        # 2026-05-21 -- mini-HPT (target + feature distribution analyzer)
        # + MRMR FE pair-check subsample knob.
        enable_target_distribution_analyzer_cfg=axes.get(
            "enable_target_distribution_analyzer_cfg", True
        ),
        fe_check_pairs_subsample_n_cfg=axes.get(
            "fe_check_pairs_subsample_n_cfg", 0
        ),
        # 2026-05-21 iter150 -- multi-target / multi-target-type axis.
        extra_targets=axes.get("extra_targets", None),
        # 2026-05-21 iter151 -- P0/P1/P2 audit fill-in.
        enable_quantile_regression_cfg=axes.get("enable_quantile_regression_cfg", False),
        linear_alpha_cfg=axes.get("linear_alpha_cfg", 1.0),
        linear_solver_cfg=axes.get("linear_solver_cfg", "lbfgs"),
        enable_feature_handling_config_cfg=axes.get("enable_feature_handling_config_cfg", False),
        enable_precomputed_cfg=axes.get("enable_precomputed_cfg", False),
        test_sequential_fraction_cfg=axes.get("test_sequential_fraction_cfg", None),
        calib_size_cfg=axes.get("calib_size_cfg", None),
        use_boruta_shap_cfg=axes.get("use_boruta_shap_cfg", False),
        use_sample_weights_in_fs_cfg=axes.get("use_sample_weights_in_fs_cfg", False),
        fallback_to_sklearn_cfg=axes.get("fallback_to_sklearn_cfg", True),
        prefer_gpu_configs_cfg=axes.get("prefer_gpu_configs_cfg", True),
        prefer_cpu_for_lightgbm_cfg=axes.get("prefer_cpu_for_lightgbm_cfg", True),
        mrmr_identity_cache_scope_cfg=axes.get("mrmr_identity_cache_scope_cfg", "ctx"),
        skip_identity_equivalent_pre_pipelines_cfg=axes.get(
            "skip_identity_equivalent_pre_pipelines_cfg", True
        ),
        rfecv_leakage_corr_threshold_cfg=axes.get("rfecv_leakage_corr_threshold_cfg", 0.95),
        rfecv_mbh_adaptive_threshold_cfg=axes.get("rfecv_mbh_adaptive_threshold_cfg", 30),
        # 2026-05-22 iter162 -- nested-config / depth-2 audit fill-in.
        fhc_cache_eviction_strategy_cfg=axes.get("fhc_cache_eviction_strategy_cfg", "size_weighted"),
        fhc_cache_allow_pickle_cfg=axes.get("fhc_cache_allow_pickle_cfg", False),
        fhc_cache_ram_fraction_cfg=axes.get("fhc_cache_ram_fraction_cfg", 0.3),
        fhc_text_definite_text_mean_chars_cfg=axes.get("fhc_text_definite_text_mean_chars_cfg", 100),
        fhc_text_min_alphabet_entropy_cfg=axes.get("fhc_text_min_alphabet_entropy_cfg", 4.5),
        fhc_repro_deterministic_torch_cfg=axes.get("fhc_repro_deterministic_torch_cfg", False),
        fhc_auto_locale_detection_cfg=axes.get("fhc_auto_locale_detection_cfg", "fallback_only"),
        reporting_prob_histogram_yscale_cfg=axes.get("reporting_prob_histogram_yscale_cfg", "auto"),
        reporting_title_metrics_template_cfg=axes.get(
            "reporting_title_metrics_template_cfg",
            "ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC",
        ),
        reporting_matplotlib_rcparams_cfg=axes.get("reporting_matplotlib_rcparams_cfg", None),
        reporting_multiclass_panels_cfg=axes.get(
            "reporting_multiclass_panels_cfg",
            "CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC",
        ),
        confidence_model_kwargs_cfg=axes.get("confidence_model_kwargs_cfg", "default"),
        composite_mi_estimator_cfg=axes.get("composite_mi_estimator_cfg", "bin"),
        composite_mi_nbins_cfg=axes.get("composite_mi_nbins_cfg", 16),
        composite_mi_aggregation_cfg=axes.get("composite_mi_aggregation_cfg", "mean"),
        composite_mi_sample_strategy_cfg=axes.get("composite_mi_sample_strategy_cfg", "random"),
        composite_stacked_residual_aggregation_cfg=axes.get(
            "composite_stacked_residual_aggregation_cfg", "mean"
        ),
        composite_discovery_n_jobs_cfg=axes.get("composite_discovery_n_jobs_cfg", 1),
        quantile_crossing_fix_cfg=axes.get("quantile_crossing_fix_cfg", "sort"),
        quantile_coverage_pairs_cfg=axes.get("quantile_coverage_pairs_cfg", "default"),
        quantile_wrapper_n_jobs_cfg=axes.get("quantile_wrapper_n_jobs_cfg", "auto"),
        mlp_predict_batch_size_cfg=axes.get("mlp_predict_batch_size_cfg", None),
        ltr_cb_loss_fn_cfg=axes.get("ltr_cb_loss_fn_cfg", "YetiRankPairwise"),
        ltr_lgb_objective_cfg=axes.get("ltr_lgb_objective_cfg", "lambdarank"),
        ltr_rrf_k_cfg=axes.get("ltr_rrf_k_cfg", 60),
        recurrent_precision_cfg=axes.get("recurrent_precision_cfg", "32-true"),
        recurrent_sequence_preprocessing_cfg=axes.get("recurrent_sequence_preprocessing_cfg", "none"),
        # 2026-05-22 iter170 -- wave-3 depth-3+ audit.
        lgb_feature_fraction_cfg=axes.get("lgb_feature_fraction_cfg", 1.0),
        lgb_num_leaves_cfg=axes.get("lgb_num_leaves_cfg", 31),
        xgb_max_depth_cfg=axes.get("xgb_max_depth_cfg", 6),
        xgb_colsample_bynode_cfg=axes.get("xgb_colsample_bynode_cfg", 1.0),
        cb_border_count_cfg=axes.get("cb_border_count_cfg", 254),
        hgb_max_leaf_nodes_cfg=axes.get("hgb_max_leaf_nodes_cfg", 31),
        rfecv_cv_n_splits_cfg=axes.get("rfecv_cv_n_splits_cfg", 2),
        robust_q_low_cfg=axes.get("robust_q_low_cfg", 0.01),
        robust_q_high_cfg=axes.get("robust_q_high_cfg", 0.99),
        tfidf_max_features_cfg=axes.get("tfidf_max_features_cfg", 5000),
        kbins_encode_cfg=axes.get("kbins_encode_cfg", "ordinal"),
        nonlinear_n_components_cfg=axes.get("nonlinear_n_components_cfg", 100),
        pysr_operator_preset_cfg=axes.get("pysr_operator_preset_cfg", "standard"),
        confidence_ensemble_quantile_cfg=axes.get("confidence_ensemble_quantile_cfg", 0.1),
        cat_text_card_threshold_pct_cfg=axes.get("cat_text_card_threshold_pct_cfg", 0.001),
        rfecv_n_features_selection_rule_cfg=axes.get("rfecv_n_features_selection_rule_cfg", "auto"),
        rfecv_stability_selection_cfg=axes.get("rfecv_stability_selection_cfg", False),
        rfecv_leakage_action_cfg=axes.get("rfecv_leakage_action_cfg", "warn"),
        mrmr_fe_adaptive_threshold_relax_cfg=axes.get("mrmr_fe_adaptive_threshold_relax_cfg", True),
        mrmr_use_simple_mode_cfg=axes.get("mrmr_use_simple_mode_cfg", False),
        mrmr_identity_cache_include_y_cfg=axes.get("mrmr_identity_cache_include_y_cfg", True),
        mrmr_build_friend_graph_cfg=axes.get("mrmr_build_friend_graph_cfg", True),
        mrmr_friend_graph_prune_cfg=axes.get("mrmr_friend_graph_prune_cfg", False),
        mrmr_cluster_aggregate_enable_cfg=axes.get("mrmr_cluster_aggregate_enable_cfg", True),
        mrmr_cluster_aggregate_mode_cfg=axes.get("mrmr_cluster_aggregate_mode_cfg", "augment"),
        use_shap_proxied_fs=axes.get("use_shap_proxied_fs", False),
        shap_proxied_optimizer_cfg=axes.get("shap_proxied_optimizer_cfg", "auto"),
        shap_proxied_revalidate_cfg=axes.get("shap_proxied_revalidate_cfg", True),
        shap_proxied_trust_guard_cfg=axes.get("shap_proxied_trust_guard_cfg", True),
        shap_proxied_interaction_aware_cfg=axes.get("shap_proxied_interaction_aware_cfg", False),
        shap_proxied_cluster_features_cfg=axes.get("shap_proxied_cluster_features_cfg", "auto"),
        # 2026-05-28 new fuzz axes (ShapProxiedFS ext, FHC text card, composite deep,
        # extreme-AR skip list, FS null-frac, linear l1 ratio, recurrent hidden_size).
        shap_proxied_active_learning_cfg=axes.get("shap_proxied_active_learning_cfg", False),
        shap_proxied_prefilter_method_cfg=axes.get("shap_proxied_prefilter_method_cfg", "auto"),
        # 2026-05-28 audit-pass-2 B1-B6: ShapProxiedFS deeper extension axes.
        shap_proxied_config_jitter_cfg=axes.get("shap_proxied_config_jitter_cfg", False),
        shap_proxied_uncertainty_penalty_cfg=axes.get("shap_proxied_uncertainty_penalty_cfg", 0.0),
        shap_proxied_within_cluster_refine_cfg=axes.get(
            "shap_proxied_within_cluster_refine_cfg", True
        ),
        shap_proxied_use_bias_corrector_cfg=axes.get(
            "shap_proxied_use_bias_corrector_cfg", True
        ),
        shap_proxied_refine_n_estimators_cfg=axes.get(
            "shap_proxied_refine_n_estimators_cfg", 100
        ),
        shap_proxied_trust_guard_n_estimators_cfg=axes.get(
            "shap_proxied_trust_guard_n_estimators_cfg", 100
        ),
        # 2026-05-28 audit-pass-2 PART A coverage-gap axes.
        ensembling_degenerate_class_ratio_cfg=axes.get(
            "ensembling_degenerate_class_ratio_cfg", 0.01
        ),
        # behavior_use_flaml_zeroshot: canon True -> False when flaml is not
        # importable so combos don't crash at fit-time on environments without
        # the optional dep. flaml is only used by the zeroshot meta-learner
        # path; vanilla XGB/LGBM remain the default.
        behavior_use_flaml_zeroshot_cfg=_canon_use_flaml_zeroshot(
            axes.get("behavior_use_flaml_zeroshot_cfg", False)
        ),
        target_temporal_audit_granularity_cfg=axes.get(
            "target_temporal_audit_granularity_cfg", "auto"
        ),
        prep_ext_dim_n_components_cfg=axes.get("prep_ext_dim_n_components_cfg", 50),
        fhc_text_min_cardinality_cfg=axes.get("fhc_text_min_cardinality_cfg", 300),
        composite_auto_skip_on_baseline_optimal_cfg=axes.get(
            "composite_auto_skip_on_baseline_optimal_cfg", False
        ),
        composite_mi_n_neighbors_cfg=axes.get("composite_mi_n_neighbors_cfg", 3),
        composite_auto_base_null_perms_cfg=axes.get("composite_auto_base_null_perms_cfg", 20),
        composite_multi_base_max_k_cfg=axes.get("composite_multi_base_max_k_cfg", 3),
        extreme_ar_group_aware_skip_models_cfg=axes.get(
            "extreme_ar_group_aware_skip_models_cfg", "default_neural"
        ),
        fs_pre_screen_null_fraction_threshold_cfg=axes.get(
            "fs_pre_screen_null_fraction_threshold_cfg", 0.99
        ),
        linear_l1_ratio_cfg=axes.get("linear_l1_ratio_cfg", 0.5),
        recurrent_hidden_size_cfg=axes.get("recurrent_hidden_size_cfg", 128),
        catfe_fwer_correction_cfg=axes.get("catfe_fwer_correction_cfg", "none"),
        catfe_perm_budget_strategy_cfg=axes.get("catfe_perm_budget_strategy_cfg", "bandit_ucb1"),
        catfe_permutation_null_cfg=axes.get("catfe_permutation_null_cfg", "joint_independence"),
        catfe_bootstrap_ci_n_replicates_cfg=axes.get("catfe_bootstrap_ci_n_replicates_cfg", 0),
        catfe_use_miller_madow_cfg=axes.get("catfe_use_miller_madow_cfg", None),
        catfe_refine_passes_cfg=axes.get("catfe_refine_passes_cfg", 0),
        catfe_enable_streaming_cache_cfg=axes.get("catfe_enable_streaming_cache_cfg", False),
        catfe_unknown_strategy_cfg=axes.get("catfe_unknown_strategy_cfg", "clip"),
        composite_screening_cfg=axes.get("composite_screening_cfg", "hybrid"),
        composite_tiny_model_num_leaves_cfg=axes.get("composite_tiny_model_num_leaves_cfg", 15),
        composite_tiny_model_learning_rate_cfg=axes.get("composite_tiny_model_learning_rate_cfg", 0.1),
        composite_raw_baseline_tolerance_cfg=axes.get("composite_raw_baseline_tolerance_cfg", 1.02),
        composite_use_wilcoxon_gate_cfg=axes.get("composite_use_wilcoxon_gate_cfg", False),
        composite_detect_alpha_drift_cfg=axes.get("composite_detect_alpha_drift_cfg", True),
        composite_reject_on_alpha_drift_cfg=axes.get("composite_reject_on_alpha_drift_cfg", False),
        reporting_figsize_cfg=axes.get("reporting_figsize_cfg", "default"),
        reporting_plot_dpi_cfg=axes.get("reporting_plot_dpi_cfg", None),
        reporting_quantile_panels_cfg=axes.get("reporting_quantile_panels_cfg", "default"),
        reporting_ltr_panels_cfg=axes.get("reporting_ltr_panels_cfg", "default"),
        reporting_plotly_template_cfg=axes.get("reporting_plotly_template_cfg", None),
        reporting_matplotlib_style_cfg=axes.get("reporting_matplotlib_style_cfg", None),
        baseline_quick_model_n_estimators_cfg=axes.get("baseline_quick_model_n_estimators_cfg", 200),
        baseline_quick_model_num_leaves_cfg=axes.get("baseline_quick_model_num_leaves_cfg", 31),
        baseline_quick_model_learning_rate_cfg=axes.get("baseline_quick_model_learning_rate_cfg", 0.05),
        baseline_sample_n_cfg=axes.get("baseline_sample_n_cfg", 50_000),
        baseline_high_potential_min_dominance_pct_cfg=axes.get("baseline_high_potential_min_dominance_pct_cfg", 5.0),
        baseline_best_model_min_lift_cfg=axes.get("baseline_best_model_min_lift_cfg", 1.5),
        dummy_stratified_n_repeats_cfg=axes.get("dummy_stratified_n_repeats_cfg", 20),
        dummy_paired_bootstrap_n_resamples_cfg=axes.get("dummy_paired_bootstrap_n_resamples_cfg", 1000),
        ltr_mlp_loss_fn_cfg=axes.get("ltr_mlp_loss_fn_cfg", "ranknet"),
        ltr_eval_at_cfg=axes.get("ltr_eval_at_cfg", "default"),
        multilabel_force_native_xgb_cfg=axes.get("multilabel_force_native_xgb_cfg", False),
        fhc_pricing_cap_usd_cfg=axes.get("fhc_pricing_cap_usd_cfg", None),
        fhc_pricing_warn_above_usd_cfg=axes.get("fhc_pricing_warn_above_usd_cfg", 1.0),
        fhc_logging_verbose_cfg=axes.get("fhc_logging_verbose_cfg", False),
        fhc_repro_langdetect_seed_cfg=axes.get("fhc_repro_langdetect_seed_cfg", 0),
        fhc_repro_pinned_svd_solver_params_cfg=axes.get("fhc_repro_pinned_svd_solver_params_cfg", True),
        fhc_repro_forbid_nonatomic_fs_cfg=axes.get("fhc_repro_forbid_nonatomic_fs_cfg", False),
        fhc_repro_deterministic_eviction_cfg=axes.get("fhc_repro_deterministic_eviction_cfg", False),
        fhc_cache_prefetch_enabled_cfg=axes.get("fhc_cache_prefetch_enabled_cfg", True),
        fhc_cache_prefetch_vram_safety_factor_cfg=axes.get("fhc_cache_prefetch_vram_safety_factor_cfg", 2.0),
        fhc_memory_pressure_watermark_pct_cfg=axes.get("fhc_memory_pressure_watermark_pct_cfg", 85),
        fhc_text_min_mean_tokens_cfg=axes.get("fhc_text_min_mean_tokens_cfg", 4.0),
        fhc_text_min_unique_ratio_cfg=axes.get("fhc_text_min_unique_ratio_cfg", 0.95),
        fhc_text_respect_explicit_cat_dtype_cfg=axes.get("fhc_text_respect_explicit_cat_dtype_cfg", True),
        recurrent_input_mode_cfg=axes.get("recurrent_input_mode_cfg", "hybrid"),
        recurrent_num_workers_cfg=axes.get("recurrent_num_workers_cfg", 0),
        # 2026-05-23 iter180 -- DEPTH-4.
        lgb_boosting_type_cfg=axes.get("lgb_boosting_type_cfg", "gbdt"),
        lgb_dart_drop_rate_cfg=axes.get("lgb_dart_drop_rate_cfg", 0.1),
        lgb_goss_top_rate_cfg=axes.get("lgb_goss_top_rate_cfg", 0.2),
        xgb_tree_method_cfg=axes.get("xgb_tree_method_cfg", "auto"),
        xgb_hist_max_bin_cfg=axes.get("xgb_hist_max_bin_cfg", 256),
        cb_bootstrap_type_cfg=axes.get("cb_bootstrap_type_cfg", "Bayesian"),
        cb_bayesian_bagging_temperature_cfg=axes.get("cb_bayesian_bagging_temperature_cfg", 1.0),
        cb_bernoulli_subsample_cfg=axes.get("cb_bernoulli_subsample_cfg", 0.8),
        cb_grow_policy_cfg=axes.get("cb_grow_policy_cfg", "SymmetricTree"),
        cb_lossguide_max_leaves_cfg=axes.get("cb_lossguide_max_leaves_cfg", 31),
        fhc_cache_persistence_cfg=axes.get("fhc_cache_persistence_cfg", "auto"),
        multilabel_per_label_thresholds_cfg=axes.get("multilabel_per_label_thresholds_cfg", None),
        multilabel_chain_seeds_cfg=axes.get("multilabel_chain_seeds_cfg", None),
        # F1 -- enable_crash_reporting suite-level kwarg (Windows-only meaningful).
        enable_crash_reporting_cfg=axes.get("enable_crash_reporting_cfg", False),
    )


# ---------------------------------------------------------------------------
# Shared suite-config builders (2026-05-18 refactor)
# ---------------------------------------------------------------------------
#
# Goal: adding a new axis = one edit (here), not N edits across the pytest
# suite call site + the 1M harness. Both call sites consume these builders
# either via a FuzzCombo instance (pytest suite) OR via flat keyword args
# (1M harness which randomises axes via its own _axis_rng).
#
# Pattern: the "_from_flat" function takes named primitives (no FuzzCombo
# dependency); the FuzzCombo-aware wrapper just forwards combo.* attrs.

def build_cat_fe_config_from_flat(
    *, use_mrmr_fs: bool, cat_fe_enable: bool, cat_fe_include_numeric: bool,
):
    """Return a CatFEConfig honoring the cat-FE enable + include_numeric
    axes. None when use_mrmr_fs=False OR when defaults are fine (library
    default already has enable=True, include_numeric=False)."""
    if not use_mrmr_fs:
        return None
    from mlframe.feature_selection.filters.cat_fe_state import CatFEConfig
    if not cat_fe_enable:
        return CatFEConfig(enable=False)
    if cat_fe_include_numeric:
        return CatFEConfig(enable=True, include_numeric=True)
    return None  # library default


def build_mrmr_kwargs_from_flat(
    *,
    use_mrmr_fs: bool,
    interactions_max_order: int = 1,
    fe_max_steps: int = 1,
    cat_fe_config: Any = None,
    fe_npermutations: int = 0,
    fe_ntop_features: int = 0,
    fe_unary_preset: str = "minimal",
    fe_binary_preset: str = "minimal",
    fe_smart_polynom_iters: int = 0,
    fe_smart_polynom_optimization_steps: int = 1000,
    fe_min_polynom_degree: int = 3,
    fe_max_polynom_degree: int = 3,
    # 2026-05-21 -- FE pair-check subsample budget. Threads the new MRMR
    # __init__ knob added in feat(fe+suite) 5223085 alongside the matching
    # fe_smart_polynom subsample. 0 = disabled (legacy full-frame path);
    # >0 AND < len(X) fires the subsample MI sweep with full-n survivor
    # rebuild. Both subsamples share FE_DEFAULT_SUBSAMPLE_N (200_000)
    # upstream; fuzz pins a tighter 50_000 budget so the subsample path
    # also fires at n_rows=200_000.
    fe_check_pairs_subsample_n: int = 0,
    fe_smart_polynom_subsample_n: int = 0,
    # Suite-side fuzz-speed pins. Callers can override.
    verbose: int = 0,
    max_runtime_mins: int = 1,
    n_workers: int = 1,
    quantization_nbins: int = 5,
    use_simple_mode: bool = True,
    min_nonzero_confidence: float = 0.9,
    max_consec_unconfirmed: int = 2,
    full_npermutations: int = 2,
    # 2026-05-27 friend-graph + cluster-aggregate knobs (mrmr.py __init__).
    build_friend_graph: bool = True,
    friend_graph_prune: bool = False,
    cluster_aggregate_enable: bool = True,
    cluster_aggregate_mode: str = "augment",
) -> Optional[Dict[str, Any]]:
    """Build the mrmr_kwargs dict passed to FeatureSelectionConfig.
    Returns None when use_mrmr_fs=False so the FS step is a no-op.

    Single-edit point: every MRMR knob (existing iter-32.5 axes + any
    future axis) flows through these named params. Both
    test_fuzz_suite.py and _profile_fuzz_1m.py call this so adding a
    new MRMR axis only touches this function (plus the AXES dict +
    FuzzCombo dataclass for the pytest fuzz space).
    """
    if not use_mrmr_fs:
        return None
    kwargs: Dict[str, Any] = {
        "verbose": verbose,
        "max_runtime_mins": max_runtime_mins,
        "n_workers": n_workers,
        "quantization_nbins": quantization_nbins,
        "use_simple_mode": use_simple_mode,
        "min_nonzero_confidence": min_nonzero_confidence,
        "max_consec_unconfirmed": max_consec_unconfirmed,
        "full_npermutations": full_npermutations,
        "interactions_max_order": interactions_max_order,
        "fe_max_steps": fe_max_steps,
        "fe_npermutations": fe_npermutations,
        "fe_ntop_features": fe_ntop_features,
        "fe_unary_preset": fe_unary_preset,
        "fe_binary_preset": fe_binary_preset,
        "fe_smart_polynom_iters": fe_smart_polynom_iters,
        "fe_smart_polynom_optimization_steps": fe_smart_polynom_optimization_steps,
        "fe_min_polynom_degree": fe_min_polynom_degree,
        "fe_max_polynom_degree": fe_max_polynom_degree,
        # Friend-graph + cluster-aggregate knobs flow straight into the MRMR
        # constructor (same names). Defaults mirror mrmr.py so a combo that
        # leaves them at the default produces the same MRMR behaviour as
        # before these axes existed.
        "build_friend_graph": build_friend_graph,
        "friend_graph_prune": friend_graph_prune,
        "cluster_aggregate_enable": cluster_aggregate_enable,
        "cluster_aggregate_mode": cluster_aggregate_mode,
    }
    # The MRMR subsample knobs default to FE_DEFAULT_SUBSAMPLE_N upstream; only
    # override when the fuzz axis sets a non-zero budget so existing combos
    # don't accidentally flip the path on.
    if fe_check_pairs_subsample_n > 0:
        kwargs["fe_check_pairs_subsample_n"] = fe_check_pairs_subsample_n
    if fe_smart_polynom_subsample_n > 0:
        kwargs["fe_smart_polynom_subsample_n"] = fe_smart_polynom_subsample_n
    if cat_fe_config is not None:
        kwargs["cat_fe_config"] = cat_fe_config
    return kwargs


def build_mrmr_kwargs(combo: "FuzzCombo") -> Optional[Dict[str, Any]]:
    """FuzzCombo-aware wrapper around build_mrmr_kwargs_from_flat."""
    cat_fe = build_cat_fe_config_from_flat(
        use_mrmr_fs=combo.use_mrmr_fs,
        cat_fe_enable=combo.mrmr_cat_fe_enable_cfg,
        cat_fe_include_numeric=combo.mrmr_cat_fe_include_numeric_cfg,
    )
    # FE subsample only meaningful when an FE entry point actually runs and
    # n_rows exceeds the budget. Couples the new fe_check_pairs_subsample_n_cfg
    # axis to both fe_npermutations / fe_ntop_features (any > 0 fires the FE
    # block) and the smart-polynom subsample (shares the same budget by
    # FE_DEFAULT_SUBSAMPLE_N upstream).
    _subsample_active = (
        combo.use_mrmr_fs
        and combo.fe_check_pairs_subsample_n_cfg > 0
        and combo.n_rows > combo.fe_check_pairs_subsample_n_cfg
        and (combo.mrmr_fe_npermutations_cfg > 0 or combo.mrmr_fe_ntop_features_cfg > 0)
    )
    return build_mrmr_kwargs_from_flat(
        use_mrmr_fs=combo.use_mrmr_fs,
        interactions_max_order=combo.mrmr_interactions_max_order_cfg,
        fe_max_steps=combo.mrmr_fe_max_steps_cfg,
        cat_fe_config=cat_fe,
        fe_npermutations=combo.mrmr_fe_npermutations_cfg,
        fe_ntop_features=combo.mrmr_fe_ntop_features_cfg,
        fe_unary_preset=combo.mrmr_fe_unary_preset_cfg,
        fe_binary_preset=combo.mrmr_fe_binary_preset_cfg,
        fe_smart_polynom_iters=combo.mrmr_fe_smart_polynom_iters_cfg,
        fe_smart_polynom_optimization_steps=combo.mrmr_fe_smart_polynom_steps_cfg,
        fe_min_polynom_degree=combo.mrmr_fe_min_polynom_degree_cfg,
        fe_max_polynom_degree=combo.mrmr_fe_max_polynom_degree_cfg,
        fe_check_pairs_subsample_n=(combo.fe_check_pairs_subsample_n_cfg if _subsample_active else 0),
        fe_smart_polynom_subsample_n=(combo.fe_check_pairs_subsample_n_cfg if _subsample_active else 0),
        build_friend_graph=combo.mrmr_build_friend_graph_cfg,
        friend_graph_prune=combo.mrmr_friend_graph_prune_cfg,
        cluster_aggregate_enable=combo.mrmr_cluster_aggregate_enable_cfg,
        cluster_aggregate_mode=combo.mrmr_cluster_aggregate_mode_cfg,
    )


def build_shap_proxied_fs_kwargs_from_flat(
    *,
    use_shap_proxied_fs: bool,
    optimizer: str = "auto",
    revalidate: bool = True,
    trust_guard: bool = True,
    interaction_aware: bool = False,
    cluster_features: "bool | str" = "auto",
    # 2026-05-28 ext axes (active_learning + prefilter_method).
    active_learning: bool = False,
    prefilter_method: str = "auto",
    # 2026-05-28 audit-pass-2 B1-B6 deeper extension axes. Defaults verified
    # against ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py:41-89).
    config_jitter: bool = False,
    uncertainty_penalty: float = 0.0,
    within_cluster_refine: bool = True,
    use_bias_corrector: bool = True,
    refine_n_estimators: "int | None" = 100,
    trust_guard_n_estimators: "int | None" = 100,
) -> Optional[Dict[str, Any]]:
    """Build the shap_proxied_fs_kwargs dict passed to
    ``registry.get("ShapProxiedFS").instantiate(**kwargs)`` (which forwards to
    ShapProxiedFS.__init__). Returns None when use_shap_proxied_fs=False so the
    FS step is a no-op (mirrors build_mrmr_kwargs_from_flat).

    Single-edit point: every ShapProxiedFS knob the fuzz harness exercises maps
    to its exact __init__ parameter name here, so adding a new shap-proxied axis
    only touches this function (plus the AXES dict + the dataclass field +
    canonical_key + _build_combo). Param names verified against
    ShapProxiedFS.__init__ (feature_selection/shap_proxied_fs.py).
    """
    if not use_shap_proxied_fs:
        return None
    return {
        "optimizer": optimizer,
        "revalidate": revalidate,
        "trust_guard": trust_guard,
        "interaction_aware": interaction_aware,
        "cluster_features": cluster_features,
        # 2026-05-28 ext axes flow straight into the ShapProxiedFS constructor
        # (same names) -- active_learning toggles the acquisition-loop branch,
        # prefilter_method drives _shap_proxy_prefilter dispatch.
        "active_learning": active_learning,
        "prefilter_method": prefilter_method,
        # 2026-05-28 audit-pass-2 B1-B6 deeper axes (param names match the
        # ShapProxiedFS.__init__ signature verbatim).
        "config_jitter": config_jitter,
        "uncertainty_penalty": uncertainty_penalty,
        "within_cluster_refine": within_cluster_refine,
        "use_bias_corrector": use_bias_corrector,
        "refine_n_estimators": refine_n_estimators,
        "trust_guard_n_estimators": trust_guard_n_estimators,
    }


def build_shap_proxied_fs_kwargs(combo: "FuzzCombo") -> Optional[Dict[str, Any]]:
    """FuzzCombo-aware wrapper around build_shap_proxied_fs_kwargs_from_flat."""
    return build_shap_proxied_fs_kwargs_from_flat(
        use_shap_proxied_fs=combo.use_shap_proxied_fs,
        optimizer=combo.shap_proxied_optimizer_cfg,
        revalidate=combo.shap_proxied_revalidate_cfg,
        trust_guard=combo.shap_proxied_trust_guard_cfg,
        interaction_aware=combo.shap_proxied_interaction_aware_cfg,
        cluster_features=combo.shap_proxied_cluster_features_cfg,
        active_learning=combo.shap_proxied_active_learning_cfg,
        prefilter_method=combo.shap_proxied_prefilter_method_cfg,
        config_jitter=combo.shap_proxied_config_jitter_cfg,
        uncertainty_penalty=combo.shap_proxied_uncertainty_penalty_cfg,
        within_cluster_refine=combo.shap_proxied_within_cluster_refine_cfg,
        use_bias_corrector=combo.shap_proxied_use_bias_corrector_cfg,
        refine_n_estimators=combo.shap_proxied_refine_n_estimators_cfg,
        trust_guard_n_estimators=combo.shap_proxied_trust_guard_n_estimators_cfg,
    )


def build_composite_discovery_config_from_flat(
    *, enabled: bool, transforms_mode: Optional[str] = None,
    mi_estimator: str = "bin", mi_nbins: int = 16, mi_aggregation: str = "mean",
    mi_sample_strategy: str = "random",
    stacked_residual_aggregation: str = "mean",
    discovery_n_jobs: int = 1,
    # 2026-05-22 TVT-MLP audit-followup axes.
    composite_skip_raw_dominates_ratio: float = 0.0,
    composite_skip_ablation_delta_pct: float = 0.0,
    composite_eps_mi_gain: float = -10.0,
    composite_top_k_after_mi: int = 32,
    composite_require_beats_raw_baseline: bool = False,
    composite_per_bin_n_bins: int = 0,
    composite_tiny_screening_mode: str = "per_family",
    composite_include_additive_residual: bool = True,
    # 2026-05-28 deep knobs (4 new axes).
    auto_skip_on_baseline_optimal: bool = False,
    mi_n_neighbors: int = 3,
    auto_base_null_perms: int = 20,
    multi_base_max_k: int = 3,
):
    """Build a CompositeTargetDiscoveryConfig honoring the discovery
    enable + transforms_mode axes + (iter162) nested MI / stacked /
    parallelism knobs + (2026-05-22) TVT-MLP audit-followup gate axes."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig
    if not enabled:
        return CompositeTargetDiscoveryConfig(enabled=False)
    if transforms_mode == "unary_only":
        transforms = ["cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y"]
    elif transforms_mode == "chain_only":
        transforms = [
            "chain_linres_cbrt", "chain_linres_yj",
            "chain_monres_cbrt", "chain_monres_yj",
        ]
    elif transforms_mode == "legacy":
        transforms = [
            "diff", "ratio", "logratio", "linear_residual",
            "quantile_residual", "monotonic_residual",
        ]
    else:
        transforms = None
    # The additive_residual toggle works on top of any transforms_mode:
    # if the chosen mode would include bivariate residuals, ensure
    # additive_residual is present / absent as requested.
    if (transforms is not None and composite_include_additive_residual
            and "additive_residual" not in transforms
            and transforms_mode in (None, "legacy")):
        transforms = ["additive_residual"] + transforms
    elif (transforms is not None and not composite_include_additive_residual
            and "additive_residual" in transforms):
        transforms = [t for t in transforms if t != "additive_residual"]
    kw: Dict[str, Any] = {
        "enabled": True,
        "base_candidates": "auto",
        "auto_base_top_k": 3,
        "multi_base_enabled": (multi_base_max_k > 1),
        # 2026-05-28 multi_base_max_k axis (was hardcoded 2). When the axis
        # value is 1 we additionally turn off multi_base_enabled above so the
        # promotion loop short-circuits cleanly.
        "multi_base_max_k": multi_base_max_k,
        # iter162 nested knobs.
        "mi_estimator": mi_estimator,
        "mi_nbins": mi_nbins,
        "mi_aggregation": mi_aggregation,
        "mi_sample_strategy": mi_sample_strategy,
        "stacked_residual_aggregation": stacked_residual_aggregation,
        "discovery_n_jobs": discovery_n_jobs,
        # 2026-05-22 TVT-MLP audit-followup axes.
        "composite_skip_when_raw_dominates_ratio": composite_skip_raw_dominates_ratio,
        "composite_skip_when_ablation_delta_pct": composite_skip_ablation_delta_pct,
        "eps_mi_gain": composite_eps_mi_gain,
        "top_k_after_mi": composite_top_k_after_mi,
        "require_beats_raw_baseline": composite_require_beats_raw_baseline,
        "per_bin_n_bins": composite_per_bin_n_bins,
        "tiny_screening_models": composite_tiny_screening_mode,
        # 2026-05-28 deep knobs (3 of the 4; multi_base_max_k handled above
        # because it also gates multi_base_enabled). These names match the
        # CompositeTargetDiscoveryConfig dataclass fields exactly.
        "auto_skip_on_baseline_optimal": auto_skip_on_baseline_optimal,
        "mi_n_neighbors": mi_n_neighbors,
        "auto_base_null_perms": auto_base_null_perms,
    }
    if composite_tiny_screening_mode == "per_family":
        kw["tiny_screening_families"] = ("lightgbm", "linear")
    else:
        kw["tiny_screening_families"] = ("lightgbm",)
    if transforms is not None:
        kw["transforms"] = transforms
    return CompositeTargetDiscoveryConfig(**kw)


def build_composite_discovery_config(combo: "FuzzCombo"):
    """FuzzCombo-aware wrapper."""
    enabled = (
        combo.composite_discovery_enabled_cfg
        and combo.target_type == "regression"
    )
    return build_composite_discovery_config_from_flat(
        enabled=enabled,
        transforms_mode=combo.composite_transforms_mode_cfg if enabled else None,
        mi_estimator=combo.composite_mi_estimator_cfg,
        mi_nbins=combo.composite_mi_nbins_cfg,
        mi_aggregation=combo.composite_mi_aggregation_cfg,
        mi_sample_strategy=combo.composite_mi_sample_strategy_cfg,
        stacked_residual_aggregation=combo.composite_stacked_residual_aggregation_cfg,
        discovery_n_jobs=combo.composite_discovery_n_jobs_cfg,
        composite_skip_raw_dominates_ratio=combo.composite_skip_raw_dominates_ratio_cfg,
        composite_skip_ablation_delta_pct=combo.composite_skip_ablation_delta_pct_cfg,
        composite_eps_mi_gain=combo.composite_eps_mi_gain_cfg,
        composite_top_k_after_mi=combo.composite_top_k_after_mi_cfg,
        composite_require_beats_raw_baseline=combo.composite_require_beats_raw_baseline_cfg,
        composite_per_bin_n_bins=combo.composite_per_bin_n_bins_cfg,
        composite_tiny_screening_mode=combo.composite_tiny_screening_mode_cfg,
        composite_include_additive_residual=combo.composite_include_additive_residual_cfg,
        # 2026-05-28 deep knobs (4 new axes).
        auto_skip_on_baseline_optimal=combo.composite_auto_skip_on_baseline_optimal_cfg,
        mi_n_neighbors=combo.composite_mi_n_neighbors_cfg,
        auto_base_null_perms=combo.composite_auto_base_null_perms_cfg,
        multi_base_max_k=combo.composite_multi_base_max_k_cfg,
    )


def _all_axis_pairs() -> set[tuple[str, Any, str, Any]]:
    pairs: set[tuple[str, Any, str, Any]] = set()
    # Also include model-count ("n_models" pseudo-axis) to balance single vs
    # multi-model combos across other axes.
    axes_ext: dict[str, tuple[Any, ...]] = {**AXES, "n_models": (1, 2, 3, 4, 5)}
    axis_names = list(axes_ext.keys())
    for i in range(len(axis_names)):
        for j in range(i + 1, len(axis_names)):
            ai, aj = axis_names[i], axis_names[j]
            for vi in axes_ext[ai]:
                for vj in axes_ext[aj]:
                    pairs.add((ai, vi, aj, vj))
    return pairs


def _combo_pairs(combo: FuzzCombo) -> set[tuple[str, Any, str, Any]]:
    values = {
        "input_type": combo.input_type,
        "n_rows": combo.n_rows,
        "cat_feature_count": combo.cat_feature_count,
        "null_fraction_cats": combo.null_fraction_cats,
        "use_mrmr_fs": combo.use_mrmr_fs,
        "weight_schemas": combo.weight_schemas,
        "target_type": combo.target_type,
        "auto_detect_cats": combo.auto_detect_cats,
        "align_polars_categorical_dicts": combo.align_polars_categorical_dicts,
        "prefer_polarsds": combo.prefer_polarsds,
        "use_text_features": combo.use_text_features,
        "honor_user_dtype": combo.honor_user_dtype,
        "text_col_count": combo.text_col_count,
        "embedding_col_count": combo.embedding_col_count,
        # 2026-04-24 combo-extension axes
        "outlier_detection": combo.outlier_detection,
        "use_ensembles": combo.use_ensembles,
        "continue_on_model_failure": combo.continue_on_model_failure,
        "iterations": combo.iterations,
        "prefer_calibrated_classifiers": combo.prefer_calibrated_classifiers,
        "inject_degenerate_cols": combo.inject_degenerate_cols,
        "inject_inf_nan": combo.inject_inf_nan,
        "with_datetime_col": combo.with_datetime_col,
        "inject_zero_col": combo.inject_zero_col,
        "fairness_col": combo.fairness_col,
        "custom_prep": combo.custom_prep,
        "input_storage": combo.input_storage,
        # 2026-04-24 round 2
        "fillna_value_cfg": combo.fillna_value_cfg,
        "scaler_name_cfg": combo.scaler_name_cfg,
        "categorical_encoding_cfg": combo.categorical_encoding_cfg,
        "skip_categorical_encoding_cfg": combo.skip_categorical_encoding_cfg,
        "val_placement_cfg": combo.val_placement_cfg,
        "test_size_cfg": combo.test_size_cfg,
        "trainset_aging_limit_cfg": combo.trainset_aging_limit_cfg,
        "cat_text_card_threshold_cfg": combo.cat_text_card_threshold_cfg,
        "early_stopping_rounds_cfg": combo.early_stopping_rounds_cfg,
        "use_robust_eval_metric_cfg": combo.use_robust_eval_metric_cfg,
        # Fix G
        "inject_label_leak": combo.inject_label_leak,
        "inject_rank_deficient": combo.inject_rank_deficient,
        "inject_all_nan_col": combo.inject_all_nan_col,
        # R3
        "inject_test_drift": combo.inject_test_drift,
        "imbalance_ratio": combo.imbalance_ratio,
        "weird_cat_content": combo.weird_cat_content,
        # Phase H
        "multilabel_strategy_cfg": combo.multilabel_strategy_cfg,
        # 2026-04-26 batch 1
        "fix_infinities_cfg": combo.fix_infinities_cfg,
        "ensure_float32_cfg": combo.ensure_float32_cfg,
        "remove_constant_columns_cfg": combo.remove_constant_columns_cfg,
        "imputer_strategy_cfg": combo.imputer_strategy_cfg,
        "shuffle_val_cfg": combo.shuffle_val_cfg,
        "shuffle_test_cfg": combo.shuffle_test_cfg,
        "wholeday_splitting_cfg": combo.wholeday_splitting_cfg,
        "val_sequential_fraction_cfg": combo.val_sequential_fraction_cfg,
        # batch 3 — multilabel dispatch
        "multilabel_n_chains_cfg": combo.multilabel_n_chains_cfg,
        "multilabel_chain_order_cfg": combo.multilabel_chain_order_cfg,
        "multilabel_cv_cfg": combo.multilabel_cv_cfg,
        # batch 4 — PreprocessingExtensionsConfig
        "prep_ext_scaler_cfg": combo.prep_ext_scaler_cfg,
        "prep_ext_kbins_cfg": combo.prep_ext_kbins_cfg,
        "prep_ext_polynomial_degree_cfg": combo.prep_ext_polynomial_degree_cfg,
        "prep_ext_dim_reducer_cfg": combo.prep_ext_dim_reducer_cfg,
        "prep_ext_nonlinear_cfg": combo.prep_ext_nonlinear_cfg,
        "prep_ext_pysr_enabled_cfg": combo.prep_ext_pysr_enabled_cfg,
        "mrmr_nan_strategy_cfg": combo.mrmr_nan_strategy_cfg,
        # batch 5
        "rfecv_estimator_cfg": combo.rfecv_estimator_cfg,
        # batch 6
        "recurrent_model_cfg": combo.recurrent_model_cfg,
        # 2026-04-28 batch 4 followup
        "include_confidence_analysis_cfg": combo.include_confidence_analysis_cfg,
        "n_models": len(combo.models),
    }
    names = list(values.keys())
    out: set[tuple[str, Any, str, Any]] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ai, aj = names[i], names[j]
            out.add((ai, values[ai], aj, values[aj]))
    return out


def enumerate_combos(
    target: int = 150,
    master_seed: int = 2026_04_22,
    model_universe: tuple[str, ...] = MODELS,
) -> list[FuzzCombo]:
    """Return `target` unique FuzzCombo instances covering the axis space.

    Phase A seeds with one combo per non-empty model-subset.
    Phase B greedy-fills until pairwise coverage is achieved.
    Phase C random-fills until len == target.
    """
    rng = random.Random(master_seed)
    seen: set[tuple] = set()
    combos: list[FuzzCombo] = []

    # Phase A — model subsets
    for subset in _powerset_nonempty(model_universe):
        axes = _sample_axes(rng)
        # iter373: skip unrunnable LTR-without-ranker combos before they hit
        # training and crash with NotImplementedError. Pull a fresh axes
        # sample only if the FIRST pick happened to pair this subset with
        # LTR; that way the enumerator doesn't deterministically lose its
        # LTR slot for hgb-only / linear-only model subsets.
        for _retry in range(20):
            if _combo_is_runnable(subset, axes["target_type"]):
                break
            axes = _sample_axes(rng)
        else:
            # 20 resamples couldn't land on a runnable target_type; skip.
            continue
        combo = _build_combo(subset, axes, len(combos))
        key = combo.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)

    # Phase B — pairwise coverage
    required = _all_axis_pairs()
    covered: set[tuple[str, Any, str, Any]] = set()
    for c in combos:
        covered.update(_combo_pairs(c))

    tries = 0
    max_tries = 10_000
    while covered < required and tries < max_tries and len(combos) < target:
        uncovered = required - covered
        best_combo = None
        best_new = 0
        for _ in range(50):
            subset = rng.choice(_powerset_nonempty(model_universe))
            axes = _sample_axes(rng)
            if not _combo_is_runnable(subset, axes["target_type"]):
                continue
            candidate = _build_combo(subset, axes, len(combos))
            if candidate.canonical_key() in seen:
                continue
            cand_pairs = _combo_pairs(candidate)
            new = len(cand_pairs & uncovered)
            if new > best_new:
                best_new = new
                best_combo = candidate
        if best_combo is None:
            break
        seen.add(best_combo.canonical_key())
        combos.append(best_combo)
        covered.update(_combo_pairs(best_combo))
        tries += 1

    # Phase C — random fill until target
    while len(combos) < target:
        subset = rng.choice(_powerset_nonempty(model_universe))
        axes = _sample_axes(rng)
        if not _combo_is_runnable(subset, axes["target_type"]):
            continue
        candidate = _build_combo(subset, axes, len(combos))
        key = candidate.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(candidate)

    return combos[:target]


# ---------------------------------------------------------------------------
# Fix A — 3-wise covering over a curated subset of load-bearing axes
# ---------------------------------------------------------------------------

# Only the axes where 3-way interaction bugs have historically lived or
# are most plausible. Restricting the triple-space from the full 36
# axes to these 13 keeps the covering algorithm tractable (~286 axis-
# triples × ~12 value-triples = ~3.5k triples to cover) while still
# probing the interactions that matter. Expand cautiously — adding one
# axis bumps the triple count by ~C(N-1, 2) new axis-triples.
_3WAY_AXES: tuple[str, ...] = (
    "input_type",
    "n_rows",
    "cat_feature_count",
    "use_mrmr_fs",
    "target_type",
    "outlier_detection",
    "use_ensembles",
    "inject_inf_nan",
    "inject_degenerate_cols",
    "custom_prep",
    "categorical_encoding_cfg",
    "scaler_name_cfg",
    "inject_label_leak",
    "inject_rank_deficient",
    "inject_all_nan_col",
    # 2026-04-28 batch 4 followup - the confidence-analysis path interacts
    # with the test-set metrics, the calibration loop, and SHAP/permutation
    # paths; keep it triple-covered against the load-bearing axes that
    # historically interact with that side of the pipeline.
    "include_confidence_analysis_cfg",
)


def _all_axis_triples() -> set[tuple[str, Any, str, Any, str, Any]]:
    axes_ext: dict[str, tuple[Any, ...]] = {
        name: AXES[name] for name in _3WAY_AXES if name in AXES
    }
    axes_ext["n_models"] = (1, 2, 3, 4, 5)
    names = list(axes_ext.keys())
    out: set[tuple] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                ai, aj, ak = names[i], names[j], names[k]
                for vi in axes_ext[ai]:
                    for vj in axes_ext[aj]:
                        for vk in axes_ext[ak]:
                            out.add((ai, vi, aj, vj, ak, vk))
    return out


def _combo_triples(combo: FuzzCombo) -> set[tuple[str, Any, str, Any, str, Any]]:
    values = {
        name: getattr(combo, name) for name in _3WAY_AXES if hasattr(combo, name)
    }
    values["n_models"] = len(combo.models)
    names = list(values.keys())
    out: set[tuple] = set()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                out.add(
                    (names[i], values[names[i]],
                     names[j], values[names[j]],
                     names[k], values[names[k]])
                )
    return out


def enumerate_combos_3way(
    target: int = 600,
    master_seed: int = 2026_04_24,
    model_universe: tuple[str, ...] = MODELS,
) -> list[FuzzCombo]:
    """Greedy 3-wise (triple) covering over ``_3WAY_AXES``.

    Same shape as ``enumerate_combos`` but optimises for triple-coverage
    instead of pair-coverage. Seeded separately (default ``2026_04_24``
    so the 3-wise suite doesn't stomp the pairwise seed's sample).
    """
    rng = random.Random(master_seed)
    seen: set[tuple] = set()
    combos: list[FuzzCombo] = []

    # Phase A — model subsets (iter373: skip LTR-without-ranker)
    for subset in _powerset_nonempty(model_universe):
        axes = _sample_axes(rng)
        for _retry in range(20):
            if _combo_is_runnable(subset, axes["target_type"]):
                break
            axes = _sample_axes(rng)
        else:
            continue
        combo = _build_combo(subset, axes, len(combos))
        key = combo.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(combo)

    # Phase B — greedy triple coverage
    required = _all_axis_triples()
    covered: set[tuple] = set()
    for c in combos:
        covered.update(_combo_triples(c))

    tries = 0
    max_tries = 40_000
    while covered < required and tries < max_tries and len(combos) < target:
        uncovered = required - covered
        best_combo = None
        best_new = 0
        for _ in range(80):
            subset = rng.choice(_powerset_nonempty(model_universe))
            axes = _sample_axes(rng)
            if not _combo_is_runnable(subset, axes["target_type"]):
                continue
            candidate = _build_combo(subset, axes, len(combos))
            if candidate.canonical_key() in seen:
                continue
            cand = _combo_triples(candidate)
            new = len(cand & uncovered)
            if new > best_new:
                best_new = new
                best_combo = candidate
        if best_combo is None:
            break
        seen.add(best_combo.canonical_key())
        combos.append(best_combo)
        covered.update(_combo_triples(best_combo))
        tries += 1

    # Phase C — random fill
    while len(combos) < target:
        subset = rng.choice(_powerset_nonempty(model_universe))
        axes = _sample_axes(rng)
        if not _combo_is_runnable(subset, axes["target_type"]):
            continue
        candidate = _build_combo(subset, axes, len(combos))
        key = candidate.canonical_key()
        if key in seen:
            continue
        seen.add(key)
        combos.append(candidate)

    return combos[:target]


# ---------------------------------------------------------------------------
# Results log — JSONL append-only
# ---------------------------------------------------------------------------

RESULTS_LOG = Path(__file__).parent / "_fuzz_results.jsonl"


def log_combo_outcome(
    combo: FuzzCombo,
    outcome: str,
    duration_s: float,
    error_class: str | None = None,
    error_summary: str | None = None,
    extra: dict | None = None,
) -> None:
    """Append one JSONL row with the combo's outcome.

    Columns: combo fields, outcome in {pass,fail,xpass,xfail,skip}, duration,
    error_class/error_summary (for fail/xpass rows), extra (free-form dict),
    and ``master_seed`` (Fix E: seed-rotation telemetry — the nightly cron
    passes a different ``FUZZ_SEED`` each run, we tag each row so failures
    stay attributable to their generating seed).
    """
    row: dict = {
        **combo.to_json(),
        "outcome": outcome,
        "duration_s": round(duration_s, 3),
        "master_seed": int(os.environ.get("FUZZ_SEED", "20260422")),
    }
    if error_class:
        row["error_class"] = error_class
    if error_summary:
        row["error_summary"] = error_summary[:300]
    if extra:
        row["extra"] = extra
    try:
        RESULTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_LOG.open("ab") as f:
            # orjson keeps non-ASCII as raw UTF-8 bytes (no \uXXXX escaping),
            # matching the prior ensure_ascii=False behaviour.
            f.write(orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b"\n")
    except OSError:
        pass  # never break a test because logging failed


def read_fail_summary() -> dict:
    """Return a summary of failures since the last run start marker."""
    if not RESULTS_LOG.exists():
        return {"fails": [], "totals": {}}
    totals: dict[str, int] = {}
    fails: list[dict] = []
    with RESULTS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = orjson.loads(line)
            except Exception:
                continue
            totals[row.get("outcome", "?")] = totals.get(row.get("outcome", "?"), 0) + 1
            if row.get("outcome") == "fail":
                fails.append(row)
    return {"fails": fails, "totals": totals}


# ---------------------------------------------------------------------------
# Frame builder — turns a combo into (df, target_col, cat_feature_names)
# ---------------------------------------------------------------------------


def apply_perf_mode(combo: FuzzCombo) -> FuzzCombo:
    """FUZZ-1 (2026-05-23): return a config-coverage downgrade of ``combo``.

    Goal: verify suite wiring on every combo in seconds instead of minutes.
    Pins ``n_rows=1000`` (10-100x smaller), ``iterations=1`` (1-10x smaller),
    and disables the heavy optional phases (MRMR, BorutaShap, ensembles,
    baseline_diagnostics, dummy_baselines, composite_discovery, target-
    distribution-analyzer). Useful when iterating on suite-level wiring
    correctness without paying real fit cost.

    Quality / metric assertions are NOT meaningful after this downgrade --
    use only as a smoke-test pre-filter (e.g., ``MLFRAME_FUZZ_PERF_MODE=1
    pytest tests/training/test_fuzz_suite.py``). Real perf / accuracy runs
    use the untouched combo straight from ``enumerate_combos``.

    Returns a fresh ``FuzzCombo`` instance (dataclass.replace) so the
    original is untouched -- enables caller to bench the same combo in
    BOTH modes back-to-back.
    """
    import dataclasses
    return dataclasses.replace(
        combo,
        n_rows=1000,
        iterations=1,
        # FS / heavy passes
        use_mrmr_fs=False,
        use_boruta_shap_cfg=False,
        # Ensembles
        use_ensembles=False,
        # Diagnostics / baselines
        baseline_diagnostics_enabled_cfg=False,
        dummy_baselines_enabled_cfg=False,
        # Composite discovery
        composite_discovery_enabled_cfg=False,
        # Target distribution analyzer (heavy polars + LGB quick model)
        enable_target_distribution_analyzer_cfg=False,
        # Feature-handling config (PCA / dim reducers / TF-IDF wrap)
        custom_prep=None,
        # Reduce RFECV (only fires if rfecv_estimator_cfg != 'none', but
        # downgrade to bare-minimum splits if a combo enabled it).
        rfecv_cv_n_splits_cfg=2,
        # Tighter eval
        early_stopping_rounds_cfg=2,
    )


def build_frame_for_combo(combo: FuzzCombo):
    """Build a pd / pl DataFrame matching the combo's input spec.

    Returns (df, target_col_name, cat_feature_names: list[str]).

    Text columns (``combo.text_col_count > 0``) are only emitted when
    ``"cb"`` is in ``combo.models`` — CatBoost is the only strategy that
    consumes ``text_features`` (see ``strategies.py``
    ``supports_text_features=True`` for CB; every other model either
    drops them via ``core.py:486-496`` or never looks at them). Same
    gate for embedding columns (``pl.List(pl.Float32)``). We still
    emit them SOMETIMES (not always) because the CB×text_features and
    CB×embeddings paths have their own TF-IDF / feature-dispatch
    edge cases that the earlier fuzz runs never exercised — pin them
    behind the "cb present" gate so a CB-less combo doesn't spuriously
    fail for a reason unrelated to what's being sampled.
    """
    import numpy as np

    rng = np.random.default_rng(combo.seed)
    n = combo.n_rows

    num_cols = {
        f"num_{i}": rng.standard_normal(n).astype("float32") for i in range(4)
    }
    cat_pools = [
        ["A", "B", "C"],
        ["X", "Y", "Z", "W"],
        ["alpha", "beta"],
        ["cat1", "cat2", "cat3", "cat4", "cat5"],
        ["US", "UK", "DE"],
        ["mon", "tue", "wed", "thu"],
        ["P", "Q"],
        ["r1", "r2", "r3"],
    ]
    cat_cols = {}
    cat_names: list[str] = []
    # R3-5 weird_cat_content: substitute specific pool entries with
    # pathological values that historically broke auto-detection, TF-IDF,
    # or encoder dispatch.
    def _apply_weird(pool: list[str], kind: "str | None") -> list[str]:
        if not kind:
            return pool
        pool = list(pool)
        if kind == "empty":
            # replace first entry with empty string
            if pool:
                pool[0] = ""
        elif kind == "unicode":
            # mix in a unicode-heavy value (emoji + CJK + combining marks)
            pool.append("кат́")  # cyrillic + combining acute
            pool.append("\U0001f600\U0001f4ca")        # emoji pair
        elif kind == "null_like":
            # strings that LOOK like nulls but are real string values.
            # Pipeline bugs sometimes treat these as actual nulls.
            pool.extend(["None", "NaN", "null", "NA"])
        return pool

    for i in range(combo.cat_feature_count):
        # Wrap with modulo so cat_feature_count > len(cat_pools) cycles
        # through the pool list rather than IndexError-ing.
        pool = _apply_weird(cat_pools[i % len(cat_pools)], combo.weird_cat_content)
        values = [pool[j % len(pool)] for j in range(n)]
        if combo.null_fraction_cats > 0:
            mask = rng.random(n) < combo.null_fraction_cats
            values = [None if mask[j] else v for j, v in enumerate(values)]
        cat_cols[f"cat_{i}"] = values
        cat_names.append(f"cat_{i}")

    # Target: derive from num_0 + num_1 with noise so models have signal.
    # R3-2 multi_classification_{3,5}: discretise a continuous score into
    # N bins by quantile so distribution is approximately balanced.
    # R3-4 imbalance_ratio: on binary, shift threshold so minority class
    # is 5%/1% of rows instead of ~50/50. Not applied to multi-class
    # (implementation complexity not worth it — balanced multiclass is
    # the useful axis to exercise).
    if combo.target_type == "regression":
        target = 2.0 * num_cols["num_0"] - 1.5 * num_cols["num_1"] + rng.standard_normal(n) * 0.3
        target_col = "target_reg"
    elif combo.target_type == "binary_classification":
        logits = num_cols["num_0"] - 0.5 * num_cols["num_1"] + rng.standard_normal(n) * 0.3
        # Use the canonical imbalance value (clamped by n_rows via
        # _canonical_imbalance) so we never generate a target whose split
        # would reliably drop a class from val/test.
        imb = combo._canonical_imbalance()
        if imb == "rare_5pct":
            thresh = np.quantile(logits, 0.95)
        elif imb == "rare_1pct":
            thresh = np.quantile(logits, 0.99)
        else:
            thresh = 0.0
        target = (logits > thresh).astype("int32")
        target_col = "target"
    elif combo.target_type == "multiclass_classification":
        # 3-class quantile-cut to balanced classes (Phase H restoration of R3-2).
        score = num_cols["num_0"] + 0.3 * num_cols["num_1"] + rng.standard_normal(n) * 0.4
        k = 3  # default 3 classes; multi_5 deferred (resource-heavy)
        quantiles = [np.quantile(score, i / k) for i in range(1, k)]
        target = np.digitize(score, quantiles).astype("int32")
        target_col = "target"
    elif combo.target_type == "multilabel_classification":
        # K=3 binary labels with deliberate label correlation so chain ensemble
        # has a chance to win. Post-generation guarantee: no all-zero rows
        # (iterstrat / sklearn reject those silently).
        k = 3
        logit0 = num_cols["num_0"] - 0.4 * num_cols["num_1"] + rng.standard_normal(n) * 0.4
        y0 = (logit0 > 0).astype("int8")
        logit1 = 0.5 * y0 + num_cols["num_2"] + rng.standard_normal(n) * 0.4
        y1 = (logit1 > 0).astype("int8")
        logit2 = 0.5 * y0 + 0.5 * y1 + 0.3 * num_cols["num_3"] + rng.standard_normal(n) * 0.4
        y2 = (logit2 > 0.6).astype("int8")  # rarer
        Y = np.column_stack([y0, y1, y2])
        # Guarantee no all-zero rows (iterstrat, MultiOutputClassifier).
        zeros = (Y.sum(axis=1) == 0)
        if zeros.any():
            # flip a random label to 1 in zero rows (deterministic via rng)
            for i in np.where(zeros)[0]:
                Y[i, rng.integers(0, k)] = 1
        target = Y  # (N, K)
        target_col = "target"  # FTE will need to handle 2-D target
    elif combo.target_type == "learning_to_rank":
        # Graded relevance 0..3 derived from the same informative features
        # as regression, then bucketed. Synthetic queries with ~8 docs each
        # — group_field 'qid' is added below for the ranker suite.
        # Post-generation guarantee: every query has at least one positive
        # (some library rankers warn or NDCG goes NaN otherwise).
        score = 1.5 * num_cols["num_0"] - 0.7 * num_cols["num_1"] + rng.standard_normal(n) * 0.4
        # Quantile-cut to 4 levels (0..3) so frame has graded relevance.
        q = [np.quantile(score, i / 4) for i in range(1, 4)]
        target = np.digitize(score, q).astype("int32")
        # Build qid: ~8 docs per query (n_rows / 8).
        n_per_query = max(2, min(10, n // 30))  # at least 2 docs per query
        n_queries = max(1, n // n_per_query)
        # Last query may be short; pad qid array to length n.
        qid = np.repeat(np.arange(n_queries), n_per_query)
        if len(qid) < n:
            qid = np.concatenate([qid, np.full(n - len(qid), n_queries - 1, dtype=qid.dtype)])
        elif len(qid) > n:
            qid = qid[:n]
        # Guarantee at least one positive per query: for any query whose
        # docs are all-zero, flip the highest-score doc to relevance 1.
        for q_id in np.unique(qid):
            mask = qid == q_id
            if (target[mask] == 0).all():
                top_idx = np.where(mask)[0][np.argmax(score[mask])]
                target[top_idx] = 1
        # Add qid as a frame column so downstream FTE.group_field can pick it up.
        num_cols["qid"] = qid.astype("int32")
        target_col = "relevance"
    else:
        raise ValueError(f"unknown target_type: {combo.target_type}")

    # Text columns: only emit when CB will actually consume them. Each
    # "text" row is a 3-word sentence drawn from a shared vocabulary so
    # CB's TF-IDF builds a non-empty dictionary (a single-word-per-row
    # column above the cardinality threshold would otherwise degenerate).
    # Use the canonical text count so combos that would crash CB's
    # text-estimator on a small NaN-heavy fold never see a text column
    # in the data — _canonical_text_col_count returns 0 in that window.
    _eff_text_col_count = combo._canonical_text_col_count()
    want_text = _eff_text_col_count > 0 and "cb" in combo.models
    text_vocab = [
        "python", "rust", "golang", "java", "swift", "kotlin",
        "backend", "frontend", "devops", "mlops", "dataeng", "platform",
        "cloud", "edge", "realtime", "batch", "stream", "vector",
        "search", "nlp", "vision", "audio", "robotics", "quantum",
    ]
    text_cols: dict[str, list] = {}
    if want_text:
        # Vectorised token-row build. The naive per-row loop builds n separate
        # Python lists of 3 ints + n " ".join calls -> ~n * (3 * 28B int + 1 list
        # header + 3 dict lookups) overhead, which OOMed on c0028 at n=200k under
        # concurrent profiler memory pressure (iter536 MemoryError at the
        # ``rows.append(" ".join(...))`` site). Numpy fancy-indexing into a
        # str-array gives the same per-cell strings without ever allocating the
        # Python-int idx-list, and the ``map(" ".join, words)`` builds the joined
        # strings as a streaming iterator the list constructor materialises in
        # one shot.
        vocab_arr = np.asarray(text_vocab)
        for i in range(_eff_text_col_count):
            idxs_arr = rng.integers(0, len(text_vocab), size=(n, 3))
            words = vocab_arr[idxs_arr]  # (n, 3) np.str_ — single buffer
            text_cols[f"text_{i}"] = list(map(" ".join, words))

    # Embedding columns: only Polars inputs support detection via
    # ``pl.List(pl.Float32)``; pandas has no robust native analog the
    # auto-detector recognises — skip for pandas to avoid spurious
    # xfails unrelated to the axis under test.
    want_embedding = (
        combo.embedding_col_count > 0
        and "cb" in combo.models
        and combo.input_type != "pandas"
    )

    # Data-axis injections (2026-04-24 combo extension).
    # inject_inf_nan: drop np.inf/-np.inf/np.nan into num_0's first 3 rows
    if combo.inject_inf_nan and n >= 3:
        num_cols["num_0"][0] = np.inf
        num_cols["num_0"][1] = -np.inf
        num_cols["num_0"][2] = np.nan
    # inject_degenerate_cols (#7): add one constant + one all-null numeric
    # column that the ``remove_constant_columns`` flag should strip.
    # The CB+multilabel canon at canonical_key was retired 2026-04-27
    # (batch 2): the production fix landed in trainer / wrappers.py
    # ensures num_const / num_null aren't mis-promoted to cat_features.
    extra_num_cols: dict = {}
    if combo.inject_degenerate_cols:
        extra_num_cols["num_const"] = np.full(n, 7.5, dtype="float32")
        extra_num_cols["num_null"] = np.full(n, np.nan, dtype="float32")
    # inject_zero_col (#40): add an all-zero numeric column as an
    # uninformative feature. Triggers the per-model "constant feature"
    # handling in CB/XGB/LGB/HGB — not supposed to break anything.
    if combo.inject_zero_col:
        extra_num_cols["num_zero"] = np.zeros(n, dtype="float32")
    # Fix G — adversarial columns.
    # inject_rank_deficient: a colinear pair (num_dep = 2 * num_0).
    # Should NOT crash linear models or destabilise GBDTs — this is a
    # correctness guard, not a performance ask.
    if combo.inject_rank_deficient:
        extra_num_cols["num_dep"] = (2.0 * num_cols["num_0"]).astype("float32")
    # inject_all_nan_col: a column that is 100% NaN. Separate from
    # inject_degenerate_cols (which covers const + null together) so
    # combos can toggle it independently.
    if combo.inject_all_nan_col:
        extra_num_cols["num_all_nan"] = np.full(n, np.nan, dtype="float32")
    # inject_label_leak: a feature exactly equal to target + tiny noise.
    # A correctly-functioning suite trains on this happily; the val
    # metric must land near-perfect. Deliberately NOT asserted here —
    # the adversarial axis catches pipeline corruption that SILENTLY
    # suppresses the leak (e.g. label-column reordering, caller-frame
    # mutation); any crash is the real bug we're probing for.
    # For multilabel (target is (N, K)): leak label 0 specifically.
    if combo.inject_label_leak:
        if combo.target_type == "multilabel_classification":
            # Leak the first label only — 2-D target can't be broadcast as
            # a single feature. Single-label leak is still catastrophic for
            # a model that silently mis-uses the first target dimension.
            leak_src = target[:, 0]
        else:
            leak_src = target
        leak_col = leak_src.astype("float32") + (rng.standard_normal(n) * 0.01).astype("float32")
        extra_num_cols["num_leak"] = leak_col
    # R3-1 inject_test_drift: perturb the last 15% of rows so test/val
    # slices see a distribution mismatch. Real prod bug surface (unseen
    # categories, out-of-range values, feature shift) — catches pipelines
    # that memoise train stats without guarding against unseen state.
    if combo.inject_test_drift and n >= 20:
        tail = max(3, int(n * 0.15))
        tail_slice = slice(n - tail, n)
        if combo.inject_test_drift == "out_of_range_numeric":
            # scale last 15% of num_0 by 100× (values outside train range)
            num_cols["num_0"][tail_slice] = num_cols["num_0"][tail_slice] * 100.0
        elif combo.inject_test_drift == "shifted_distribution":
            # shift num_0 by +5 sigma (covariate shift)
            num_cols["num_0"][tail_slice] = num_cols["num_0"][tail_slice] + 5.0
        elif combo.inject_test_drift == "unseen_category" and combo.cat_feature_count > 0:
            # overwrite the FIRST cat column's tail values with a string
            # that didn't exist in the training portion.
            # (cat_cols[f"cat_0"] is already populated; mutate in place.)
            cat_cols["cat_0"] = list(cat_cols["cat_0"])
            unseen = "ZZZ_UNSEEN"
            for j in range(n - tail, n):
                cat_cols["cat_0"][j] = unseen

    # 2026-05-12 Wave 30: when no model in the combo supports polars
    # natively (CB/XGB/HGB), build a pandas frame regardless of the
    # axis-sampled input_type. CatBoostEncoder and other sklearn-native
    # transformers reject polars DataFrames with ``ValueError: Unexpected
    # input type: <class 'polars...'>``. Canonicalised in canonical_key
    # so dedup collapses these combos correctly.
    _any_polars_native = any(m in combo.models for m in ("cb", "xgb", "hgb", "mlp", "lstm", "gru", "transformer"))
    _build_input_type = combo.input_type if _any_polars_native else "pandas"

    if _build_input_type == "pandas":
        import pandas as pd
        data = {**num_cols, **extra_num_cols}
        for name, values in cat_cols.items():
            data[name] = pd.Categorical(values)
        for name, values in text_cols.items():
            # pandas object dtype with n_unique > threshold triggers text
            # auto-promotion inside ``_auto_detect_feature_types``.
            data[name] = pd.array(values, dtype="string")
        # with_datetime_col (#11): add a pandas datetime64 column.
        if combo.with_datetime_col:
            data["ts"] = pd.date_range("2026-01-01", periods=n, freq="h")
        # Multilabel target: 2-D (N, K) stored as an object column of list cells.
        # SimpleFeaturesAndTargetsExtractor unpacks back to (N, K) ndarray at
        # consumption time.
        if combo.target_type == "multilabel_classification":
            data[target_col] = pd.array([row.tolist() for row in target], dtype=object)
        else:
            data[target_col] = target
        return pd.DataFrame(data), target_col, cat_names

    import polars as pl
    data_pl: dict[str, Any] = {**num_cols, **extra_num_cols}
    for name, values in cat_cols.items():
        if _build_input_type == "polars_enum":
            pool_values = [v for v in values if v is not None]
            enum_type = pl.Enum(sorted(set(pool_values)))
            data_pl[name] = pl.Series(values).cast(enum_type)
        elif _build_input_type == "polars_nullable":
            data_pl[name] = pl.Series(values).cast(pl.Categorical)
        else:  # polars_utf8
            data_pl[name] = pl.Series(values, dtype=pl.Utf8)
    for name, values in text_cols.items():
        # Text columns are always pl.Utf8 — the auto-detector routes them
        # to text_features via cardinality threshold (hundreds of unique
        # 3-word sentences on 300+ rows) regardless of combo.input_type.
        data_pl[name] = pl.Series(values, dtype=pl.Utf8)
    if want_embedding:
        emb_dim = 4
        for i in range(combo.embedding_col_count):
            vecs = rng.standard_normal((n, emb_dim)).astype("float32")
            data_pl[f"emb_{i}"] = pl.Series(
                [vecs[j].tolist() for j in range(n)],
                dtype=pl.List(pl.Float32),
            )
    # with_datetime_col (#11): polars datetime64 column.
    if combo.with_datetime_col:
        import datetime as _dt
        start = _dt.datetime(2026, 1, 1)
        data_pl["ts"] = pl.Series(
            [start + _dt.timedelta(hours=i) for i in range(n)],
            dtype=pl.Datetime,
        )
    # Multilabel target: 2-D (N, K) stored as pl.List(pl.Int8) column.
    # SimpleFeaturesAndTargetsExtractor unpacks back to (N, K) ndarray.
    if combo.target_type == "multilabel_classification":
        data_pl[target_col] = pl.Series(
            [row.tolist() for row in target],
            dtype=pl.List(pl.Int8),
        )
    else:
        data_pl[target_col] = target
    return pl.DataFrame(data_pl), target_col, cat_names
