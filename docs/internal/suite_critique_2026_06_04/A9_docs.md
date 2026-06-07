# A9 — Documentation audit (mlframe), 2026-06-04

READ-ONLY review of doc accuracy/currency vs current code. Verified every key API
reference by grepping the symbol in `src/` and confirming signature/default/import path.

## Verdict summary

- **README.md is the worst offender.** Its headline "Quick examples" section is largely
  **fictional** — the primary entry-point quickstart, the `CompositeTargetEstimator`
  example, the calibration example, and three FS/FE one-liners reference symbols/params
  that **do not exist** or have **wrong names/signatures**. A new user copy-pasting the
  README quickstart hits `TypeError` / `ImportError` / `AttributeError` on the very first
  call. This is the single highest-impact finding (A9-01..A9-07).
- **The `docs/*_guide.md` user guides are mostly excellent and accurate** — `dummy_baselines_guide.md`,
  `baseline_diagnostics_guide.md`, `SELECTION_BIAS.md`, `MULTI_OUTPUT.md`,
  `DEBUGGING_UPSTREAM_ERRORS.md` verified clean (every config field / symbol / module path resolves).
- **Two AP12/AP13 "guide" docs (`calibration_policy.md`, `honest_diagnostics_guide.md`) cite a
  wrong module path and a non-existent function + an entire block of `ReportingConfig` knobs that
  were never wired.** These read as design specs written ahead of implementation (A9-08..A9-11).
- **Several research/roadmap docs present already-shipped work as a pending backlog** (currency rot):
  `MRMR_RESEARCH_2026_05_28.md`, `pysr_fe_upgrade_research.md`, `feature_handling_examples.md`,
  `internal/fuzz_audit.md`, `NUMERICAL_STABILITY_REPORT.md` (A9-12..A9-18).
- **Organization**: `docs/` mixes user guides with internal audit/research artifacts and has **no index**;
  research docs leak absolute temp paths from a different machine (A9-19..A9-22).

Confidence is HIGH on all stale-API findings (each backed by a code grep showing the real symbol).

---

## Findings

### A9-01 — P0 — stale-api — README quickstart for `train_mlframe_models_suite` is fictional
- Doc: `README.md:100-116` (the headline "One-call multi-model training" example)
- Code: `src/mlframe/training/core/_main_train_suite.py:95-141`
- What's wrong: README calls
  `train_mlframe_models_suite(df=df, target="y", models=[...], regression=False, cv_folds=5, early_stopping_rounds=50, use_polars=True)`
  and then uses `result.models["cb"].metrics["holdout_brier"]`, `result.models["lgb"].calibration_plot()`,
  `result.ensemble("stack").predict_proba(X_new)`.
  The real signature is `train_mlframe_models_suite(df, target_name, model_name, features_and_targets_extractor, mlframe_models=None, ...)`.
  There is **no** `target=`, `models=`, `regression=`, `cv_folds=`, `early_stopping_rounds=`, or `use_polars=` top-level kwarg;
  `features_and_targets_extractor` is **required positional**; the function returns a plain `Tuple[Dict, Dict]` of
  `(models_dict, metadata_dict)` — there is no `result` object, no `.models[...].metrics`, no `.calibration_plot()`, no `.ensemble()`.
  The example raises `TypeError` immediately.
- Concrete fix: replace with the correct call shape used in `docs/examples/composite_targets.md:39-45` and
  `docs/dummy_baselines_guide.md:34-42` (which ARE correct): `target_name=`, `model_name=`,
  `features_and_targets_extractor=fte`, `mlframe_models=[...]`, then read `models, metadata = ...`. Drop the fictional `result.*` API.
- Confidence: High

### A9-02 — P0 — stale-api — README `CompositeTargetEstimator` example uses non-existent params
- Doc: `README.md:124-134`
- Code: `src/mlframe/training/_composite_target_estimator.py:55,113-128`
- What's wrong: README does
  `CompositeTargetEstimator(base_estimator="lgb", meta_estimator="ridge", cv=5, target_transform="residual")`.
  The real `__init__` is `(base_estimator=None, transform_name="diff", base_column="", fallback_predict=..., ...)`.
  There is no `meta_estimator`, no `cv`, no `target_transform`; the correct knob is `transform_name`, and `base_estimator`
  takes an estimator object, not the string `"lgb"`. The README prose also mis-describes the class as "Train K base learners...
  meta-learner on out-of-fold residuals" (a stacking ensemble), but the class is a **single inner regressor fit on a transformed
  target with invert-at-predict** — not a K-learner stacker.
- Concrete fix: rewrite to `CompositeTargetEstimator(base_estimator=LGBMRegressor(), transform_name="linear_residual", base_column="...")`
  and correct the prose. `feature_importances_` delegation (line 133) IS real — keep it.
- Confidence: High

### A9-03 — P0 — stale-api — README calibration example: function name, params, and return type all wrong
- Doc: `README.md:141-149`
- Code: `src/mlframe/metrics/_classification_report.py:186-209` (real fn `fast_calibration_report`)
- What's wrong: README imports `from mlframe.metrics import compute_calibration_report` and calls
  `compute_calibration_report(y_true=, y_proba=, n_bins=15, method="quantile")` then reads `report.brier_rel`, `.brier_res`,
  `.brier_unc`, `.ece`. **`compute_calibration_report` does not exist anywhere in `src/`.** The real function is
  `fast_calibration_report(y_true, y_pred, nbins=10, ...)` — param is `y_pred` (not `y_proba`), `nbins` (not `n_bins`),
  there is **no `method=` param**, and it returns a **tuple** (not an object with `.brier_*`/`.ece` attributes).
- Concrete fix: either point the example at `fast_calibration_report` with the correct tuple-unpacking, or document the actual
  public surface in `mlframe.metrics`. Verified `compute_calibration_report` appears only in `README.md` and the generated
  `src/mlframe.egg-info/PKG-INFO` (a copy of the README) — never in code.
- Confidence: High

### A9-04 — P0 — stale-api — README MRMR/RFECV example imports non-existent `mrmr_classif` and `RFECVCustom`
- Doc: `README.md:157-164`
- Code: real classes at `src/mlframe/feature_selection/filters/mrmr.py:142` (`MRMR`) and
  `src/mlframe/feature_selection/wrappers/_rfecv.py` (`RFECV`, re-exported in `wrappers/__init__.py:23,31`)
- What's wrong: README does
  `from mlframe.feature_selection.filters.mrmr import mrmr_classif` and
  `from mlframe.feature_selection.wrappers import RFECVCustom`, then
  `mrmr_classif(X, y, K=30, scheme="fcq")` and `RFECVCustom(estimator=..., step=0.1, cv=5)`.
  **`mrmr_classif` does not exist** anywhere in `src/` (no function by that name). **`RFECVCustom` does not exist** — the
  exported class is `RFECV`. Both imports raise `ImportError`.
- Concrete fix: use the real estimator API: `from mlframe.feature_selection.filters.mrmr import MRMR` /
  `from mlframe.feature_selection.wrappers import RFECV`, and the `.fit(X, y)` estimator pattern. (The README's later FS
  examples at lines 178-238 correctly use `ShapProxiedFS` and `MRMR` — only this block is wrong.)
- Confidence: High

### A9-05 — P0 — stale-api — README financial-FE example calls non-existent `compute_market_features`
- Doc: `README.md:273-283`
- Code: `src/mlframe/feature_engineering/financial.py:11-12` (public API is `create_ohlcv_wholemarket_features`,
  `merge_perticker_and_wholemarket_features`, `add_ohlcv_ratios_rlags`, `add_fast_rolling_stats`, `add_ohlcv_ta_indicators`)
- What's wrong: README does `from mlframe.feature_engineering.financial import compute_market_features` and
  `compute_market_features(df, ohlc_cols=("open","high","low","close"))`. **`compute_market_features` does not exist**
  anywhere in `src/`. The financial module has no such symbol.
- Concrete fix: replace with the real entry point (`create_ohlcv_wholemarket_features(...)`) and its actual signature,
  or drop the example.
- Confidence: High

### A9-06 — P0 — stale-api — README time-series example: wrong `create_aggregated_features` signature
- Doc: `README.md:272-280`
- Code: `src/mlframe/feature_engineering/timeseries.py:193-216`
- What's wrong: README calls
  `create_aggregated_features(df, value_col="price", time_col="ts", windows=[5,15,60], aggs=["mean","std",...])`.
  The real signature is `create_aggregated_features(window_df, row_features, create_features_names, features_names,
  dataset_name, vars_mask_regexp=None, ...)` — there is **no** `value_col`, `time_col`, `windows`, or `aggs` param.
  The example raises `TypeError`.
- Concrete fix: document the actual signature or replace with a runnable example matching it.
- Confidence: High

### A9-07 — P0 — stale-api — README batch-inference example imports non-existent `batch_predict`
- Doc: `README.md:304-313`
- Code: `src/mlframe/inference/predict.py` (public defs: `read_trained_models`, `get_models_raw_predictions`; no `batch_predict`)
- What's wrong: README does `from mlframe.inference.predict import batch_predict` and
  `batch_predict(model=clf, X=large_frame, batch_size=10_000, n_workers="auto")`. **`batch_predict` does not exist** anywhere
  in `src/`. The "Batch inference with adaptive worker count" section describes a function that was never shipped (or removed).
- Concrete fix: either point at the real inference API in `inference/predict.py` / `inference/postanalysis.py`, or remove the section.
- Confidence: High

### A9-08 — P1 — stale-api — `calibration_policy.md` cites wrong module for `pick_best_calibrator`
- Doc: `calibration_policy.md:3,55` (and echoed in `honest_diagnostics_guide.md:94`, `audit_2026_05_24_summary.md:43`)
- Code: `src/mlframe/calibration/policy.py:280` (real home); `src/mlframe/calibration/quality.py` has only
  `make_custom_calibration_plot` (line 89)
- What's wrong: doc says `pick_best_calibrator` lives in `src/mlframe/calibration/quality.py`. It actually lives in
  `calibration/policy.py`. `quality.py` does not define it.
- Concrete fix: change the path to `src/mlframe/calibration/policy.py` in all three docs.
- Confidence: High

### A9-09 — P1 — stale-api — `calibration_policy.md` references non-existent `expected_calibration_error`
- Doc: `calibration_policy.md:13,55`
- Code: real ECE helpers are `_ece_score` (`calibration/policy.py:91`, private) and
  `compute_ece_and_brier_decomposition` (`metrics/_calibration_metrics.py:79`)
- What's wrong: doc presents `expected_calibration_error(y_true, y_pred_proba_C, n_bins=10)` as a callable and lists it as a
  symbol in `quality.py`. **No function named `expected_calibration_error` exists** in `src/`.
- Concrete fix: reference `_ece_score` / `compute_ece_and_brier_decomposition` (and note the private status), or expose a public alias.
- Confidence: High

### A9-10 — P1 — stale-api — `calibration_policy.md` documents `ReportingConfig` knobs that don't exist
- Doc: `calibration_policy.md:28-45` (config block + `force_sigmoid`/`force_isotonic`/`force_nocal` opt-out)
- Code: `src/mlframe/training/_reporting_configs.py` (only `honest_estimator_diagnostics: bool = True` at line 175 is real)
- What's wrong: doc shows `ReportingConfig(pick_calibrator_policy="oof_ece_bootstrap_ci", pick_calibrator_n_bins=10,
  pick_calibrator_n_bootstrap=1000, pick_calibrator_alpha=0.05, render_calibration_plot=True)` and a `pick_calibrator_policy="force_sigmoid"`
  opt-out. **None of these fields exist on `ReportingConfig`.** The doc is a design spec for a configuration surface that was never wired.
  Separately, the real `pick_best_calibrator` (policy.py:280-293) has `n_bins=15` default (doc says 10) and a larger candidate
  palette than the doc's `{NoCal, Sigmoid, Isotonic}` (code also tries betacal, ml_insights/Spline).
- Concrete fix: either implement the `ReportingConfig` knobs or rewrite the doc to describe the real `pick_best_calibrator(...)`
  function signature and remove the fictional config fields. Update n_bins default (15) and candidate list.
- Confidence: High

### A9-11 — P1 — stale-api — `honest_diagnostics_guide.md` documents `ReportingConfig` knobs that don't exist
- Doc: `honest_diagnostics_guide.md:51-62`
- Code: `src/mlframe/training/_reporting_configs.py:175` (only `honest_estimator_diagnostics` exists)
- What's wrong: doc shows `ReportingConfig(honest_estimator_diagnostics=True, honest_gap_warn_threshold=0.10,
  honest_diagnostics_n_bootstrap=1000, honest_diagnostics_alpha=0.05, honest_diagnostics_delong_top_k=2,
  persist_honest_diagnostics_json=True)`. Only `honest_estimator_diagnostics` is a real field; the other five
  (`honest_gap_warn_threshold`, `honest_diagnostics_n_bootstrap`, `honest_diagnostics_alpha`,
  `honest_diagnostics_delong_top_k`, `persist_honest_diagnostics_json`) are **not present** on `ReportingConfig`.
- Concrete fix: prune the non-existent fields, or wire them. The aggregator (`training/honest_diagnostics.py`) and
  `bootstrap_metric`/`delong_test` (`evaluation/bootstrap.py:40,334`) DO exist — only the config knobs are fictional.
- Confidence: High

### A9-12 — P1 — currency — `feature_handling_examples.md` claims FHC is unwired, but it has shipped
- Doc: `feature_handling_examples.md:3-5` ("end-to-end consumer wiring ... lands in a follow-up. Until consumer wiring
  lands, the FHC is built and validated but the existing `train_mlframe_models_suite` ignores it")
- Code: FHC is consumed: `src/mlframe/training/core/_phase_train_one_target_helpers.py:198,216` (`feature_handling_apply`
  runs per-target when `ctx.feature_handling_config` is set); validation at `_phase_config_setup.py:135-148`; suite kwarg
  at `_main_train_suite.py:129,244`.
- What's wrong: the headline disclaimer that the suite "ignores" FHC is stale — the consumer wiring landed. Readers are told
  the recipes are aspirational when they are now live.
- Concrete fix: remove the "ignores it / contract for what's coming" disclaimer; state that FHC is wired via
  `feature_handling_config=`. (All the FHC symbols/presets in the doc — `FeatureHandlingConfig`, `tfidf_only`, `cb_native_only`,
  `embedding_only`, `provider_status`, `TextHandlerSpec`, `ModelHandlingOverride` — verified to exist.)
- Confidence: High

### A9-13 — P1 — currency — `MRMR_RESEARCH_2026_05_28.md` backlog items have since shipped
- Doc: `MRMR_RESEARCH_2026_05_28.md:86-94,337-357` ("Recommended next steps", "Cross-agent prioritised backlog" with P0/P1/P2/P3)
- Code: the top backlog items now exist as sibling modules:
  `filters/_jmim_scorer.py` (+ `redundancy_aggregator='jmim'` param at `mrmr.py:234`), `filters/_chao_shen.py`,
  `filters/_bur_term.py`, `filters/_ksg.py`, `filters/_conditional_permutation.py`, `filters/_pid_decomposition.py`,
  `filters/_relaxmrmr_3d.py`, `filters/_adaptive_nbins.py`, `filters/_stability_cluster.py`.
- What's wrong: the doc presents JMIM (P0), Chao-Shen (P0), KSG (P2), Conditional Permutation Test (P2), PID (P3),
  RelaxMRMR-3D (P3), adaptive nbins (P1), cluster stability (P2) etc. as future work, but they have landed. The "backlog"
  framing is now misleading.
- Concrete fix: mark shipped items DONE (or move the doc to `audit/`/`internal/` as a historical research record). It is a
  research synthesis, not a user guide — see A9-20.
- Confidence: High

### A9-14 — P2 — currency — `pysr_fe_upgrade_research.md` proposes work that has shipped; references moved module
- Doc: `pysr_fe_upgrade_research.md:5-6,75,196-258,271-283` (gap-analysis "ADD/CHANGE" verdicts, "Top-5 features we were not
  using", operator-preset design)
- Code: shipped: `src/mlframe/feature_engineering/pysr_operators.py` defines `safe_log`/`safe_sqrt`/preset system
  (`_operators_for_preset`, `OPERATOR_JULIA_SIGNATURES`); `pysr_operator_preset` wired into
  `training/_preprocessing_configs.py` + `training/_pipeline_extensions.py`. Also `_apply_pysr_fe` /
  `run_pysr_feature_engineering` now live in `training/_pipeline_extensions.py`, not `training/pipeline.py` as the doc's
  intro (line 5-6, 75, 287) states.
- What's wrong: the "we are NOT using X / verdict ADD" framing describes the pre-implementation state; the safe-operator presets
  and several knobs now exist. The `pipeline.py:_apply_pysr_fe` path references are partially stale (code moved to `_pipeline_extensions.py`).
- Concrete fix: annotate shipped items; fix the `pipeline.py` -> `_pipeline_extensions.py` module reference. Treat as historical research (A9-20).
- Confidence: High (symbols verified) / Medium (which exact knobs shipped vs deferred not exhaustively diffed)

### A9-15 — P2 — currency — `NUMERICAL_STABILITY_REPORT.md` file:line references are stale (code moved)
- Doc: `NUMERICAL_STABILITY_REPORT.md:17-37,42-57,168-198,210` (audit table cites `numerical.py:740`, `:147-156`, `:705-741`,
  `:281-291`, `:1065`, `:442-497`, etc.; "bug fix" at `numerical.py:738-749`)
- Code: `compute_moments_slope_mi` and the weighted-moments kernel (incl. the `weighted_skew`/`weighted_kurt` fix) now live in
  `src/mlframe/feature_engineering/_numerical_numba.py:449,596-606` — NOT in `numerical.py` (now 866 lines; the cited 1065-line
  references no longer exist). The `_numerical_stable.py` reference impls (`welford_mean_var_seq`, `welford_moments_seq`,
  `kahan_sum_seq`, `kahan_dot_seq`, `kahan_two_pass_var_seq`) and the bench test (`tests/.../test_numerical_stability_bench.py`)
  all exist and verify.
- What's wrong: every `numerical.py:NNN` line reference in the audit table is stale after the module split; a reader following
  them lands on wrong/absent code. The bug fix DID land (now Kahan-compensated `weighted_skew_c`/`weighted_kurt_c` in `_numerical_numba.py`).
- Concrete fix: repoint file:line refs to `_numerical_numba.py`, or (better) move this dated audit to `audit/`/`internal/` (A9-20).
- Confidence: High

### A9-16 — P2 — consistency — `composite_targets.md` documents PRE-flip composite defaults (knn/MI-only)
- Doc: `examples/composite_targets.md:92-94,127-130` ("Inside the defaults: ... kNN MI, MI-only screening"; "Stick with `"knn"`
  (default) if you have power-law / heavy-tail targets")
- Code: `src/mlframe/training/_composite_target_discovery_config.py:309` (`mi_estimator: str = "bin"`) and `:344`
  (`screening: str = "hybrid"`)
- What's wrong: the doc states the defaults are `mi_estimator="knn"` and `screening="mi"` (MI-only). The actual defaults are
  `"bin"` and `"hybrid"` — the R10b flip documented in `CLAUDE.md`. The Tier-1 "Inside the defaults" callout and the Tier-2
  note both contradict the code (and CLAUDE.md). A reader is told "knn is the default" when it is not.
- Concrete fix: update the "Inside the defaults" and "Stick with knn (default)" lines to reflect `mi_estimator="bin"` /
  `screening="hybrid"` defaults; reframe knn as the explicit opt-in for heavy-tail targets.
- Confidence: High

### A9-17 — P2 — stale-api — `composite_targets.md` benchmark commands use a non-importable module path
- Doc: `examples/composite_targets.md:280-284`
- Code: top-level `benchmarks/` is NOT under `src/mlframe/`; `import mlframe.benchmarks` raises `ModuleNotFoundError` (verified
  with the live interpreter). Files exist at repo-root `benchmarks/composite_target_benchmark.py`, `benchmarks/composite_profile.py`.
- What's wrong: doc says `python -m mlframe.benchmarks.composite_target_benchmark --fast` and
  `python -m mlframe.benchmarks.composite_profile --feature all`. These fail — `mlframe.benchmarks` is not an installed package.
- Concrete fix: change to `python benchmarks/composite_target_benchmark.py --fast` (run from repo root) or package the benchmarks
  under `src/mlframe/`. (`MULTI_TARGET_REGRESSION.md` / other docs avoid this by not using `-m mlframe.benchmarks`.)
- Confidence: High

### A9-18 — P2 — currency — `MULTI_TARGET_REGRESSION.md` mixes "shipped" status with a future roadmap of the same work; cites non-existent `MultiTargetDispatchConfig`
- Doc: `MULTI_TARGET_REGRESSION.md:11-102` (Status: everything ✅ landed) vs `:118-171` ("Suite-integration roadmap ...
  Each step is a separate PR", Phases 1-9), and `:138,156` (`MultiTargetDispatchConfig`)
- Code: MTR core verified shipped: `TargetTypes.MULTI_TARGET_REGRESSION` (`_configs_base.py:126`),
  `is_multi_target_regression`/`is_any_regression` (`_configs_base.py:152,160`), auto-route via
  `multilabel_strategy="multi_target_regression"` (`_composite_target_discovery_config.py:773,940`),
  `MTRPerColumnEqualMeanEnsemble` (`_phase_composite_post_xt_ensemble.py:27`). But **`MultiTargetDispatchConfig` does not exist**
  (the auto-route uses `multilabel_strategy` on the discovery config instead).
- What's wrong: the doc's bottom half reads as a future PR roadmap for work its top half says is already done — a reader cannot
  tell what is real. And `MultiTargetDispatchConfig` (referenced in Phase 3.9 / Phase 6.15) was never created.
- Concrete fix: collapse the redundant "roadmap" into a short "design history" note, or delete it now that MTR landed; remove
  `MultiTargetDispatchConfig` references (replace with `CompositeTargetDiscoveryConfig.multilabel_strategy`).
- Confidence: High

### A9-19 — P2 — consistency — `date_features_kaggle_research.md` states cyclical encoding is OFF by default; code defaults it ON
- Doc: `date_features_kaggle_research.md:102-103` ("The default we ship is OPT-IN (`cyclical=False`)") and `:184`
  ("New `add_cyclical: bool = False` kwarg")
- Code: `src/mlframe/feature_engineering/basic.py:219` (`add_cyclical: bool = True`)
- What's wrong: the real default is `add_cyclical=True`, contradicting the doc's "opt-in / cost is zero for tree-only users" claim.
  The doc body also uses two different kwarg names (`cyclical` in §3, `add_cyclical` in §7); the real one is `add_cyclical`.
  (The `_DEFAULT_DATE_METHODS` field/dtype table in §7 — year int32, quarter int8, month int8, week_of_year int8, day int8,
  day_of_year int16, weekday int8, is_weekend bool — matches `basic.py:89-98` exactly.)
- Concrete fix: update §3 and §7 to `add_cyclical=True` default; remove the `cyclical=False` wording and the single `cyclical=`
  name variant.
- Confidence: High

### A9-20 — P2 — organization — internal audit/research artifacts live in user-facing `docs/`
- Docs: `docs/audit_2026_05_24_summary.md`, `docs/MRMR_RESEARCH_2026_05_28.md`, `docs/pysr_fe_upgrade_research.md`,
  `docs/NUMERICAL_STABILITY_REPORT.md`, `docs/WAVE5_GPU_ROADMAP.md`, `docs/date_features_kaggle_research.md`
- What's wrong: these are dated audit summaries / multi-agent research syntheses / GPU-roadmap disposition records — internal
  artifacts, not user documentation. They sit alongside genuine user guides (`dummy_baselines_guide.md`, etc.) with no
  separation. `docs/internal/` already exists (holds `fuzz_audit.md`) and is the right home; `audit/` holds the rest of the
  audit trail. Mixing them makes `docs/` un-navigable and exposes stale internal planning as if it were product docs.
- Concrete fix: move the six artifacts above to `docs/internal/` (or `audit/`). Keep `docs/` for user-facing guides.
- Confidence: High (judgment call on exact destination; the mis-placement itself is clear)

### A9-21 — P2 — organization — `docs/` has no index / navigation
- Doc: `docs/` root (no `README.md` / `index.md`); only `docs/examples/README.md` exists and indexes just the examples subdir.
- What's wrong: 16 top-level markdown files with no entry point. README.md links a couple of guides inline but there is no
  single map of "which doc covers what". New contributors cannot discover `MULTI_OUTPUT.md`, `SELECTION_BIAS.md`, the guides, etc.
- Concrete fix: add `docs/README.md` indexing the user guides (and, after A9-20, a separate `docs/internal/` index). Model it on
  the existing `docs/examples/README.md` table.
- Confidence: High

### A9-22 — Low — quality — `MRMR_RESEARCH_2026_05_28.md` leaks absolute temp paths from a different machine
- Doc: `MRMR_RESEARCH_2026_05_28.md:6-8`
- What's wrong: embeds `D:\Temp\AppData\Local\Temp\claude\C--Users-TheLocalCommander\...\tasks\*.output` paths (raw agent
  transcript locations) on a host named `TheLocalCommander` — dead, machine-specific, leaks an unrelated username. No reader can
  open them.
- Concrete fix: remove the raw-transcript path block (or relocate the artifacts into the repo if they have lasting value).
- Confidence: High

### A9-23 — Low — consistency — Python-version support claim differs between CONTRIBUTING.md, README.md, and pyproject
- Doc: `CONTRIBUTING.md:22` ("CI verifies 3.9 through 3.13") vs `README.md:59-61` ("tested on 3.9 through 3.14 ... ships cp314 wheels")
- Code: `pyproject.toml:12` `requires-python = ">=3.9"` and classifiers list 3.9–**3.14** (`pyproject.toml:35-40`); interpreter on
  this box is 3.14.3 and `import mlframe` works.
- What's wrong: CONTRIBUTING says the ceiling is 3.13; README and pyproject say 3.14. Minor but a contributor reading CONTRIBUTING
  may assume 3.14 is unsupported.
- Concrete fix: bump CONTRIBUTING.md to "3.9 through 3.14" to match README + pyproject.
- Confidence: High

### A9-24 — Low — currency — `internal/fuzz_audit.md` "planned upgrades" A & D have shipped
- Doc: `internal/fuzz_audit.md:98-148` ("Planned upgrades (A–G)": A `test_fuzz_3way_suite.py`, D `test_fuzz_metamorphic.py`)
- Code: `tests/training/test_fuzz_3way_suite.py` and `tests/training/test_fuzz_metamorphic.py` both exist
  (`test_fuzz_regression_sensors.py` exists too, matching §5).
- What's wrong: A (3-wise) and D (metamorphic dual-runs) are described as planned but the test files now exist. Lower severity
  because this is an internal doc, but the "planned" framing is stale for at least A and D.
- Concrete fix: mark A/D (and any other shipped items) as DONE, keeping the genuinely-deferred ones (F coverage-feedback) as planned.
- Confidence: Medium (file existence verified; did not confirm each file fully implements the doc's spec)

---

## Docs verified ACCURATE (no findings)

- `dummy_baselines_guide.md` — every `DummyBaselinesConfig` field (`enabled`, `apply_to_target_types`, `ts_extra_periods`,
  `per_group_*`, `stratified_n_repeats`, `random_within_query_n_repeats`, `paired_bootstrap_n_resamples`,
  `strongest_min_beat_runner_up_prob`, `bootstrap_ci_threshold`, `bootstrap_ci_n_resamples`, `best_model_min_lift`,
  `random_state`), `compute_dummy_baselines`, `_baseline_inputs_hash`, `train_mlframe_ranker_suite`, and the
  `_profile_dummy_baselines.py` / `_smoke_dummy_baselines_e2e.py` module paths verified. Correct suite signature.
- `baseline_diagnostics_guide.md` — all `BaselineDiagnosticsConfig` fields and `format_baseline_diagnostics_report` verified. Correct suite signature.
- `SELECTION_BIAS.md` — `compute_label_distribution_drift`, `format_drift_report`, `PULearningWrapper` verified (importable from
  `mlframe.training`; constructor params and positional call match).
- `MULTI_OUTPUT.md` — `MultilabelDispatchConfig`, `make_train_test_split`, `canonical_predict_proba_shape`/`predict_from_probs`
  (public aliases of private impls, re-exported), multilabel metrics, `metrics_registry` funcs, `_WelfordAccumulator`,
  `ensemble_probabilistic_predictions` all verified.
- `DEBUGGING_UPSTREAM_ERRORS.md` — the three `profiling/bench_polars_*.py` benches and `_polars_nullable_categorical_cols` verified.
- `examples/README.md` — accurate small index of the examples subdir.
- `CONTRIBUTING.md` — accurate apart from A9-23 (`mlframe.__version__` repro works; markers, line-length 160, sklearn-matrix surface all correct).
- README sections that ARE correct: install/extras (verified against `pyproject.toml` extras), the FS section (`ShapProxiedFS`
  params + `preflight` staticmethod, `MRMR` friend-graph + `fe_auto` + `recommend_enabled_fe`, `ParamOracle`), the post-hoc
  calibration concept, the two-layer caching strategy (`_PRE_PIPELINE_CACHE`, `_pipeline_cache.py`), and Roadmap/Testing/Contributing.
  Only the seven "Quick examples" code blocks A9-01..A9-07 are broken.
