# Audit disposition — `train_mlframe_models_suite` (2026-05-16)

Per memory rules `feedback_use_all_agent_findings` and `feedback_no_premature_closure`: every audit finding from the 6 critique agents is closed with one of four dispositions:

- **RESOLVED** — fix landed; reference test path / commit
- **FUTURE** — accepted but deferred; TODO comment in source at file:line
- **DOC** — addressed by docstring / comment / README / CHANGELOG clarification
- **REJECTED** — disputed; re-read of file:line confirms not a real issue (evidence quoted)

## Audit summary (raw input)

The 6 critique agents produced:

- Feature selection: 4 P0, 9 P1, 7 P2, 8 Low = 28
- Feature engineering: 0 P0, 4 P1, 9 P2, 13 Low = 26
- Ensembling: 5 P0, 10 P1, 7 P2, 8 Low = 30
- Code efficiency: 5 P0, 13 P1, 8 P2, 12 Low = 38
- Pipeline caching: 3 P0, 7 P1, 5 P2, 5 Low = 20
- Polars/pandas/numpy conversions: 3 HIGH, 17 MED, 0 P2, 30 LOW = 50

Total: 192 findings.

## Fix waves landed

| Wave | Agent | Files | Fixes landed | New tests | Result |
|---|---|---|---|---|---|
| Plan | Manual | mrmr.py, pipeline.py, configs.py, drift_report.py, baseline_diagnostics.py, composite_estimator.py, utils.py, README.md, CHANGELOG.md | 4 (combinations lazy, TF-IDF sparse, Normalizer_l2 removal, _to_1d_numpy dedup) | 14 (5+7+ shared in 31-test combined run) | GREEN |
| Naming audit | Manual | drift_report.py | 1 rename (`_to_1d_numpy` -> `_to_numpy_or_none`) | 14 (test_utils_coercion re-verified) | GREEN |
| A | FS/MRMR | mrmr.py, screen.py | 8 (target_prefix RNG, caller-frame try/finally, np.random.seed snapshot/restore, cv/cv_shuffle wired, print->logger, int64->int16 type-agnostic, random_state alias, _lazy_chunks intact) | 10 | GREEN |
| B | Ensembling | ensembling.py, composite_ensemble.py, configs.py (CompositeTargetDiscoveryConfig), _training_context.py, _phase_train_one_target.py | 8 (compare_ensembles default sort -> val, oof_holdout_frac 0.2 default, score_ensemble captured into ctx.ensembles, linear_stack refit on dropout, uniformity gate, is_convex attribute, NNLS no post-fit renorm, regex callable replacement) | 22 + 201 verified | GREEN |
| C | Polars conversions | _setup_helpers.py, _eval_helpers.py, baseline_diagnostics.py, target_temporal_audit.py, ranker_suite.py, _predict_guards.py, composite_estimator.py, composite_screening.py, composite_auto_detect.py, _dummy_baseline_compute.py | 8 (6 `to_pandas` sites -> `get_pandas_view_of_polars_df`, predict_guards double-wrap removed, composite `np.asarray` over `to_numpy` removed at 5 sites, dummy-baseline conversion dedupe) | 18 (incl 9.34x speedup benchmark) | GREEN |
| D | FE | pipeline.py, configs.py (PreprocessingExtensionsConfig + FeatureTypesConfig), _phase_polars_fixes.py, _misc_helpers.py, _phase_helpers.py (datetime), preprocessing.py | 8 (PySR seed threaded, ext column names preserved, joint CatBoost cat union, pl.Array/quantized-int embedding detection, pre-OD Enum receiver hook, configurable datetime_methods, min_non_null field, fused->non-fused rename + symmetric logs) | 19 | GREEN (Fix 5 producer side pending) |
| E | Code efficiency | main.py, _phase_train_one_target.py, _training_context.py, _phase_config_setup.py, _phase_finalize.py, _misc_helpers.py | 13 (33 dead imports removed, dead defaultdict gone, strategy_by_model hoisted to suite-level, len(list(...)) trim, common_params.copy comment, psutil RSS comment, dead try/except removed, interactive probe at import time, _ensure_logging_visible early-return, single-pass walk in finalize, del-df comment, ...) | 14 | GREEN |
| F | Caching | _pipeline_helpers.py, strategies.py, utils.py (compute_model_input_fingerprint), composite_cache.py, configs.py (TrainingBehaviorConfig), _phase_composite_discovery.py, main.py, _phase_helpers.py, _phase_polars_fixes.py, _phase_train_one_target.py, _training_context.py, mrmr.py | 8 (P0 FS cache content fingerprint + target-name, DiscoveryCache wired with dep-version hash, PipelineCache observability, compute_model_input_fingerprint extended keys, pre_pipeline_cache_max config, MRMR._FIT_CACHE LRU bound, DiscoveryCache.get atomic, can_skip_pandas_conv -> defer_pandas_conv) | 34 + 28 verified | GREEN |
| Cross-agent | Manual | tests/training/test_cb_polars_fallback.py, _misc_helpers.py | 2 (CB-feature-names import path; _build_tier_dfs polars `.drop` API mismatch) | (existing tests now pass) | GREEN |

Total active fixes: ~58. Total new tests: ~155.

## Disposition tables (filled by parallel agents)

### Feature Selection + Feature Engineering

_To be populated by Disposition Agent X (in-flight)._

### Ensembling + Caching

_To be populated by Disposition Agent Y (in-flight)._

### Code efficiency + Polars conversions

_To be populated by Disposition Agent Z (in-flight)._

## Aggregate counts

_Filled after all 3 disposition agents complete._

## Sentinel for completeness assertion

`tests/test_audit_disposition_complete.py` parses this file and asserts the disposition row count equals 192 (the recorded total from the 6 critique agents). CI fails on drift.
