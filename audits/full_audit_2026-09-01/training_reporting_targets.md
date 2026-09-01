# training_reporting_targets
Files reviewed: 23 | LOC: ~9,800

## Summary
The cluster is in good shape on the classic bug classes this audit hunts: no `df.drop(cols)`-without-`columns=`, no pandas-object-in-boolean-context, and most broad `except Exception` sites have already been narrowed and given loud log lines. The remaining findings concentrate in two places. First, **verdict text that contradicts the number printed beside it**: the residual audit prints `residuals look ~Gaussian: |skew|=2.00 (< 0.3)` because the mild-skew branch falls through to the Gaussian verdict without returning, and the distribution analyzer prints `near_constant_target(rel_std=...)` for a zero-mean target where the "rel_std" it prints is actually an *absolute* std, which then early-returns and kills every other detector. Second, **chart/diagnostic accounting that is not honest**: every regression report unconditionally records `regression_panels` as FAILED (because `binary_panels` has a non-empty default, the gate opens for regression, and the dispatcher's legitimate `None` "nothing to render" return is bucketed as a failure with no cause); three default-ON calibration chart families write files that are never registered in `metrics["charts"]`, so they are invisible to the combined-HTML index and to any chart-presence assertion; and `DiagnosticsBudget.report()` -- whose module docstring promises "everything skipped is named in a single line at the end, so a shortened report never looks like a complete one" -- is never called anywhere in the codebase. Contract drift is real but bounded: 13 `FeatureHandlingConfig` knobs are accepted under `extra="forbid"` (and exercised by the fuzz suite) yet read nowhere in `src/`, `per_target` is validated but never applied, and `LeakageSafeEncoder`'s `time_aware`/`cv_splitter` safety valve is unreachable from the config surface. One measured-mechanism perf item: the disk-cache eviction pass runs a full directory `listdir`+`getmtime` after every disk write in the 5-50 GB free-space band and then evicts nothing.

## Findings

### TRAINING_REPORTING_TARGETS-1 [P1] mislabelled-verdict
**File:** src/mlframe/training/reporting/_reporting.py :565-577
**Summary:** Every regression report records `regression_panels` in `metrics["charts"]["failed"]` even though no regression panel grid exists to render, so the chart accounting says FAILED for something that was never attempted and no cause is recorded.
**Failure scenario:** A plain regression target with `ReportingConfig` left at defaults. `binary_panels` defaults to the non-empty `"ROC PR SCORE_DIST KS THRESHOLD GAIN PIT"` (src/mlframe/training/_reporting_configs.py:203), so the guard at _reporting.py:519 (`plot_file and plot_outputs and (binary_panels or ...)`) opens for a regression report. `render_multi_target_panels` returns `None` -- its own docstring (src/mlframe/reporting/auto_dispatch.py:171-178) says `None` means "nothing was rendered (regression, missing inputs, or all panel templates empty)". Line 572 then does `_charts["failed"].append(f"{_which}_panels")` with `_which = "regression"`. `_panel_failures` is empty (nothing raised), so no `panel_exceptions` entry is added either. Result: an operator, or a batch run asserting on chart presence, reads "regression_panels FAILED" on every regression report and goes hunting for a rendering bug that does not exist; conversely a real failure in this slot is indistinguishable from this no-op.
**Suggested fix:** Distinguish "not applicable / nothing matched" from "attempted and failed". Only append to `failed` when `_panel_failures` is non-empty; otherwise record the tag under a `skipped` bucket with the reason (target_type has no panel grid, template empty), or have `render_multi_target_panels` return a sentinel separating "no branch matched" from "branch matched and produced nothing".
**Evidence:** _reporting.py:519 gate; _reporting.py:567 `_which` derivation; _reporting.py:572 unconditional `failed.append`; auto_dispatch.py:168-182 return contract; _reporting_configs.py:203 non-empty `binary_panels` default.

**Disposition:** RESOLVED. A no-op render now records a `skipped` bucket; `failed` is written only when `_panel_failures` is non-empty. `tests/training/test_reporting_verdicts_say_what_they_measured.py`.

### TRAINING_REPORTING_TARGETS-2 [P1] mislabelled-verdict
**File:** src/mlframe/training/targets/regression_residual_audit.py :399-404 (and :388-393)
**Summary:** The residual audit's "Gaussian (well-behaved)" verdict prints `|skew|=<value> (< 0.3)` for skew values that are not below 0.3, because the mild-skew branch appends a rationale but does not return.
**Failure scenario:** Regression residuals with skew=+2.0, excess_kurt=+0.1, |hetero|=0.1, and any negative value in `y_true` or `y_pred` (so the non-negativity gate at line 305 does not fire). The heavy/mild/platykurtic kurtosis branches (325/333/350/362) all miss; line 375 `if abs_skew >= SKEW_MODERATE` appends a rationale and falls through; line 399 emits `residuals look ~Gaussian: |skew|=2.00 (< 0.3), excess kurt=+0.10 (within +/-0.5)` and line 407 returns hypothesis `"Gaussian (well-behaved)"` with suggested loss `"MSE (default) - diagnostics support the standard regression assumption"`. The same contradiction appears on the formal-test path: line 390 emits `moment-based heuristics within Normal band (|skew|=2.00, ...)`. This text is rendered into the default regression log block (`format_residual_audit_report`, called from _reporting_regression/__init__.py:752-756) and into the residual-histogram panel title (`plot_residual_diagnostics`, line 760), so an operator is told a strongly right-skewed residual is textbook Normal and that MSE is appropriate.
**Suggested fix:** Either make the mild-skew branch terminal (return a `"Near-Gaussian (skewed)"` hypothesis when `abs_skew >= SKEW_MODERATE`), or build the final rationale from the measured comparison rather than restating the threshold as if it held. The first option is the honest one: the verdict label itself is wrong, not just the parenthetical.
**Evidence:** Full `_diagnose` control flow (lines 292-407): only lines 310/312/331/340/360/372/398 return; line 375's branch has no return, so any `abs_skew` value reaches line 399 as long as the kurtosis gates miss.

**Disposition:** RESOLVED. The skew branch is terminal, returning `Near-Gaussian (mildly skewed)` or, above `SKEW_HIGH`, `Skewed (strongly asymmetric)` with a robust loss. Note the reachable regime is narrower than the finding describes: at skew +2.0 the formal normality test rejects first and the verdict was already correct. The defect fires where the K-squared test lacks power -- measured at n=200, standardised gamma(60), skew +0.32, which pre-fix returned `Gaussian (well-behaved)` with the self-contradicting `|skew|=0.32 (< 0.3)`. Same test file.

### TRAINING_REPORTING_TARGETS-3 [P1] mislabelled-verdict
**File:** src/mlframe/training/targets/_target_distribution_analyzer_target_fn.py :135-142
**Summary:** For a zero-mean regression target the code substitutes the absolute std for `rel_std`, compares it against a relative threshold (1e-3), prints the absolute value under the name `rel_std`, and early-returns, suppressing every other detector in the analyzer.
**Failure scenario:** `y` = daily returns or centred residuals with mean ~0 and std=8e-4. Line 135: `abs(mu) > 1e-9` is False, so `rel_std = sigma = 8e-4`. Line 138: `8e-4 < 1e-3`, so `pathologies.append("near_constant_target(rel_std=8.00e-04)")` and the function returns at lines 139-144. The target has a perfectly healthy dispersion; it is labelled degenerate, the printed `rel_std=8.00e-04` is not a relative std at all, and heavy-tail / skew / multi-modal / strong-AR / clustered-target detection plus all `knob_overrides` (huber objectives, `use_layernorm=False`, `prefer_group_aware`) are never computed for that target. The reverse also holds: a huge-scale zero-mean target (std=1e6) can never trip the gate even when genuinely constant.
**Suggested fix:** When `|mu| <= 1e-9`, use a scale-free degeneracy test that does not pretend to be a ratio -- e.g. `sigma <= 1e-12 * max(1.0, ptp(y))` or `nunique(y) == 1` -- and record it in `diagnostics` under a distinct key (`abs_std`) so the printed name matches the printed quantity. Do not early-return on the zero-mean fallback path.
**Evidence:** Line 135 `rel_std = abs(sigma) / (abs(mu) + 1e-9) if abs(mu) > 1e-9 else (sigma if sigma > 0 else 0.0)`; line 137 stores it as `diagnostics["rel_std"]`; line 138 compares against `_NEAR_CONSTANT_REL_STD = 1e-3` (_target_distribution_analyzer.py:72); lines 139-144 return before any other detector runs.

**Disposition:** RESOLVED, with one correction to the suggested fix. The ratio is reported only when there is a mean to divide by; at zero mean the spread is recorded as `abs_std` so the name matches the quantity. The suggested `sigma <= 1e-12 * ptp(y)` test does not work: sigma/ptp is about 1/6 for any ordinary distribution and never approaches a small threshold, so it would call nothing degenerate. At zero mean there is no scale to be relative to, so only an exactly-constant target (`ptp == 0` or `sigma == 0`) is degenerate. Same test file.

### TRAINING_REPORTING_TARGETS-4 [P1] wrong-split-label
**File:** src/mlframe/training/reporting/_reporting_diagnostics.py :171-211 (call site :485)
**Summary:** The learning-curve diagnostic refits the model on subsets of the split currently being reported (test / val / oof), then labels the two curves "train score" and "holdout score" -- neither of which is a train-split quantity.
**Failure scenario:** `_render_post_fit_diagnostics` is invoked once per split from `report_model_perf` with `df` = that split's feature frame and `targets` = that split's labels (the same `df` used for `model.predict(df)` at _reporting_probabilistic.py:214 / _reporting_regression/__init__.py:161). `_build_learning_curve(model, df, targets, ...)` forwards them straight into `compute_learning_curve`, which carves its own 20% holdout out of that frame and fits clones on log-spaced prefixes of the remainder. On the `_test` report the panel's "train score" series is a score on a subset of the test rows and "holdout score" is a score on a disjoint subset of the same test split; the panel title (`learning_curve_panel`, diagnostics/learning_curve.py:472) then stamps a `data_starved` / `saturated` verdict derived entirely from test-split rows, and the whole result is stamped into `metrics["learning_curve"]`. Off-default (`LearningCurveConfig.enabled=False`, diagnostics/learning_curve.py:121), which is why this is P1 not P0. A secondary defect on the same path: the scorer is hardcoded to `roc_auc` for anything non-regression (line 191), so a multiclass or multilabel target raises inside `get_scorer`/scoring and is swallowed by the broad `except Exception` at line 209, surfacing only as "learning_curve diagnostic failed".
**Suggested fix:** Thread the train-split frame plus train targets into `_render_post_fit_diagnostics` (they exist in the trainer scope that builds the per-split reports) and run the learning curve once per model against the train pool, not once per reported split. If the intent really is "curve on the reported split", rename the series and verdict to name the split. Also pick the scorer from `target_type` (`roc_auc_ovr` for multiclass, per-label averaging for multilabel) instead of hardcoding binary `roc_auc`.
**Evidence:** _reporting.py:648-653 passes the report's own df/targets; _reporting_diagnostics.py:193-202 passes them unchanged to compute_learning_curve; diagnostics/learning_curve.py:360-371 splits the holdout out of that same array; learning_curve.py:478-479 labels the series train score / holdout score.

**Disposition:** RESOLVED for both halves. The panel's series and subtitle name the split the curve was computed on (`fit-subset score (test)` / `held-out score (test)`) instead of claiming train and holdout, and the scorer is chosen from `target_type` -- `roc_auc_ovr_weighted` for multiclass, `r2` for quantile and ranking -- so a multiclass target no longer raises inside the scorer and vanishes into the broad handler. Threading the train split in instead was considered and not done: it would change `report_model_perf`'s signature and every caller, for a diagnostic that is off by default. Same test file.

### TRAINING_REPORTING_TARGETS-5 [P2] silent-swallow
**File:** src/mlframe/training/reporting/_reporting_probabilistic_calib.py :90, :155, :210
**Summary:** Three default-ON chart families -- fairness-calibration, calibration-by-feature, 2-D calibration heatmap -- write files via `render_and_save` but never register them in `metrics["charts"]["saved"]` or `["paths"]`, so they are omitted from the combined HTML index and from every chart-presence check; their failures are swallowed at DEBUG with no `failed` entry either.
**Failure scenario:** A binary-classification run with `plot_file` set and `plot_outputs` at its default. `fairness_calibration_charts`, `calibration_by_feature_charts` and `calibration_heatmap_2d_charts` all default to True at _reporting.py:453-455. Each helper calls `render_and_save(spec, ..., base_path)` and returns. `_render_post_fit_diagnostics` later builds the combined HTML from `metrics.get("charts", {}).get("paths", [])` at _reporting_diagnostics.py:502-507 -- the fairness / by-feature / 2-D charts are on disk but absent from that list, so the single-page index handed to a reviewer silently omits them. Separately, if `compose_fairness_calibration_figure` raises for one group feature, lines 91-92 log at DEBUG and continue: the chart lands in neither `saved` nor `failed`, so it reads as "never requested".
**Suggested fix:** Give the three helpers the same `metrics.setdefault("charts", ...)` bookkeeping the training-curve and learning-curve paths already use at _reporting_diagnostics.py:124-135 and :493-496 -- append tag plus `base_path` on success and the tag to `failed` on exception -- and raise the exception log level from `logger.debug` to `logger.warning`.
**Evidence:** _reporting_probabilistic_calib.py:90, :155 and :210 call `render_and_save` with no `metrics["charts"]` update; the only `metrics` writes in the file are the disparity / heterogeneity dicts at lines 94-95, 159-160 and 197-202; _reporting_diagnostics.py:501-507 sources the combined HTML exclusively from `charts["paths"]`.

### TRAINING_REPORTING_TARGETS-6 [P2] contract-drift
**File:** src/mlframe/training/reporting/_diagnostics_budget.py :87-103
**Summary:** `DiagnosticsBudget.report()` is never called anywhere in the codebase, so the promise that a budget-shortened diagnostics block names what it dropped is not kept -- a truncated report is indistinguishable from a complete one.
**Failure scenario:** `ReportingConfig.diagnostics_max_seconds` set to e.g. 120 on a wide run. `_render_post_fit_diagnostics` constructs the budget at _reporting_diagnostics.py:275 and routes about 14 diagnostics through `_budget.run`. Once the budget is spent, `run` appends to `self.skipped` and returns None silently at lines 82-84; likewise `self.out_of_scope` under the `heavy_diagnostics_for="best"` policy. `_render_post_fit_diagnostics` returns at line 507 without ever calling `_budget.report()`, so neither the "this report is INCOMPLETE" warning nor the out-of-scope INFO line is ever emitted. The module docstring at lines 8-10 states the opposite: everything skipped is named in a single line at the end, so a shortened report never looks like a complete one.
**Suggested fix:** Call `_budget.report()` at the end of `_render_post_fit_diagnostics`, before the combined-HTML block, and stamp `budget.skipped` and `budget.out_of_scope` into `metrics["charts"]` so a batch run can count truncated reports programmatically.
**Evidence:** A repo-wide grep for `.report()` in src/mlframe returns only evaluation/adversarial_validator.py, its bench, and composite/discovery -- no call site for `DiagnosticsBudget.report`. _reporting_diagnostics.py:275 constructs the budget; lines 328-499 use `_budget.run`; lines 500-507 end the function.

### TRAINING_REPORTING_TARGETS-7 [P2] dropped-value
**File:** src/mlframe/training/targets/_target_temporal_plot.py :60-66
**Summary:** The DSL render path returns before setting `result.plot_path`, so under the default `plot_outputs` the temporal-audit metadata always reports a null plot path even though the chart was written.
**Failure scenario:** `behavior_config.target_temporal_audit_column` set and `reporting_config.plot_outputs` non-empty, which is the default suite configuration. training/core/_phase_train_one_target_model_setup.py:349-354 takes the `plot_outputs` branch of `_plot_target_over_time`; that branch renders and returns None at line 66 without touching `result.plot_path`. Line 359 then stores `_audit.to_dict()` into `metadata["target_temporal_audit"]`, and `TemporalAuditResult.to_dict` at target_temporal_audit.py:154 serialises a null `plot_path`. Only the legacy matplotlib-PNG branch at line 129 sets it. A downstream consumer reading that metadata concludes no temporal-audit chart exists.
**Suggested fix:** Set `result.plot_path` to `base_path`, or to the resolved path returned by `render_and_save`, in the DSL branch before returning, mirroring line 129.
**Evidence:** _target_temporal_plot.py:60-66 versus :126-130; the dataclass field comment at target_temporal_audit.py:131 -- filled in if plot_target_over_time saves a file.

### TRAINING_REPORTING_TARGETS-8 [P2] config-knob-never-read
**File:** src/mlframe/training/feature_handling/config.py :80, :107, :110-112, :118, :143-144, :151, :161, :163-166, :262, :276-277
**Summary:** Sixteen `FeatureHandlingConfig` fields are accepted -- the models are `extra="forbid"`, so setting them looks like a supported API -- but are read nowhere in src/mlframe; a user tuning them gets silence.
**Failure scenario:** A user writes `FeatureHandlingConfig(memory=MemoryConfig(pressure_watermark_pct=60), cache=CacheConfig(prefetch_enabled=False, prefetch_device="cpu", max_per_column_entries=50), pricing=PricingConfig(cap_usd=5.0), repro=ReproConfig(deterministic_torch=True))` expecting a memory-pressure guard, prefetch disabled, a per-column cache cap, a spend gate on paid embedding providers, and deterministic torch. None of these values is ever read; the run behaves exactly as if they were left at defaults, with no warning. A per-field grep over src/mlframe excluding _benchmarks returns exactly one hit -- the definition -- for each of: `pressure_watermark_pct` :80, `eviction_async` :107, `prefetch_enabled` :110, `prefetch_device` :111, `prefetch_vram_safety_factor` :112, `max_per_column_entries` :118, `cap_usd` :143, `warn_above_usd` :144, `redact_column_names` :151, `deterministic_torch` :161, `langdetect_seed` :163, `pinned_svd_solver_params` :164, `forbid_nonatomic_fs` :165, `deterministic_eviction` :166, `auto_locale_sample_size` :276, `auto_locale_english_threshold` :277. Several are even swept by the fuzz suite at tests/training/_fuzz_combo/axes.py:836-842, which therefore proves nothing about them. Separately `per_target` :262 is validated for cache-identity consistency by `_validate_per_target_consistency` but is never consulted when resolving handler chains: `feature_handling_apply` reads only `_effective_text_specs` and `_effective_cat_specs`, which look at `per_model` alone, so a per-target override silently has no effect.
**Suggested fix:** For each field, either wire it into its consumer or delete it. Where wiring is a larger job, add a `model_validator` that emits a `logger.warning` naming the field when a non-default value is supplied -- the pattern apply.py:315-329 already uses for the unimplemented `group_columns` -- so "set it and get nothing" is at least visible.
**Evidence:** Per-field grep over src/mlframe yielding a single definition-site hit each; apply.py:190-196 resolving specs from `_effective_*_specs` only; the `# reserved` comment on config.py:262.

### TRAINING_REPORTING_TARGETS-9 [P2] perf-wasted-scan
**File:** src/mlframe/training/feature_handling/cache.py :388-427 (invoked at :386)
**Summary:** The disk-cache eviction pass fires after every disk write when free space is below `disk_evict_when_free_below_gb` (default 50 GB) but only evicts down to `disk_min_free_gb` (default 5 GB), so between 5 and 50 GB free it performs a full directory listdir plus a per-file getmtime and then evicts nothing.
**Failure scenario:** `cache.persistence="read_write"` on a box with 30 GB free -- the common case on a dev machine or a container volume. Every `_write_disk` call reaches `_maybe_evict_disk`; the early return at line 401 does not fire because 30e9 is below the 50e9 trigger; line 405 builds `entries` with an `os.listdir` plus one `os.path.getmtime` stat per .bin file; line 409 sorts them; then the very first loop iteration hits the `free_bytes >= target_free_bytes` break at line 411, since 30e9 >= 5e9, and returns having deleted nothing. With a mature cache directory of N entries this is one listdir plus N stats plus an N-element sort per cached artefact write, repeated for every column x handler x model in the suite, and it never reclaims a byte until the disk is down to 5 GB.
**Suggested fix:** Move the `free_bytes >= target_free_bytes` short-circuit above the directory scan -- it is a pure function of the already-probed `free_bytes` -- and reconcile the two thresholds; evicting down to `disk_evict_when_free_below_gb` with `disk_min_free_gb` as the hard floor is almost certainly the intended semantics given the trigger value.
**Evidence:** cache.py:396-401 trigger on `disk_evict_when_free_below_gb`; :403 target from `disk_min_free_gb`; :405-409 unconditional scan and sort; :411-412 loop-entry break; defaults 50.0 and 5.0 at config.py:104-105.

### TRAINING_REPORTING_TARGETS-10 [P3] mislabelled-count
**File:** src/mlframe/training/reporting/_reporting_regression/_mtr.py :107-111
**Summary:** The MTR per-target chart summary reports K charts rendered and a `_target0 ... _target{K-1}` range even when the loop skipped columns via `continue`.
**Failure scenario:** A 4-target MTR report where target 2 has fewer than 5 finite true/pred pairs. Lines 82-90 log a throttled WARNING and `continue`, so `{base}_target2.*` is never written; line 107 nonetheless logs "MTR per-target charts: rendered 4 chart base paths at {base}_target0 ... {base}_target3". An operator, or a script globbing that range, looks for a file that does not exist.
**Suggested fix:** Count actual renders in a local counter incremented after `render_and_save` and report that, plus the list of skipped column indices.
**Evidence:** _mtr.py:78-106 loop with the `continue` at :90; :107-111 log using `_K` and `_K - 1`.

### TRAINING_REPORTING_TARGETS-11 [P3] mislabelled-verdict
**File:** src/mlframe/training/targets/_target_distribution_analyzer_target_fn.py :239
**Summary:** The strong-AR pathology string labels the value `lag1_corr=` while the value printed is the maximum absolute autocorrelation over lags 1/2/3/5 -- contradicted by the `source=global_lag3` token printed immediately beside it.
**Failure scenario:** A long-memory target with lag-1 autocorr 0.2 and lag-3 autocorr 0.85. Line 209 sets `ar, ar_lag = _max_abs_lag_autocorr(y_for_stats)`, giving 0.85 and 3; line 210 sets `ar_source = "global_lag3"`. Line 239 then emits `strong_AR_target(lag1_corr=0.850, source=global_lag3)`. The true lag-1 value 0.2 is available in `diagnostics["lag1_autocorr"]` at line 212 but is not the number shown, so anyone reading the pathology list concludes the series has a 0.85 lag-1 autocorrelation.
**Suggested fix:** Rename the token to match the quantity, e.g. `max_abs_autocorr={ar:.3f}`.
**Evidence:** `_max_abs_lag_autocorr` returns the strongest lag per the comment at lines 202-206; line 210 builds the `global_lag{ar_lag}` source tag; line 239 formats the same value as `lag1_corr`.

### TRAINING_REPORTING_TARGETS-12 [P3] documented-param-never-read
**File:** src/mlframe/training/honest_diagnostics.py :83-84 (call site :485)
**Summary:** `_bootstrap_block`'s `preds` parameter is accepted and passed by the caller but never read in the body.
**Failure scenario:** `run_honest_diagnostics` passes `getattr(entry, "test_preds", None)` at line 485. The body of `_bootstrap_block`, lines 86-278, references only `y_true`, `probs` / `p_pos` and `rng_seed`; `preds` is dead. A reviewer adding a crisp-prediction metric such as an accuracy or F1 CI will reasonably assume the plumbing already works.
**Suggested fix:** Delete the parameter and the call-site argument, or use it -- crisp-metric bootstrap CIs are the obvious intent given the docstring's "top-line metrics".
**Evidence:** Full read of `_bootstrap_block` lines 83-278: no occurrence of `preds` after the signature.

### TRAINING_REPORTING_TARGETS-13 [P3] unreachable-safety-knob
**File:** src/mlframe/training/feature_handling/target_encoders.py :144-145 and :506-511 (blocked at handlers.py :105-119 and apply.py :485-493)
**Summary:** `LeakageSafeEncoder`'s `time_aware` and `cv_splitter` -- documented as the safe path for genuinely temporal targets -- cannot be reached from the config surface, so every configured target encoder gets shuffled K-fold OOF regardless of temporal structure.
**Failure scenario:** A user configures `CatHandlerSpec(method="target_mean", params=TargetEncodeParams(...))` on a time-ordered dataset. `TargetEncodeParams` at handlers.py:105-119 has no `time_aware` or `cv_splitter` field and is `extra="forbid"`, so the knobs cannot even be expressed; `_apply_target_encoder._fit` at apply.py:485-493 constructs `LeakageSafeEncoder` with only method / smoothing / woe_smoothing / cv / prior / random_state. The encoder therefore falls to `KFold(shuffle=True)` at target_encoders.py:511, computing each row's encoding from folds that include future rows -- exactly the pattern the `time_aware=False` default comment at lines 173-175 calls legacy. A grep for `LeakageSafeEncoder` over src/mlframe shows apply.py is the only production construction site.
**Suggested fix:** Add `time_aware: bool = False` and optionally a splitter selector to `TargetEncodeParams`, and thread both through `_apply_target_encoder._fit`; the encoder side already supports them.
**Evidence:** target_encoders.py:173-177 field docs; :506-511 splitter selection; handlers.py:111-119 `TargetEncodeParams` field list under `extra="forbid"`; apply.py:485-493 construction; grep showing no other `LeakageSafeEncoder(` call site in src.

### TRAINING_REPORTING_TARGETS-14 [P3] contract-drift
**File:** src/mlframe/training/reporting/_diagnostics_budget.py :40-43
**Summary:** The comment claims an empty-string `mode` is surfaced rather than silently coerced, but an empty string silently behaves exactly like "best".
**Failure scenario:** `ReportingConfig.heavy_diagnostics_for=""`, or any typo such as `"ALL "`. The assignment `self.mode = ("best" if mode is None else str(mode)).lower()` yields the empty string; `allows()` then returns `self.is_primary` for every heavy diagnostic, i.e. the restrictive behaviour, with no warning. The comment directly above says a caller passing an empty string is asking for something this cannot honour and that silently turning it into "best" hides the mistake rather than surfacing it -- nothing is surfaced.
**Suggested fix:** Validate `mode` against the set best/all in `__init__` and emit a `logger.warning`, or raise, on anything else.
**Evidence:** _diagnostics_budget.py:40-43 comment versus assignment; :45-49 `allows` treating any non-"all" mode as restrictive.

### TRAINING_REPORTING_TARGETS-15 [P3] stale-doc
**File:** src/mlframe/training/targets/_target_distribution_analyzer.py :168-171
**Summary:** The feature-side detector doc block still describes the NaN-heavy rule as "fraction > 50%", but the constant it documents was deliberately raised to 0.99.
**Failure scenario:** A reader consults the module doc block to understand why a 60%-NaN column was not flagged, concludes the detector is broken, and files a bug. The actual threshold is `_NAN_FRACTION_THRESHOLD = 0.99` at line 91, changed with a long rationale comment at lines 84-90 that the doc block 80 lines below was never updated to match.
**Suggested fix:** Update the doc block to say fraction >= 0.99 and cross-reference the structural-missingness rationale already written at lines 84-90.
**Evidence:** _target_distribution_analyzer.py:91 versus :168-171, which still reads "NaN-heavy features (fraction > 50%) ... At >=50% the imputer is dominating the column".

## Coverage
Read in full:
- src/mlframe/training/reporting/_reporting.py (675)
- src/mlframe/training/reporting/_reporting_diagnostics.py (507)
- src/mlframe/training/reporting/_reporting_probabilistic.py (914)
- src/mlframe/training/reporting/_reporting_probabilistic_calib.py (212)
- src/mlframe/training/reporting/_diagnostics_budget.py (106)
- src/mlframe/training/reporting/_reporting_regression/__init__.py (778)
- src/mlframe/training/reporting/_reporting_regression/_mtr.py (128)
- src/mlframe/training/reporting/_reporting_regression/_sensors.py (177)
- src/mlframe/training/targets/regression_residual_audit.py (776)
- src/mlframe/training/targets/target_temporal_audit.py (577)
- src/mlframe/training/targets/_target_temporal_audit_from_agg.py (162)
- src/mlframe/training/targets/_target_temporal_plot.py (134)
- src/mlframe/training/targets/_target_distribution_analyzer.py (201)
- src/mlframe/training/targets/_target_distribution_analyzer_target_fn.py (325)
- src/mlframe/training/targets/_train_eval_select_target.py (304)
- src/mlframe/training/targets/_ttr_eval_set_scaling.py (205)
- src/mlframe/training/feature_handling/config.py (567)
- src/mlframe/training/feature_handling/apply.py (665)
- src/mlframe/training/feature_handling/target_encoders.py (674)
- src/mlframe/training/diagnostics/learning_curve.py (496)
- src/mlframe/training/honest_diagnostics.py (527)

Read in part, targeted at the audited bug classes:
- src/mlframe/training/feature_handling/handlers.py (spec / params classes)
- src/mlframe/training/feature_handling/cache.py (in-memory eviction, disk eviction)

Read for call-site context, outside the cluster:
- src/mlframe/training/core/_phase_train_one_target_model_setup.py (temporal-audit render block)
- src/mlframe/training/core/_phase_temporal_audit.py (batch audit wiring)
- src/mlframe/reporting/auto_dispatch.py (render_multi_target_panels return contract)
- src/mlframe/training/_reporting_configs.py (panel-template defaults)

Grep sweeps across the whole cluster -- reporting/, targets/, feature_handling/, diagnostics/, honest_diagnostics.py -- for: `.drop(` without `columns=`; `... or []` boolean-context on pandas objects; `except Exception` returning False or None; `@njit` / `prange` / `cuda.jit` / `cupy` / `kernel_tuning_cache` before writing any perf finding; and a per-field grep for every FeatureHandlingConfig knob.
