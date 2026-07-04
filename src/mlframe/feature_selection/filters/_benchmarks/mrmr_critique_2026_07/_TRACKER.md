# MRMR multi-agent critique — master disposition tracker (2026-07)

## PROGRESS LOG (live)
- DONE (fixed + tested + pushed): FE-F1 (d804cdf4), EX-1/S-F5/ST-3/FE-F7/EN-2 (33343a58), FE-F2/FE-F3/FE-F4 (bf7023e3), S-F1 (84a0f6a8), FE-F5 (34e831f5).
- REVERTED -> FUTURE (override a deliberate, bench/test-pinned design; need a dedicated multi-seed biz-value bench to settle, NOT a code change on a critique argument):
  - N-F3 (perm_pvalue full-budget extrapolation): pinned by bench_perm_pvalue_addone.py + test_fleuret_perm_pvalue_addone::test_stopped_p_uses_full_budget_denominator (break-position independence is deliberate). Concrete path: add a bench measuring surfaced-confidence calibration on failure-pileup vs significance stops across seeds; only change if it improves calibration without worsening selection.
  - S-F3 (greedy-jmim `**(nexisting+1)` exponent): deliberately placed (evaluation.py:384-398). Concrete path: jmim greedy-path biz-value bench (exponent on/off, multi-seed) on a synergy fixture; remove only if selection equal-or-better.
- NOT OURS (parallel session): the JMIM bit-equivalence + secondary-signal failures in test_layer86.py trace to _orthogonal_jmim_fe -> _jmim_scorer/_mi_greedy_cmi_fe, whose recent commits are the parallel session's GPU-resident-FE binning rewrites (51f7ad5c/7d783ac1/...). Not in our edited path; left for that session to reconcile.
- REMAINING (being worked): N-F1, N-F2, S-F2, S-F4, ST-1, ST-4, P-1..P-4/P-11, and the Low/doc items. Statistical P1s (N-F1/N-F2/S-F2) touch the tuned selection core and are handled with the same discipline: validate with a bench or mark FUTURE with a concrete path, never a blind selection change.



Seven read-only critique agents reviewed the MRMR modules (core selection, numerical/statistical machinery, FE
families + leak-safety, performance, stopping/fallback, encoding recipes, extra FE families). Every finding is
listed here with its disposition and live status. Per-agent full reports are the sibling `mrmr_crit_*.md` files.

Status legend: DONE (fixed+tested+pushed) | WIP | TODO | DOC | FUTURE | REJECTED.

## Core selection (mrmr_crit_selection.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| S-F1 | P1 | GPU relevance null uses 2 perms vs CPU 32 → p-value floor 0.333 ≥ alpha always → subtracts null_mean from every feature incl. strong signal; hardware-dependent selection | gpu.py:974 / evaluation.py:530 | FIX (bump GPU null budget to max(nperm,32)) | DONE (84a0f6a8) |
| S-F2 | P1(jmim) | JMIM candidates scored by joint-MI but confirmed against the CMIM null (statistic mismatch). | fleuret.py confidence chain | DONE (sf2) — use_jmim threaded through get_fleuret_criteria_confidence_parallel -> parallel_fleuret -> get_fleuret_criteria_confidence -> evaluate_gain (mirrors use_su/use_mm), so JMIM picks are confirmed against the JMIM statistic. Default (jmim off) bit-identical: the confidence cache is only touched when not confidence_mode, so the fresh-MI jmim branch is safe. jmim bit-equiv tests green + sensor tests. |
| S-F3 | P2(jmim) | Fleuret `**(nexisting+1)` exponent wrongly amplifies JMIM joint-MI>1 | evaluation.py:363 | FIX | FUTURE-bench (reverted) — deliberately-placed exponent; jmim greedy biz-value bench (on/off) before removal |
| S-F4 | P2(off-dflt) | positive_mode/extra_knowledge breaks partial_gains monotonicity | evaluation.py:411,618 | FIX | DONE (sf4, default bit-identical; resume disabled on the non-monotone extra-knowledge path) |
| S-F5 | Low | redundancy_aggregator typos silently degrade to Fleuret | _mrmr_class.py:~3355 | FIX (validate) | DONE (33343a58) |

## Numerical / statistical (mrmr_crit_numstat.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| N-F1 | P1 | Observed relevance uses Miller-Madow but permutation null uses plug-in → over-reject + double bias subtraction | permutation.py:625 vs 341/229/165 | FIX (thread use_mm into null kernels) | DONE (nf1, no-op MM-off; opt-in MM null now matches observed) |
| N-F2 | P1 | Relevance debiased but Fleuret redundancy raw plug-in -> objective biased vs high-cardinality candidates. | evaluation.py ; _entropy_kernels.py conditional_mi | PART-1 DONE (nf2a) — conditional_mi(use_mm) subtracts the analytic MM CMI bias; threaded through evaluate_gain + the fleuret confidence chain + find_best_partial_gain so the redundancy carries the SAME MM correction as the relevance. Default MM-off bit-identical (275 tests green; 2 [mi_greedy] failures are ND-2, unrelated). PART-2 (default-path null-mean redundancy debias, flag-gated + wide biz-value bench) remains FUTURE. |
| N-F3 | P2 | `_perm_pvalue(full_budget=)` overstates confidence on pile-up early breaks | permutation.py:62 | FIX | FUTURE-bench (reverted) — full-budget extrapolation is deliberate + bench-pinned; calibration bench before changing |
| N-F4 | P2 | fixed alpha=0.05, no multiple-testing correction across candidates | evaluation.py:549 | FIX (BH/BY) or DOC | DOC — alpha=0.05 stability across 0.02-0.10 is already documented at evaluation.py:48-61 (selection stable because real signal clears p~0, noise sits near p~1). Pool-wide BH/BY multiple-testing correction is a selection-changing FUTURE needing its own bench; per-candidate alpha kept. |
| N-F5 | P2 | 32-perm null-mean variance can flip near-tied selection | permutation.py:46 | DOC/FUTURE (shrinkage) | DONE-doc |
| N-F6 | P3 | Chao-Shen not matched by its null (≡ N-F1 generalized) | — | FIX-with-N-F1 | FUTURE (with N-F1) |
| N-F7 | P3 | analytic chi-square null assumes fixed occupancy on tied equi-freq bins | _analytic_mi_null.py | DOC | DONE-doc |
| N-F8 | P3 | analytic_batch_noise_gate no per-column perm fallback | _analytic_mi_null.py:253 | DOC/FIX | DONE-doc |

## FE families + leak-safety — temporal (mrmr_crit_fe_temporal.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| FE-F1 | P1 | temporal expanding/lag replay seeds entire train history ignoring timestamps → look-ahead leak/skew | _temporal_agg_fe.py:634,768 | FIX (per-row time merge) | DONE (d804cdf4) |
| FE-F2 | P2 | temporal entity key raw `.astype(str)` not canonical token → int/float drift routes all test rows to global prior | _temporal_agg_fe.py:92 | FIX (canonical_group_token) | DONE (bf7023e3) |
| FE-F3 | P2 | target_aware_group_bin small-group OOF fallback uses all-rows edges → y-leak into MI gate | _grouped_quantile_fe.py:413,442 | FIX (per-fold train-only edges) | DONE (bf7023e3) |
| FE-F4 | P2 | `_compute_target_encoding` naive path (n_oof_folds<=0) emits row's own y | _cat_target_encoding_and_weighted.py:81 | FIX (guard K>=2) | DONE (bf7023e3) |
| FE-F5 | Low | rolling replay casts ns timestamps to float64 (precision loss at window boundary) | _temporal_agg_fe.py:731 | FIX (keep int64) — expanding/lag in d804cdf4, rolling in 34e831f5 | DONE (34e831f5) |
| FE-F6 | Low | grouped_quantile pct_rank self-inclusion fit vs replay (≈1/m offset) | _grouped_quantile_fe.py:206,309 | DOC | DONE-doc (batch-lows) |
| FE-F7 | Low | temporal winner name collision → duplicate columns | _fe_stage_temporal_agg.py:101 | FIX (dedup on concat) | DONE (33343a58) |

## FE families — encoding/grouped recipes (mrmr_crit_fe_encoding.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| EN-1 | P2 | asymmetric key coercion in cross/TE builders (safe today via pre-canonicalization; fragile) | _encoding_recipes.py:110,331,375 | FIX (canonicalize keys in builders) | DONE (en1) |
| EN-2 | Low | polars branch crashes when pandas absent | _encoding_recipes.py:73 etc | FIX (guard pd) | DONE (33343a58) |
| EN-3 | Low | _apply_mi_greedy_transform forces float64 on source cols | _missingness_ratio_recipes.py:37 | FIX (validate numeric at build) | DONE (batch-lows) |
| EN-4 | Low | integer-factorize numeric source truncates floats at replay | _recipe_extract.py:193,198 | FIX (round-to-nearest) | DONE (batch-lows) |
| EN-5 | Low | target_aware_group_bin global-fallback OOF optimism (== FE-F3) | _grouped_quantile_fe.py:413 | dedup of FE-F3 | see FE-F3 |

## FE families — extra families (mrmr_crit_fe_extra.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| EX-1 | P2 | `_is_argmax_eligible` finiteness guard is a tautology → NaN columns not excluded | _conditional_gate_fe.py:357 | FIX (isfinite(a).all()) | DONE (33343a58) |
| EX-2 | Low | argmax/gate meaning shifts on serve-only NaN (no replay NaN policy) | _conditional_gate_fe.py:147 | FIX/DOC | DONE (batch-lows2) |
| EX-3 | Low/P2 | conditional-gate tau optimized on same in-sample y (selection optimism) | _conditional_gate_fe.py:718 | DOC | DONE-doc (batch-lows2) |
| EX-4 | Low/P2 | RankGauss stores full sorted non-unique array (memory) + docstring wrong ("unique") | _extra_fe_families.py:795 | FIX (unique+counts / doc) | DONE-doc (batch-lows) |

## Performance (mrmr_crit_perf.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| P-1 | P1 | whole-matrix `factors_data.copy()` every screen call (100GB-frame rule) | _screen_predictors.py:530 | FIX (mutate-and-restore / gate) | FUTURE — mutate-and-restore rewrite of the tuned confirm/permutation shuffle (extra_x_shuffling defaults ON so gating doesn't help the default); needs selection-equivalence + RAM bench; overlaps the parallel session's active screen-perf work. Site note added. |
| P-2 | P2 | forced float64 copy of FE candidate matrix even when already f64/contig | _fe_cpu_batch.py:64,80,75 | FIX (branch on dtype/flags) | FUTURE — branch the f64 copy on dtype/flags in _fe_cpu_batch (GPU/batch area actively edited by the parallel GPU-resident-FE session; coordinate to avoid conflict). |
| P-3 | P2 | redundant target re-discretization across ~15 FE blocks | _fit_impl_core.py:476-5343 | FIX (shared _y_discrete) | FUTURE — compute a single _y_discrete once and thread it through the ~15 FE blocks in the 9.9k monolith (actively edited by the parallel session); needs a careful monolith pass + bench. |
| P-4 | P2(gated) | STRICT mi_from_codes recomputes y-marginal inside x-loop | _fe_batched_mi.py:113 | FIX | FUTURE — precompute the y-marginal once in STRICT mi_from_codes (opt-in STRICT path; parallel GPU-resident-FE territory). |
| P-5..10 | P2/3 | GPU/batch micro-opts (per-col launch loop, per-pair .get() sync, legacy argsort, host guard scan, double upcast, hardcoded split-N crossover no KTC) | _fe_batched_mi.py, batch_pair_mi_gpu.py, gpu.py | FUTURE (bench-gated, mostly opt-in STRICT) | FUTURE — bench-gated GPU/batch micro-opts, mostly opt-in STRICT; parallel GPU-resident-FE session's active area |
| P-11 | P3 | fused_propensity re-derives V/V2/classes | _fe_interaction_prerank.py:227 | FIX | DONE (p11, bit-identical ~15%) |

## Stopping / fallback (mrmr_crit_stopping.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| ST-1 | P2 | UAED elbow trim mixes raw/combined index spaces → support/output desync (uaed_auto_size on + engineered) | _fit_impl_core.py:9854 | FIX | DONE (st1) |
| ST-2 | P2 | count-floor top-up gates on _abs_floor(=0.0) → admits any non-constant noise | _finalise.py:254 | DOC/FUTURE (documented ≥K contract; add uninformative flag) | DOC — the count-floor top-up gating on _abs_floor is the intentional min_features_fallback >=K contract (documented). A stricter floor would break >=K; the honest improvement (set fallback_metadata_['uninformative'] when a topped-up column sits within its null) is a small FUTURE diagnostic, not a semantics change. |
| ST-3 | Low | fallback support_ omits dtype=int64 → int32 on Windows | _finalise.py:281 | FIX | DONE (33343a58) |
| ST-4 | Low | ran_out_of_time_ misses screen-level timeout | _fit_impl_core.py:6711 | FIX | DONE (st4) |

## Rollup
6 P1, 13 P2, ~13 Low/P3 across 7 agents. FE-F1 DONE. Remainder tracked above; being worked in batches with tests
+ benchmarks and pushed frequently. A verifier agent runs at the end to confirm every ID reached a terminal status.

## NEW DISCOVERIES (found while implementing the critique fixes)
| ID | Sev | Finding | file:line | Status |
|----|-----|---------|-----------|--------|
| ND-1 | P2? | Poly-FE transform-replay KeyError: a 'poly_<coeff-array>' unary name (registered in unary_transformations at _fit_impl_core.py:6229, applied at fit via hermval in feature_engineering.py:301) is NOT persisted in the recipe so _apply_unary_binary can't replay it. Traced: build_unary_binary_recipe has no poly-coeff path, and unary_transformations is NOT in scope at the _step_score builder; _fe_auto_escalation builds separate 'poly_{basis}' recipes. Opt-in (fe_max_polynoms=0). | _recipe_unary_binary.py:66 ; _fit_impl_core.py:6229 ; feature_engineering.py:301 | TODO (dedicated session) — store the coeff array in recipe.extra (poly_<side>_coef) at the mechanism(s) that build poly recipes, recognize the 'poly_' prefix as a pseudo-name in _apply_unary_binary and replay via hermval(vals, coeffs); then a fe_max_polynoms fit->transform regression test. Spans 3+ files/mechanisms. |

| ND-2 | P2? | test_biz_val_mrmr_sample_weight_fe [mi_greedy]: weight-vs-duplication RAW-set equivalence dropped 3/3 -> 1/3. DIAGNOSED: mi_greedy binning is UNWEIGHTED (no sample_weight in _mi_greedy_cmi_fe), and MRMR.fit(sample_weight) RESAMPLES WITH REPLACEMENT (_maybe_resample_for_sample_weight) -- so weighted (stochastic resample) vs physical-duplication are only APPROXIMATELY equivalent. The parallel session's sync-free quantile binning (branchless dedup + manual percentile, 51f7ad5c/7d783ac1) increased tie-sensitivity, so the BORDERLINE x0 pick now diverges between the two paths (x1, the primary signal, is still recovered by both on every seed). | _mi_greedy_cmi_fe.py binning ; tests/.../test_biz_val_mrmr_sample_weight_fe.py:185 | TODO (dedicated) — determine if the sync-free binning tie-handling can be made weight-stable (restore exact match) OR re-frame the over-specified exact-raw-set assertion to signal-recovery + high-overlap (the resample path is stochastic, so exact borderline-feature match is not a guaranteeable contract). Needs the parallel binning tie-analysis; not a rushed call. |
