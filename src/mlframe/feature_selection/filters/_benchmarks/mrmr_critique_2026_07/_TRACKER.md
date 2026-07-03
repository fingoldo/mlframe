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
| S-F2 | P1(jmim) | JMIM scored by joint-MI but confirmed against CMIM null → statistic mismatch | fleuret.py:257 | FIX (thread use_jmim) | TODO |
| S-F3 | P2(jmim) | Fleuret `**(nexisting+1)` exponent wrongly amplifies JMIM joint-MI>1 | evaluation.py:363 | FIX | TODO |
| S-F4 | P2(off-dflt) | positive_mode/extra_knowledge breaks partial_gains monotonicity | evaluation.py:411,618 | FIX | TODO |
| S-F5 | Low | redundancy_aggregator typos silently degrade to Fleuret | _mrmr_class.py:~3355 | FIX (validate) | DONE (33343a58) |

## Numerical / statistical (mrmr_crit_numstat.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| N-F1 | P1 | Observed relevance uses Miller-Madow but permutation null uses plug-in → over-reject + double bias subtraction | permutation.py:625 vs 341/229/165 | FIX (thread use_mm into null kernels) | TODO |
| N-F2 | P1 | Relevance MM+null-debiased, redundancy raw plug-in → objective biased vs high-cardinality candidates | evaluation.py:548 / _entropy_kernels.py:342 | FIX (validate wide; CMI bias in _fe_cmi_redundancy_null.py:200) | TODO |
| N-F3 | P2 | `_perm_pvalue(full_budget=)` overstates confidence on pile-up early breaks | permutation.py:62 | FIX | TODO |
| N-F4 | P2 | fixed alpha=0.05, no multiple-testing correction across candidates | evaluation.py:549 | FIX (BH/BY) or DOC | TODO |
| N-F5 | P2 | 32-perm null-mean variance can flip near-tied selection | permutation.py:46 | DOC/FUTURE (shrinkage) | TODO |
| N-F6 | P3 | Chao-Shen not matched by its null (≡ N-F1 generalized) | — | FIX-with-N-F1 | TODO |
| N-F7 | P3 | analytic chi-square null assumes fixed occupancy on tied equi-freq bins | _analytic_mi_null.py | DOC | TODO |
| N-F8 | P3 | analytic_batch_noise_gate no per-column perm fallback | _analytic_mi_null.py:253 | DOC/FIX | TODO |

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
| EN-1 | P2 | asymmetric key coercion in cross/TE builders (safe today via pre-canonicalization; fragile) | _encoding_recipes.py:110,331,375 | FIX (canonicalize keys in builders) | TODO |
| EN-2 | Low | polars branch crashes when pandas absent | _encoding_recipes.py:73 etc | FIX (guard pd) | DONE (33343a58) |
| EN-3 | Low | _apply_mi_greedy_transform forces float64 on source cols | _missingness_ratio_recipes.py:37 | FIX (validate numeric at build) | DONE (batch-lows) |
| EN-4 | Low | integer-factorize numeric source truncates floats at replay | _recipe_extract.py:193,198 | FIX (round-to-nearest) | DONE (batch-lows) |
| EN-5 | Low | target_aware_group_bin global-fallback OOF optimism (== FE-F3) | _grouped_quantile_fe.py:413 | dedup of FE-F3 | see FE-F3 |

## FE families — extra families (mrmr_crit_fe_extra.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| EX-1 | P2 | `_is_argmax_eligible` finiteness guard is a tautology → NaN columns not excluded | _conditional_gate_fe.py:357 | FIX (isfinite(a).all()) | DONE (33343a58) |
| EX-2 | Low | argmax/gate meaning shifts on serve-only NaN (no replay NaN policy) | _conditional_gate_fe.py:147 | FIX/DOC | TODO |
| EX-3 | Low/P2 | conditional-gate tau optimized on same in-sample y (selection optimism) | _conditional_gate_fe.py:718 | DOC | TODO |
| EX-4 | Low/P2 | RankGauss stores full sorted non-unique array (memory) + docstring wrong ("unique") | _extra_fe_families.py:795 | FIX (unique+counts / doc) | DONE-doc (batch-lows) |

## Performance (mrmr_crit_perf.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| P-1 | P1 | whole-matrix `factors_data.copy()` every screen call (100GB-frame rule) | _screen_predictors.py:530 | FIX (mutate-and-restore / gate) | TODO |
| P-2 | P2 | forced float64 copy of FE candidate matrix even when already f64/contig | _fe_cpu_batch.py:64,80,75 | FIX (branch on dtype/flags) | TODO |
| P-3 | P2 | redundant target re-discretization across ~15 FE blocks | _fit_impl_core.py:476-5343 | FIX (shared _y_discrete) | TODO |
| P-4 | P2(gated) | STRICT mi_from_codes recomputes y-marginal inside x-loop | _fe_batched_mi.py:113 | FIX | TODO |
| P-5..10 | P2/3 | GPU/batch micro-opts (per-col launch loop, per-pair .get() sync, legacy argsort, host guard scan, double upcast, hardcoded split-N crossover no KTC) | _fe_batched_mi.py, batch_pair_mi_gpu.py, gpu.py | FUTURE (bench-gated, mostly opt-in STRICT) | TODO |
| P-11 | P3 | fused_propensity re-derives V/V2/classes | _fe_interaction_prerank.py:227 | FIX | TODO |

## Stopping / fallback (mrmr_crit_stopping.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| ST-1 | P2 | UAED elbow trim mixes raw/combined index spaces → support/output desync (uaed_auto_size on + engineered) | _fit_impl_core.py:9854 | FIX | TODO |
| ST-2 | P2 | count-floor top-up gates on _abs_floor(=0.0) → admits any non-constant noise | _finalise.py:254 | DOC/FUTURE (documented ≥K contract; add uninformative flag) | TODO |
| ST-3 | Low | fallback support_ omits dtype=int64 → int32 on Windows | _finalise.py:281 | FIX | DONE (33343a58) |
| ST-4 | Low | ran_out_of_time_ misses screen-level timeout | _fit_impl_core.py:6711 | FIX | TODO |

## Rollup
6 P1, 13 P2, ~13 Low/P3 across 7 agents. FE-F1 DONE. Remainder tracked above; being worked in batches with tests
+ benchmarks and pushed frequently. A verifier agent runs at the end to confirm every ID reached a terminal status.
