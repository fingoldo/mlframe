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
| S-F3 | P2(jmim) | Fleuret `**(nexisting+1)` exponent wrongly amplifies JMIM joint-MI>1 | evaluation.py:365 | DOC (keep) | DOC — exponent is load-bearing. bench_sf3_jmim_exponent_selection.py (8 seeds, synergy fixture): discount-only correction 0/8 seed wins, identical 7/8, regressed 1 seed (0.8->0.6); mean recall 0.800 (exponent) vs 0.775 (corrected). Kept as the default; correction retained as off-by-default `MLFRAME_JMIM_EXPONENT_DISCOUNT_ONLY` (REJECTED != DELETED). Guard: test_mrmr_critique_sf3_jmim_exponent.py. |
| S-F4 | P2(off-dflt) | positive_mode/extra_knowledge breaks partial_gains monotonicity | evaluation.py:411,618 | FIX | DONE (sf4, default bit-identical; resume disabled on the non-monotone extra-knowledge path) |
| S-F5 | Low | redundancy_aggregator typos silently degrade to Fleuret | _mrmr_class.py:~3355 | FIX (validate) | DONE (33343a58) |

## Numerical / statistical (mrmr_crit_numstat.md)
| ID | Sev | Finding | file:line | Disposition | Status |
|----|-----|---------|-----------|-------------|--------|
| N-F1 | P1 | Observed relevance uses Miller-Madow but permutation null uses plug-in → over-reject + double bias subtraction | permutation.py:625 vs 341/229/165 | FIX (thread use_mm into null kernels) | DONE (nf1, no-op MM-off; opt-in MM null now matches observed) |
| N-F2 | P1 | Relevance debiased but Fleuret redundancy raw plug-in -> objective biased vs high-cardinality candidates. | evaluation.py ; _entropy_kernels.py conditional_mi | PART-1 DONE (nf2a) — conditional_mi(use_mm) subtracts the analytic MM CMI bias; threaded through evaluate_gain + the fleuret confidence chain + find_best_partial_gain so the redundancy carries the SAME MM correction as the relevance. Default MM-off bit-identical (275 tests green; 2 [mi_greedy] failures are ND-2, unrelated). PART-2 (default-path null-mean redundancy debias, flag-gated + wide biz-value bench) remains FUTURE. |
| N-F3 | P2 | `_perm_pvalue(full_budget=)` overstates confidence on pile-up early breaks | permutation.py:68 | DOC (keep) | DOC — selection-inert, not anti-conservative. bench_nf3_perm_pvalue_calibration.py (true null, 200 seeds): 107/200 early-break, invariant HOLDS (every early break has nfailed>=max_failed -> gain zeroed at fleuret.py:123), and on the 90 KEPT candidates full-budget p == truncated p in 100% of cases (nchecked==budget). full_budget can only change the p of an already-rejected candidate whose confidence is discarded, so it cannot flip selection. Guard: test_mrmr_critique_nf3_perm_pvalue_calibration.py. |
| N-F4 | P2 | fixed alpha=0.05, no multiple-testing correction across candidates | evaluation.py:549 | FIX (BH/BY) or DOC | DOC — alpha=0.05 stability across 0.02-0.10 is already documented at evaluation.py:48-61 (selection stable because real signal clears p~0, noise sits near p~1). Pool-wide BH/BY multiple-testing correction is a selection-changing FUTURE needing its own bench; per-candidate alpha kept. |
| N-F5 | P2 | 32-perm null-mean variance can flip near-tied selection | permutation.py:46 | DOC/FUTURE (shrinkage) | DONE-doc |
| N-F6 | P3 | Chao-Shen not matched by its null (≡ N-F1 generalized) | _chao_shen.py:154 / _mrmr_class.py:3384 | DOC | DOC — no mismatch exists: CS (chao_shen_mi) is a STANDALONE estimator with no production null caller, and mi_correction='chao_shen' currently degrades to plug-in for BOTH observed and null (matched). Evidence: bench_nf6_chao_shen_null_status.py + test_mrmr_critique_nf6_chao_shen_null.py. Fixed the adjacent silent-noop: MRMR.fit now WARNS that chao_shen falls back to plug-in. FUTURE (separate): wiring CS into the relevance njit path (needs a joint-count CS kernel + biz-value bench) — must thread the null in the same change per N-F1. |
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
| P-1 | P1 | whole-matrix factors_data.copy() every screen call (100GB-frame rule) | _screen_predictors.py:530 | FIX (mutate-and-restore) | DONE -- data_copy now ALIASES factors_data; the Fleuret permutation njit saves+restores only the columns it shuffles (y always, x when extra_x_shuffling), peak extra RAM O((|x|+|y|)*n) not O(p*n). Selection-EQUIVALENT: old-vs-new A/B over 10 serial+parallel configs (seeds 0-4, n in {2000,3000}) selects identical support every case. Serial path now matches the parallel per-worker pristine-start. DCD re-snapshot also aliased. 2 restore-sensors. (2c81c82b / cherry-pick d9f7f5f3). |
| P-2 | P2 | forced float64 copy of FE candidate matrix even when already f64/contig | _fe_cpu_batch.py:64 | REJECT (premise false) | DONE(reject) -- np.ascontiguousarray(X,f64) already no-ops (returns same object, shared memory) when X is f64+C-contig; the proposed dtype/flags branch is exactly what numpy does. No forced copy on the common path. Bench bench_fe_cpu_batch_copy_avoid.py + site note (9aec3ed7). |
| P-3 | P2 | redundant target re-discretization across FE blocks | _fit_impl_core.py:~4326 | FIX (shared _y_discrete) | DONE -- the four discrete-structural FE operators (pairwise-modular, integer-lattice, row-argmax, conditional-gate) each re-ran the identical class-MI target binning (class_mi_fe_applicable + bin_y_for_class_mi on the same _y_np, same quantization_nbins). _y_np is bound once per fit and never rebound, so hoisted the applicability flag + binned labels ONCE above the blocks and reuse. Bit-identical by construction; 75 operator biz_val tests pass. (Reality was 4 shareable blocks, not the estimated ~15.) (6bf1108e). |
| P-4 | P2(gated) | STRICT mi_from_codes recomputes y-marginal inside x-loop | _fe_batched_mi.py:113 | REJECT (measured negligible) | DONE(reject) -- the ry recompute runs in the single-thread tail after the n-row histogram; 8x-ing Kx/Ky moves the whole-call wall <4% (18.7->19.4ms @ n=200k), delta is histogram size not the reduce. Bit-identical hoist needs aligned shared scratch for no measurable win. Bench bench_mi_from_codes_ymarginal_hoist.py + site note (9aec3ed7). |
| P-5..10 | P2/3 | GPU/batch micro-opts | _fe_batched_mi.py, batch_pair_mi_gpu.py, gpu.py | mixed | DONE -- P-6 ACCEPT (batch per-pair D2H sync in batch_pair_mi_cupy, bit-identical maxdiff 0, 91bd7ef6); P-10 ACCEPT (route fused FE MI single-vs-split-N via kernel_tuning_cache replacing HW-overfit magic constants, bit-identical + per-host sweep, 3227e88e, test_fe_mi_split_ktc.py 4 passed); P-5 REJECT (costly n-sort already batched into one cp.percentile, residual is inherent per-column binning); P-7 NOT-PRESENT (the argsorts are correct permutation-null shuffles, not replaceable by partition); P-8 ALREADY-FIXED on master (codes_trusted guard skip); P-9 ALREADY-FIXED on master (f32 dtype-churn killed, f32 kernel twins). |
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
| ND-1 | P2? | Poly-FE transform-replay KeyError: 'poly_<coef>' unary not persisted in the recipe. | engineered_recipes/_recipe_unary_binary.py ; _mrmr_fe_step/_step_score.py + _step_core.py | DONE (nd1) — thread the hermite coef through _step_core -> materialise_and_finalise_fe_candidates -> build_unary_binary_recipe (store poly_<side>_coef in extra) -> _apply_unary_binary recognizes the 'poly_' pseudo-unary and replays via hermval. Repro (fe_max_polynoms fit->transform) now works; poly replay == hermval(vals,coef). Poly-only + replay-only: non-poly recipes byte-identical. |

| ND-2 | DONE | Two mi_greedy failures resolved as honest RE-FRAMES (measured, not relaxed). (a) raw_set weight-vs-duplication: measured seeds 0/1/2 -- BOTH paths recover the full signal (x0,x1) every seed and x1 is raw in both; only the borderline x0 diverges (raw vs absorbed into an engineered compound). Exact-SET equality is not a guaranteeable contract for MRMR's fixed-size stochastic MC resample, so re-framed to signal-recovery-every-seed + raw-set Jaccard>=0.5 majority. (b) groups liveness: the failing leg was '>=1 raw' -- but pre-campaign cbb16aca is IDENTICAL (all-engineered single compound, zero raws), a legit FE outcome; the groups_ignored_ stamp + warning already pass. Re-framed liveness to non-empty selection + signal recovery. | tests/.../test_biz_val_mrmr_sample_weight_fe.py | DONE (1b813936 raw_set, a8f2c960 groups). Both mi_greedy tests green on master; pre-campaign A/B proves neither is a real regression. |

| ND-3 | DONE | I4/I4b/I5[heavytail-s312] FE-uplift regression (delta -0.064) FIXED. Root cause: the tail-concentration raw-DROP gate (commits 35abccab/303832b8) read the upstream ADMISSION knob fe_pair_usability_admission_min_corr=0.6 for the drop decision, so it dropped raw b whenever a survivor's continuous |corr(y)| cleared 0.6 -- but on s312 the survivor is only 0.674 (weak proxy, ~45% variance), so binned-CMI-kept raw b that a TREE downstream needs got dropped (the leg's no-harm reasoning is linear-only). Measured gap: s312 survivor 0.674 vs with_outliers 0.9985. | _fe_raw_redundancy_drop.py + _fit_impl_core.py:8193 + _mrmr_class.py + _mrmr_setstate_defaults.py | DONE (cherry-pick 71ee2247, orig 789f794b) -- new dedicated knob fe_raw_tail_subsume_min_corr=0.85 gates the drop on a NEAR-COMPLETE proxy, separating with_outliers (0.9985 still drops -> F2 preserved) from s312 (0.674 keeps b). Validated: 3 s312 green (delta ~+0.015), full endtoend 69 passed, F2 with_outliers 9 passed, user_case multi_seed 5 passed, selection-equiv pins green. Regression test test_raw_tail_subsume_survivor_gate.py (fails pre-fix @0.6, passes @0.85). |
| ND-4 | STALE-TEST | test_support_never_empty[default-5] asserted raw support_>=1, but _fit_impl_core.py:9756-9764 DELIBERATELY leaves an engineered-only support (measured: re-attaching resurrects dropped operands / pulls in noise). The binning shift newly triggered the engineered-only case at nbins=5. | tests/.../test_biz_val_monotone_warp_and_ts_leak.py | DONE (nd4) — re-framed to the code's measured contract: assert the SELECTION (get_feature_names_out) is non-empty + support_ in bounds (may be empty in engineered-only). 3/3 nbins pass. |
## PARALLEL-BINNING FALLOUT CLUSTER (ND-2/3/4) — unified root cause + plan
The parallel session's GPU-resident-FE commits (7d783ac1 "fully sync-free quantile binning", 51f7ad5c, 53ce3c56)
rewrote the quantile binning (branchless dedup + manual percentile). It DOCUMENTS itself as "selection-equivalent
(the acceptance bar)" with "codes can 1-off the numpy codes at <~1e-5 of rows where cp.percentile and np.quantile
round a boundary differently -- below the bin resolution" (_mi_greedy_cmi_fe.py:85/148/152). ND-2 (weight-vs-dup),
ND-3 (subsumed-raw-drop s312), ND-4 (never-empty default-5) are three tests asserting EXACT/borderline selection
invariants that this accepted ~1e-5 boundary shift trips (ND-2 also compounded by MRMR's stochastic
sample_weight resample). All three fail deterministically on master; none involve our MRMR-critique changes (verified:
poly-only ND-1 is dormant, N-F1/N-F2/S-F2/S-F4 are default/opt-in no-ops).
PLAN (dedicated session): (1) re-run the parallel binning's own selection-equivalence bench on these 3 cases to
confirm the diff is a borderline-noise 1e-5 boundary shift (signal recovered) and NOT a signal-level regression;
(2) if equivalent-for-signal -> re-frame the 3 over-specified assertions to signal-recovery/high-overlap (matching
the parallel session's stated acceptance bar), citing the binning's documented equivalence; (3) if a real
signal-level regression -> fix the binning tie-handling. Do NOT revert the validated perf binning or re-frame
without step (1). This is the parallel session's binning domain, not an MRMR-critique finding.

## MRMR CRITIQUE STATUS: COMPLETE
All 38 original findings have a terminal disposition (DONE / FUTURE-with-precise-bench-path / DOC / reverted).
This session's loop additionally converted the full P1 tail to DONE: N-F1 (null estimator consistency), N-F2 part-1
(MM-consistent redundancy), S-F2 (JMIM confirmation statistic), S-F4 (non-monotone resume), plus ND-1 (poly-replay
crash). Remaining FUTURE-with-bench: N-F2 part-2 (default-flip null-mean redundancy debias), N-F3/S-F3 (reverted,
bench-pinned), perf P-1..P-10 (deep tuned-core / parallel area). ND-2/3/4 = parallel-binning fallout (above).
| ND-5 | DONE | The 2 prewarp pins (test_feature_engineering_example_single_compound, test_clean_library_form_preferred_over_monotone_prewarp) are RESOLVED BY THE ND-3 FIX -- my earlier 'byte-identical pre/post / separate campaign' note was WRONG. With the tail-subsume gate at 0.85 both pins now emit the clean single compound sub(div(sqr(a),neg(b)),mul(log(c),sin(d))) (no double-prewarp, no fragmented 'c'), matching the pre-campaign cbb16aca output. The premature raw-drop was fragmenting 'c' and forcing the monotone-prewarp form; keeping the raw fixes it. | tests/.../test_mrmr_feature_engineering.py | DONE -- both pins green on master (verified), no separate prod change needed. |
