# Screening/confirmation greedy loop + permutation nulls

14 findings, 5 proposals.

## Findings

### [P1] bug -- src/mlframe/feature_selection/filters/_confirm_predictor.py:587

**confirm_candidate's primary marginal permutation-confirmation call (mi_direct / mi_direct_gpu) never receives a seed, so it always uses the hardcoded default base_seed=0 regardless of ctx.random_seed.**

confirm_candidate reads `random_seed = ctx.random_seed` at line 531 but only ever uses it to build `_fleuret_base_seed` for the secondary conditional recheck (line 656). The primary calls at lines 572 (mi_direct_gpu) and 587 (mi_direct) pass no base_seed/random_seed kwarg at all, so mi_direct's `base_seed: int = 0` default is used on every single call, for every candidate, every round, of the entire fit. Because parallel_mi_prange's LCG state is seeded purely from `base_seed` and the permutation index (never from the candidate's data), this means the SAME row-shuffle sequence is applied to y for every distinct candidate's confirmation test across the whole fit -- calling `MRMR(random_seed=1)` vs `MRMR(random_seed=2)` produces byte-identical confirmation-permutation draws, contradicting the explicit intent stated in the neighboring comment block (line 866-868: 'random_seed is now also threaded in end-to-end ... so the random_seed= knob actually moves this component's draws') which is true only for the opt-in cmi_perm_stop/cpt_test paths, not for this default-on confirmation gate. Verified via grep: no call site anywhere in src/ passes `base_seed=` or `random_seed=` into mi_direct/mi_direct_gpu.

### [P1] bug -- src/mlframe/feature_selection/filters/evaluation.py:566

**evaluate_candidate's baseline relevance-null calls (mi_direct / mi_direct_gpu, baseline_npermutations budget) also never receive a seed, so ctx.random_seed never affects the per-candidate baseline permutation gate that runs on every single scored candidate.**

evaluate_candidate accepts `random_seed: int | None = None` as a parameter (used further down only for `_cmi_cpt_seed`, the opt-in CMI-perm-stop/CPT gates) but the mi_direct_gpu call at line 566 and the mi_direct call at line 611 pass neither base_seed nor random_seed, so both silently use base_seed=0 for the baseline-relevance permutation null on every candidate, every round -- the same-seed-reuse issue as the confirm_candidate finding, but for the higher-frequency baseline scoring path that runs during score_candidates rather than only at confirmation time.

### [P1] bug -- src/mlframe/feature_selection/filters/_confirm_predictor.py:656

**_fleuret_base_seed omits the candidate's own identity from its hash, so every distinct candidate confirmed at the same selected_vars depth within one greedy round gets the IDENTICAL permutation stream for the Fleuret conditional recheck.**

`_fleuret_base_seed = int(((int(random_seed or 0) * 2654435761) + len(selected_vars) + 1) & 0xFFFFFFFFFFFFFFFF)` depends only on random_seed and len(selected_vars) -- never on X or a cand_idx. get_fleuret_criteria_confidence's LCG-driven Fisher-Yates shuffle of the y (and x, when extra_x_shuffling) columns is a pure function of base_seed and n (row count), independent of the shuffled column's content, so two different candidates X1 and X2 confirmed back-to-back at the same |selected_vars| receive the exact same row-permutation sequence applied to the same y data -- their permutation p-values are not independent draws. This is the SAME bug class the same file already fixed for cmi_perm_stop/cpt_test just 8 lines earlier at evaluation.py:868 (`_cmi_cpt_seed = hash((random_seed, cand_idx, tuple(sorted(selected_vars))))`, explicitly folding in cand_idx 'so each round's null draw independent of the others for the same candidate') -- but the fix was not applied to this, the primary and far more heavily used confirmation seed in the same 2026-07-19 threading pass.

### [P1] cpu_gpu_parity -- src/mlframe/feature_selection/filters/_confirm_predictor.py:552

**GPU and CPU confirmation permutation draws are sourced from fundamentally different (and non-equivalent) RNG mechanisms for the identical statistical test: GPU consumes a persistent global cupy stream seeded once per fit, CPU resets to the fixed base_seed=0 on every call.**

mi_direct_gpu (gpu.py) takes no base_seed/seed parameter at all -- its randomness comes from whatever state cupy's global RNG is in, seeded once at the top of screen_predictors via `cp.random.seed(random_seed)` and then advancing sequentially across every subsequent GPU call in the fit. mi_direct's CPU path is architected around an explicit per-call `base_seed` specifically to avoid a shared mutable stream (per the module's own docstring: global numpy RNG 'raced under joblib parallel workers'), but per findings above that base_seed is never actually populated from ctx.random_seed, so it resets to the identical 0-seeded stream on every call. The result: a fit run with use_gpu=True draws a different permutation for every confirmed candidate (advancing global state), while the same fit run with use_gpu=False draws the identical permutation for every confirmed candidate -- a real behavioral divergence between the two backends for the same nominal algorithm, not just an FP-rounding difference.

### [P1] perf -- src/mlframe/feature_selection/filters/_screen_predictors.py:602

**The 2026-07-19 joblib-pool retirement hardcoded `workers_pool = None` unconditionally, which silently kills pool reuse for the STILL-ACTIVE Fleuret conditional-confirmation parallel path and makes the `seed_workers_pool` parameter completely dead.**

`seed_workers_pool` is accepted as a screen_predictors parameter (line 245) and documented as avoiding a fresh joblib.Parallel spawn 'on every one of the ~3 screen/re-screen rounds per fit' -- but it is never read anywhere in the function body (verified by grep: only the signature default and two docstring mentions exist). The comment at line 601 acknowledges 'seed_workers_pool is likewise accepted-but-unused for this purpose' for the retired evaluate_candidates pool, but does not flag that `workers_pool = None` (the same local variable) is also what gets stored on `ctx.workers_pool` and threaded into confirm_candidate -> get_fleuret_criteria_confidence_parallel (fleuret.py line 127-129), which the SAME comment says 'still branch[es] on it' -- i.e. is not retired. Because ctx.workers_pool is always None, `if workers_pool is None: workers_pool = Parallel(n_jobs=n_workers, **parallel_kwargs)` in fleuret.py fires on EVERY confirm_candidate call that takes the parallel branch (n_workers>1 and full_npermutations > NMAX_NONPARALLEL_ITERS=2 -- note the default full_npermutations=3 already exceeds this threshold, so only n_workers>1 gates it), rebuilding a brand-new thread pool per candidate confirmation instead of once per screen call as seed_workers_pool was designed to provide.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_confirm_predictor.py:656

**No test anywhere in the suite exercises whether two different candidates confirmed in the same greedy round get statistically-independent permutation draws (CPU mi_direct path or the Fleuret conditional recheck) -- the exact property the three seed-threading bugs above violate.**

Grepping the whole tests/ tree for `mi_direct(...base_seed=` or `_fleuret_base_seed` turns up zero production-parity assertions; test_mrmr_provenance_seed.py only checks that the ctor's random_seed value is recorded in provenance metadata, never that it actually changes the permutation draws used inside confirm_candidate/evaluate_candidate. A test asserting `random_seed=1` vs `random_seed=2` produce different confirmation-permutation sequences (or that two distinct candidates at the same round produce different shuffles) would have caught findings #1-3 directly.

### [P2] bug -- src/mlframe/feature_selection/filters/evaluation.py:162

**should_skip_candidate's DCD prune-mask lookup swallows any exception with a bare `except Exception: pass` and zero logging (not even DEBUG).**

If `_dynamic_cluster_discovery.should_be_pruned` raises for any reason (an import error, an attribute error from a version-skewed DCDState, a genuine bug), this per-candidate hot-path call silently falls through to 'not pruned' with no trace anywhere in the logs -- a real DCD/should_be_pruned regression would look like DCD simply stopped pruning candidates, with no diagnostic breadcrumb to explain why.

### [P2] bug -- src/mlframe/feature_selection/filters/evaluation.py:962

**find_best_partial_gain's DCD-prune import guard is a bare `except Exception: _should_be_pruned = None` with no logging.**

A genuine import-time error in _dynamic_cluster_discovery (not just 'module absent') is indistinguishable from the expected no-DCD case; the redirect-target search then silently stops excluding DCD-pruned candidates without any log entry identifying why, re-opening exactly the redirect-loop bug the surrounding docstring describes as previously causing '6 features -> 2, -4% downstream AUC.'

### [P2] bug -- src/mlframe/feature_selection/filters/_screen_predictors_gate.py:67

**build_dcd_state only logs a DCD-init failure when `verbose` is truthy; at the library's own default verbose=0 (screen_predictors' default), a DCD init exception is completely unlogged (not even at DEBUG).**

A user running MRMR with dcd_config set but verbose=0 (the default) who hits a DCD initialization bug gets silent fallback to the legacy non-DCD path with zero diagnostic trace -- they only discover DCD never engaged by noticing selection quality, with no log line anywhere pointing at the root cause.

### [P2] bug -- src/mlframe/feature_selection/filters/permutation.py:651

**mi_direct's analytic-null-path gate check is a bare `except Exception: _analytic_ok = False` with no logging, unlike every other exception handler in this same function (which all log at WARNING/DEBUG with the exception).**

A real bug in analytic_null_enabled()/analytic_null_min_n()/use_su_normalization() (not just 'scipy absent') silently routes every call back to the permutation path with no trace, inconsistent with the rest of the file's convention (e.g. the GPU circuit breaker two blocks below logs at WARNING on first fault) -- a regression here would look like 'the analytic-null speedup mysteriously stopped engaging' with nothing in the log to explain it.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/_permutation_null_pair_resident.py:137

**pooled_pair_permutation_null_joint_mi_floor_cupy's marginal-entropy (h_x) precompute launches one cp.bincount GPU kernel per pair in a Python for-loop, the exact 'per-pair launch trap' this module's own docstring calls out and avoids everywhere else (including the per-shuffle joint-MI loop 40 lines below it in the same function, which uses a single batched flat-index bincount instead).**

On a wide prospective-pair pool (n_pairs in the hundreds to low thousands, the regime this floor exists to guard), this precompute issues n_pairs separate small GPU kernel launches once per pooled_pair_permutation_null_joint_mi_floor_cupy call (once per screen round, not per-shuffle, so the cost is bounded but real) instead of reusing the same base_pair/per_pair_extent flat-index padding trick the per-shuffle loop already implements a few lines later -- the fix is mechanical (batch h_x the same way h_xy is batched) and the module's own docstring already states the design principle this violates.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_screen_predictors.py:245

**No test verifies that seed_workers_pool (or any pool) is actually reused across repeated confirm_candidate calls within one screen_predictors invocation for the still-active Fleuret parallel confirmation path -- only the retired evaluate_candidates pool has a dedicated 'stays serial' regression test (test_screen_predictors_evaluate_candidates_serial_only.py).**

A future contributor re-enabling n_workers>1 in production would have no test signal that the Fleuret confirmation path silently rebuilds a fresh joblib.Parallel object on every candidate instead of reusing one -- the perf regression in finding #4 would ship undetected indefinitely.

### [P2] design -- src/mlframe/feature_selection/filters/_permutation_null.py:1

**GPU residency across the maxT permutation-null floor family (order-1/order-2, resident + shufflegen KTC variants) is clean: no unnecessary host<->device round trips found.**

_(no issues found in this cluster for this angle)_

### [P2] design -- src/mlframe/feature_selection/filters/evaluation.py:1

**No significant silent-numeric-coercion issues (implicit float->int truncation feeding an njit kernel, unguarded NaN/inf before a kernel that assumes finite input) were found in this cluster beyond a single very-minor `int(baseline_npermutations)` truncation in evaluate_candidate that only matters if a caller passes a non-integer permutation count.**

_(no issues found in this cluster for this angle)_

## Proposals

### (coverage_gap) Regression test: random_seed must change CPU confirmation-permutation draws

Add a test that runs the same small fixture through screen_predictors/confirm_one_predictor with random_seed=1 and random_seed=2 (use_gpu=False, n_workers=1, full_npermutations large enough to be observable) and asserts the recorded permutation outcomes (e.g. cached_confident_MIs confidences, or a monkeypatched spy on mi_direct's base_seed argument) actually differ. This would fail today (base_seed is always the mi_direct default of 0) and pass once findings #1/#2 are fixed.

### (coverage_gap) Regression test: distinct candidates in one round get independent Fleuret-confirm permutation streams

Confirm two candidates at the same len(selected_vars) within one confirm_one_predictor call (use_simple_mode=False so the Fleuret conditional recheck fires) and assert their applied y-row-permutation sequences differ, e.g. by spying on _fleuret_shuffle_col_lcg's state trajectory or by constructing a fixture where identical-vs-independent shuffles are statistically distinguishable. Would catch finding #3.

### (residency_step) Batch the h_x marginal-entropy precompute in the order-2 resident maxT floor

In _permutation_null_pair_resident.py's pooled_pair_permutation_null_joint_mi_floor_cupy, replace the per-pair Python-loop cp.bincount calls building h_x with the same base_pair/per_pair_extent flat-index + single cp.bincount (or scatter_add) approach already used 40 lines later for the per-shuffle joint entropy h_xy -- removes n_pairs separate kernel launches per call.

### (other) Fold candidate identity into confirm_candidate's base_seed derivations

Mirror evaluation.py's _cmi_cpt_seed pattern (hash((random_seed, cand_idx or a stable hash of X, tuple(sorted(selected_vars))))) for both the primary mi_direct/mi_direct_gpu confirmation call in confirm_candidate and for _fleuret_base_seed, so every (candidate, round) pair gets an independent-yet-reproducible permutation stream, and thread the same derived seed into evaluate_candidate's baseline mi_direct/mi_direct_gpu calls.

### (other) Re-thread seed_workers_pool through ctx for the Fleuret parallel confirmation path

Since _screen_predictors.py's workers_pool retirement (line 602) collaterally kills pool reuse for get_fleuret_criteria_confidence_parallel, restore seed_workers_pool wiring specifically for that still-active path (build once per screen_predictors call when n_workers>1, store on ctx.workers_pool, return it via seed_workers_pool for the next round) while keeping the score_candidates pool retired as documented.
