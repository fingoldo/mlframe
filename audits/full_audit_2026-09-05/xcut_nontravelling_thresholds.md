# Cross-cutting audit: tests that discriminate via a number that does not travel between machines

**Scope:** `tests/` (3,493 files), AST scan of every `assert` containing a numeric-literal comparison,
classified into TIME / FIT / HW / FLOATEQ buckets (2,177 raw comparison sites), then manual reading of
the candidates. Read-only; no test file was modified. The suite was not run.

**Method note.** The AST pass flagged 281 wall-clock/ratio comparison sites. Only **41 of 3,493 test
files** import any of the conftest widening helpers (`perf_time_budget`, `perf_speedup_floor`,
`skip_if_host_contended`, `numba_disabled_timeout`, `skip_under_numba_disabled_jit`); **280 of the 281
timing comparison sites sit in files that import none of them.** That ratio is the headline finding:
the helpers exist and are well documented in `tests/conftest.py:282-380`, and they are essentially
unused.

---

## Findings

### NT-01 [P2] prewarm-cache-budget-dies-under-NUMBA_DISABLE_JIT
**File:** tests/feature_selection/test_prewarm.py:287, tests/feature_selection/test_prewarm.py:289, tests/feature_selection/test_prewarm.py:75
**Summary:** `warm_elapsed < 0.5` / `second_elapsed < 0.5` / `elapsed < 30.0` are trusted to prove that
`prewarm_fs_numba_cache` covered a dispatcher signature. The number is a proxy for "no JIT compile fired
here".
**Failure scenario:** The nightly `numba-coverage.yml` job runs with `NUMBA_DISABLE_JIT=1`. There is then
no dispatcher cache and no compile at all; the interpreted kernel on the same input runs 10-1000x slower
than the compiled one (per conftest's own docstring at tests/conftest.py:291-295). The 0.5s bound is
breached on a completely healthy build: **false red**, and permanently so, not flakily. Symmetrically,
on a machine slow enough that even a cold compile lands under 0.5s the assertion would pass with the
prewarm removed: **false green**.
**Evidence:** Read the file; no import of `perf_time_budget` or `skip_under_numba_disabled_jit`. The
comment at line 286 states the cold-compile reference is "~3-5s on the same machine" -- an explicitly
host-anchored number. `skip_under_numba_disabled_jit` (tests/conftest.py:319) was written for exactly
this class ("JIT-cache-artifact checks have no valid answer once compilation itself is disabled") and is
unused here.
**Suggested fix:** The direct property is *whether the signature is in the dispatcher's compiled-signature
table*, which numba exposes: assert the target signature is present in
`compute_mi_from_classes.signatures` (or `.nopython_signatures`) after prewarm, and that calling with the
test's dtypes adds no new entry to it. That is a structural check, immune to host speed, and it also
distinguishes "prewarmed" from "fast anyway" -- which the timer cannot. Guard the whole test with
`skip_under_numba_disabled_jit` since a signature table is meaningless with JIT off.

### NT-02 [P2] brier-logloss-prewarm-50ms-proxy
**File:** tests/metrics/test_prewarm_bool_dtype.py:46, tests/metrics/test_prewarm_bool_dtype.py:68
**Summary:** `elapsed_ms < 50.0` is trusted to prove the `(bool, float64)` signature of
`_fast_brier_score_loss_par` / `_fast_log_loss_binary_par` is prewarmed.
**Failure scenario:** Same as NT-01. Under `NUMBA_DISABLE_JIT=1` the interpreted parallel loop over
n=1000 will not be a fresh JIT compile but may still exceed 50ms; under normal JIT on a contended box a
1000-element call plus cProfile/coverage instrumentation can also cross 50ms. **False red.** The
converse is worse: if someone deletes the prewarm call and the CI box happens to compile the tiny
signature in under 50ms (small kernels do compile fast), the test stays green while the contract is
broken -- **false green**.
**Evidence:** Read both tests; the comment at lines 43-45 explicitly names the discriminating gap as
"typically <1ms" vs "1500-4000ms" cold compile, i.e. the test author knew the real signal is a 3-order
jump and then encoded a middle number. No conftest helper imported.
**Suggested fix:** Assert on the dispatcher signature table as in NT-01 -- `_fast_brier_score_loss_par.signatures`
must already contain `(bool[:], float64[:])` immediately after `_ensure_prewarmed()` and before the timed
call. That is the actual contract ("this signature is prewarmed"), not a timing shadow of it.

### NT-03 [P2] gpu-shared-fused-kernel-2s-budget
**File:** tests/feature_selection/gpu/test_batch_pair_mi_shared_fused.py:170
**Summary:** `elapsed < 2.0` is trusted to prove the CUDA shared-memory kernel still uses a parallel
(not serial) reduction.
**Failure scenario:** The comment at lines 167-169 anchors the number to "the reference host" (~0.1-0.5s
parallel, ~3s serial). On a lower-end or older GPU, or a GPU shared with another process (this repo runs
concurrent worktree sessions on one box), a correctly-parallel kernel at n=50000/k=20/nbins=5000 can
exceed 2.0s: **false red**. On a much faster GPU the *serial* reduction could land under 2.0s:
**false green** for the exact regression the test names.
**Evidence:** Read the test; no `perf_time_budget`. The discriminator (parallel vs serial reduction) is a
launch-geometry property, not a wall-clock property.
**Suggested fix:** Assert the launch geometry directly. The repo already exposes a resident-state
`launch_config(...)` returning `threads` / `shared_per_block` / `use_fused`
(tests/feature_selection/gpu/test_gpu_strict_resident_scaffold.py:67 reads it), so assert that the kernel
was launched with more than one thread per reduction block for this shape -- that is what "not serial"
means and it is hardware-independent. Keep a wall-clock check only as a coarse hang guard, wrapped in
`perf_time_budget`.

### NT-04 [P2] cmim-hotpath-5s-budget
**File:** tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_linear_preselect/test_cmim_hotpath_perf.py:163
**Summary:** `elapsed < 5.0` is trusted to prove "the cached-yz / factorize-pack fast path" has not
regressed.
**Failure scenario:** Under `NUMBA_DISABLE_JIT=1` or a 2-vCPU shared runner, the healthy fast path
exceeds 5s: **false red**. On a fast box the *uncached* path may also finish under 5s at n=2500:
**false green**.
**Evidence:** Read the test; a warm-up call is excluded from the budget (good), but no `perf_time_budget`.
**Suggested fix:** The cache is the contract. Count cache hits/misses (or spy on the factorize-pack
entry point via monkeypatch and assert it is called once, not once-per-pair) across the second call. A
call-count assertion is exact and travels.

### NT-05 [P2] cprofile-wrapped-wall-budgets
**File:** tests/feature_selection/fe/test_conditional_dispersion_fe.py:435, tests/feature_selection/fe/basis/test_wavelet_basis_fe.py:439, tests/reporting/test_charts_confusion_margins.py:263, tests/reporting/test_charts_prediction_stability.py:279
**Summary:** Four budgets (`wall < 12.0`, `elapsed < 5.0`, `elapsed < 5.0`, `elapsed < 8.0`) are measured
*with `cProfile.Profile()` enabled around the timed region*.
**Failure scenario:** cProfile's per-call overhead is a function of call count, and call count varies with
Python minor version, numba JIT state (JIT-off means orders of magnitude more Python-level calls) and
library versions. The measured wall is therefore not the code's wall. On a different interpreter or with
`NUMBA_DISABLE_JIT=1`, profiler overhead alone can breach the budget: **false red**. Nothing about the
structural claim ("the 100k cap is in effect", "the binned-MI passes are not quadratic") is being
measured.
**Evidence:** Read all four. `test_charts_prediction_stability.py:274-278` already documents one widening
(5.0 -> 8.0) caused purely by concurrent CI load; `test_preprocessing.py:377-382` documents the same
ratchet (2.0 -> 3.0). That ratchet is the signature of this bug class: the number is being tuned to the
environment rather than to the contract. None of the four imports `perf_time_budget`.
**Suggested fix:** Where the claim is "not quadratic", assert the *scaling exponent*, as
tests/reporting/test_panel_emphasis.py:370-374 already does correctly (median per-doubling ratio across
five sizes) -- that ratio is dimensionless and cancels host speed. Where the claim is "a cap is in
effect" (test_charts_quantile.py:489, "the 100k cap"), assert the cap directly: the number of rows the
decomposition actually consumed. Keep cProfile output for the failure message only, outside the timed
region.

### NT-06 [P2] loky-pool-prewarm-15s-vs-26s-baseline
**File:** tests/feature_selection/test_polynom_loky_pool_prewarm.py:146
**Summary:** `warm_elapsed < 15.0` discriminates a warm loky pool from a cold one, against a stated
"~26-28s cold baseline".
**Failure scenario:** Both arms are process-spawn dominated. Windows process spawn is several times more
expensive than Linux fork; a Linux CI box may have a *cold* dispatch under 15s (**false green** -- the
prewarm could be deleted and the test still passes), while a loaded Windows box may have a *warm*
dispatch over 15s (**false red**). The 26-28s baseline is a single-host measurement, unlike the ratio it
implies.
**Evidence:** Read the test. It correctly `thread.join(timeout=120)`s to guarantee warmth, then throws
that determinism away by asserting on a time. No `perf_time_budget`.
**Suggested fix:** The direct property is that the dispatch reused existing workers rather than spawning
new ones. Capture the worker PIDs from the prewarmed executor and assert the real dispatch ran on the
same PIDs (or assert `executor._max_workers` / the reusable-executor identity is unchanged). That is an
exact, host-independent statement of "the pool was warm".

### NT-07 [P2] mrmr-high-dim-embedding-wall-budgets
**File:** tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_contracts_robustness/test_high_dim_embedding.py:286, :310, :340, :433, :476, :496
**Summary:** Six parametrised wall budgets (60s / 60s / 30s / 300s / 60s / 90s) across three seeds each,
guarding "no super-linear blow-up at embedding scale".
**Failure scenario:** These are full MRMR fits at p=200/p=500. Under xdist they contend with every other
worker; under `NUMBA_DISABLE_JIT=1` they are unrunnable at these budgets. Any of the 15+ parametrisations
can trip: **false red**, and it is the single largest concentration of unguarded wall-clock in the suite.
**Evidence:** Read the file. The docstring at line 426 already acknowledges cross-parametrisation JIT
warm-up variance ("one parametrize value triggers compilation, the rest amortise") -- i.e. the number is
known to depend on execution order, which xdist reshuffles. No helper imported. The *structural*
assertions in the same tests (support size bounded, signal columns recovered) are the valuable ones and
are already present.
**Suggested fix:** The blow-up claim is a scaling claim: measure p=100 and p=200 in the same process and
assert the ratio is sub-quadratic (`t200 / t100 < 3.0`), which cancels host speed. Failing that, wrap the
absolute budgets in `perf_time_budget(...)` -- the helper exists precisely for a coarse "order-of-magnitude
regression" gate -- and keep the recovery/support assertions as the real contract.

### NT-08 [P2] shap-preflight-25s-30s-budgets
**File:** tests/feature_selection/shap_proxied/test_shap_proxy_preflight.py:337, tests/feature_selection/shap_proxied/test_shap_proxy_preflight.py:416
**Summary:** `elapsed < 25.0` / `elapsed < 30.0` are asserted to prove "deep booster `max_depth` is 3" and
"the `n_estimators` cap is in effect" -- the failure messages say so verbatim.
**Failure scenario:** xgboost training time depends on thread count, OpenMP build and CPU. A 2-vCPU
runner can exceed 30s with the cap correctly applied (**false red**); a 32-core box can finish an
*uncapped* booster inside 30s (**false green** for the named contract).
**Evidence:** Read both. The failure strings name the exact structural properties that should be asserted
instead. No helper imported.
**Suggested fix:** Assert the booster's parameters directly -- `max_depth == 3` and `n_estimators == <cap>`
on the model the preflight built (or on the kwargs passed to it, via a monkeypatched constructor).
Both are exactly the property the timer is trying to infer, and both are free.

### NT-09 [P3] shap-preflight-additive-ratio-thresholds
**File:** tests/feature_selection/shap_proxied/test_shap_proxy_preflight.py:69, :73, :302
**Summary:** `additive_ratio > 0.7` for an additive fixture and `< 0.6` for an XOR fixture -- a model-fit
diagnostic used to separate two data regimes.
**Failure scenario:** The ratio comes from a fitted booster; a different xgboost version or thread count
moves it. The gap between the two arms (0.6 vs 0.7) is only 0.1 wide, so a small shift can invert the
classification of the borderline case: **false red**, or **false green** if the diagnostic degrades toward
a constant near 0.65.
**Evidence:** Read the test. Currently the measured values are stated to be comfortably apart, so this is
tight-but-safe rather than actively wrong.
**Suggested fix:** Assert the *ordering* rather than the two absolute cuts: `additive_ratio(additive_fixture)
> additive_ratio(xor_fixture) + margin`, computed in the same process with the same library. A paired
comparison cancels the version/thread dependence that an absolute cut does not.

### NT-10 [P2] speedup-floors-without-perf_speedup_floor
**File:** tests/evaluation/test_bootstrap_fused_binary_bundle.py:162, tests/feature_engineering/test_biz_val_cross_sectional_neighbors.py:170, tests/feature_selection/biz_val/test_biz_val_filters_hermite_fe.py:216, tests/feature_selection/contracts/test_evaluation.py:562, tests/feature_selection/info_theory/test_bulk_shuffle_three_mis.py:259, tests/feature_selection/info_theory/test_gil_release_threading_speedup.py:194, tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_interaction_info_prefilter_speedup.py:141, :355, tests/feature_selection/shap_proxied/test_shap_proxy_cluster_su_fused_setup.py:178, tests/feature_selection/shap_proxied/test_shap_proxy_treeshap.py:222, tests/feature_selection/shap_proxied/test_shap_proxy_treeshap_interactions.py:202, tests/metrics/test_bootstrap_auc_presort.py:116, :221, tests/training/baselines/test_per_group_baseline_polars_native.py:130, tests/training/composite/cache/test_prebin_matrix_cache.py:157, tests/training/composite/test_biz_val_group_aggregate_macro.py:156, tests/training/feature_handling/test_biz_val_ordered_target_encoder_batch.py:51, tests/training/feature_selection/test_mi_y_prebin_speedup.py:130, tests/training/neural/test_ranker_getitems_batched.py:129, tests/training/neural/test_ranknet_loss_optimization.py:124, tests/training/neural/test_ranknet_pair_cache.py:170, tests/training/neural/test_torch_dataset_concurrency.py:376, tests/training/neural/test_weighted_loss_dot.py:171
**Summary:** 23 two-arm speedup-ratio floors (1.05x to 5.0x) asserted with a bare literal.
**Failure scenario:** A ratio is more robust than an absolute time, but a single scheduling stall landing
on one arm inverts it -- conftest.py:342-349 documents exactly this and provides `perf_speedup_floor`,
which relaxes the floor under xdist without abandoning the assertion. None of these 23 use it.
**Direction:** **false red** under `-n` contention. The tightest floors (1.05x at
test_ranknet_loss_optimization.py:124, 1.15x at three sites) are within measurement noise of a small
back-to-back timing and will flake first.
**Evidence:** Read all sites via the AST scan output plus source context for 15 of them. Three sibling
tests in the same suites *do* handle this correctly and show the intended pattern:
test_biz_val_filters_hermite_fe.py:153-158 branches on `running_under_xdist()`,
test_biz_val_filters_permutation.py:108-109 skips under xdist, and
tests/feature_selection/shap_proxied/test_shap_proxy_cluster_su_bitmap.py:231 uses the best available
shape -- paired interleaved trials, asserting *majority of paired wins* plus `median_ratio >= 1.0`, which
cancels load noise instead of tolerating it.
**Suggested fix:** For the ones where a structural cause exists, assert it: the cache tests
(test_prebin_matrix_cache.py, test_ranknet_pair_cache.py, test_evaluation.py:562,
test_biz_value_mrmr_interaction_info_prefilter_speedup.py) should count cache hits/kernel invocations,
not time -- a cache hit is a discrete event. For the genuinely-about-speed ones (treeshap vs shap, GIL
release, bulk prange), adopt the bitmap test's paired-trial + majority-wins shape and wrap the residual
floor in `perf_speedup_floor`.

### NT-11 [P2] gil-release-threading-speedup-1.2x
**File:** tests/feature_selection/info_theory/test_gil_release_threading_speedup.py:194
**Summary:** `speedup > 1.2` with `n_jobs=4` threading is trusted to prove `nogil=True` is still set on
`compute_mi_from_classes`.
**Failure scenario:** On a 2-vCPU CI runner four threads cannot deliver 1.2x regardless of `nogil`:
**false red**, deterministically. Conversely on a free-threaded or heavily-cored build the assertion can
pass through other effects.
**Evidence:** Read the test; no helper, and no guard on available core count.
**Suggested fix:** `nogil` is a compile-time attribute of the dispatcher. Assert it directly on the numba
dispatcher's target options rather than inferring it from a thread-scaling measurement, and gate any
retained scaling check on `psutil.cpu_count(logical=False) >= 4`.

### NT-12 [P3] robust-fourier-axis-50x-spread-ratio
**File:** tests/feature_selection/biz_val/test_biz_val_filters_robust_basis_axis.py:83
**Summary:** `median_ratio >= 50.0` (measured ~312x) separating the robust axis from the "collapsed"
legacy axis.
**Failure scenario:** The legacy denominator is a near-collapsed spread; the ratio is
`robust / max(legacy, 1e-12)`, so it is a ratio of a normal number to a near-zero one. Its magnitude is
governed by how close to zero the legacy spread lands, which is a float-precision-sensitive quantity
(float32 vs float64 accumulation order differs between numba and numpy paths). **False red** if the
legacy path degrades gracefully on another platform; the 6x margin currently makes this unlikely.
**Evidence:** Read lines 74-86. This is not a fit score, it is a computed spread; the margin is large.
**Suggested fix:** The claim is "legacy collapses, robust does not". Assert that directly and
separately: `legacy_spread < small_absolute_bound` AND `robust_spread` within a sane band -- two
statements about two quantities, neither of which is a quotient by an almost-zero.

### NT-13 [P2] gpu-k-chunk-precondition-depends-on-device-vram
**File:** tests/feature_selection/gpu/test_gpu_resident_fe.py:102
**Summary:** `_gpu_k_chunk(100_000) < 384` is a *precondition* asserting the test will exercise more than
one VRAM chunk. The value is derived from the live device's free memory.
**Failure scenario:** On a large-VRAM device (or simply a device with more free memory at that moment,
e.g. no other worktree session holding a context) the chunk size reaches 384 and the assertion fails:
**false red**, on a machine where the code is entirely correct. It also silently stops testing the
chunk-boundary path on such devices even if it were skipped instead.
**Evidence:** Read lines 95-105. A sibling test,
tests/feature_selection/fe/gpu/test_gpu_k_chunk_vram_fraction_ktc.py:47-51, does this correctly by
passing `free_bytes=fb` explicitly.
**Suggested fix:** Force multiple chunks deterministically -- pass an explicit `free_bytes` /
`vram_fraction` (the resolver already accepts both) so the chunk count is a property of the test, not of
the device. Then assert the chunk count is `> 1` rather than asserting a byte-derived size threshold.

### NT-14 [P2] brute-force-n-chunks-depends-on-core-count
**File:** tests/feature_selection/shap_proxied/test_shap_proxy_search.py:96
**Summary:** `_resolve_brute_force_n_chunks() >= 8` -- a hardware-aware default asserted against a literal.
**Failure scenario:** The resolver derives chunk count from core count; on a 2-vCPU CI runner it will
plausibly return fewer than 8: **false red**. The test's actual contract, on the very next line, is that
forcing `n_chunks` in `(8, 16, 32, 64)` produces bit-identical results to the default -- which does not
depend on the default's value at all.
**Evidence:** Read lines 89-99. The bit-identity loop is the real assertion and it is already correct.
**Suggested fix:** Drop the `>= 8` line, or replace it with `>= 1`. The invariance-under-chunking property
is the contract and it travels.

### NT-15 [P1] naive-bayes-log-odds-honest-negative-0.0005-auc-margin
**File:** tests/competition/test_biz_val_naive_bayes_log_odds.py:112, tests/competition/test_biz_val_naive_bayes_log_odds.py:87
**Summary:** `auc_avg - auc_logodds >= 0.0005` (honest-negative arm) and `auc_logodds - auc_avg >= 0.001`
(positive arm). Two AUC values measured at 0.7436 vs 0.7421 and 0.9576 vs 0.9553 are being separated by a
margin of five ten-thousandths.
**Failure scenario:** On n=500 test rows one ROC-AUC unit is 1/(n_pos*n_neg) ~ 1.6e-5, so 0.0005 is
roughly 30 pair-swaps. Any change in scipy/sklearn's tie handling, BLAS summation order, or float32-vs-
float64 accumulation in `predict_proba` moves both AUCs by more than that. **False red** is the likely
outcome; but this is P1 because the *honest-negative* arm can also invert into **false green**: if the
log-odds implementation regresses into something that merely differs from averaging, the assertion
`auc_avg - auc_logodds >= 0.0005` is satisfied by noise, and the test reports that the documented
limitation is still demonstrated when nothing about conditional dependence is actually being shown.
**Evidence:** Read lines 60-115. The docstring at 91-98 states the intent: "log-odds summation
over-multiplies correlated evidence". That mechanism is not what is measured.
**Suggested fix:** Assert the mechanism directly. Under conditional dependence, log-odds summation
double-counts correlated evidence, which shows up as *over-confident* probabilities: assert the
log-odds arm's mean predicted probability is further from the empirical positive rate than the averaging
arm's (a calibration statement), or that the log-odds score distribution is more dispersed. Both are
first-order effects with a large margin, unlike a 0.0005 AUC delta. Raise `n_test` at minimum -- 500 rows
cannot resolve 0.0005.

### NT-16 [P1] gmm-honest-negative-0.005-auc-margin-at-ceiling
**File:** tests/competition/test_biz_val_gmm_classifier.py:138
**Summary:** `auc_gbm - auc_gmm >= 0.005` where the measured values are 1.000 and 0.990 -- i.e. the
comparison happens at the AUC ceiling.
**Failure scenario:** `GradientBoostingClassifier` is at 1.000; the entire margin lives in the GMM arm's
0.990. A sklearn version bump, a different `make_classification` internal, or a change in the GMM
initialisation moves that. **False green** is the real hazard: at a ceiling, "GBM beats GMM by 0.005" is
satisfied whenever GMM is merely slightly imperfect, including if `GaussianMixtureClassifier` regressed
into something unrelated to a Gaussian mixture. The honest-negative claim ("the trick only wins on true
GMM data") is not tested by that.
**Evidence:** Read lines 100-142. The sibling positive test at :115-116 uses 0.03/0.05 margins, an order
of magnitude larger, on the same fixture family.
**Suggested fix:** The claim is about *which data-generating process favours which model*. Assert the
interaction, not one cell of it: `(auc_gmm - auc_gbm)` on the true-mixture fixture must exceed
`(auc_gmm - auc_gbm)` on the non-mixture fixture by a wide margin. A difference-in-differences is what
"the trick is narrow" means, and it is robust to both arms drifting.

### NT-17 [P1] conditional-gate-e2e-auc-0.999-and-delta-above-zero
**File:** tests/feature_selection/biz_val/test_biz_val_e2e_operator_model_lift.py:158
**Summary:** `auc_on >= 0.999` plus `delta > 0.0` on a held-out LGBM fit, guarding that the
conditional-gate operator carries the regime-switch signal.
**Failure scenario:** The docstring itself records that OFF measures 0.9976 at seed 42 -- a ceiling
effect -- so `delta > 0.0` is separating two numbers roughly 0.0014 apart. LightGBM's histogram
construction and thread count both perturb that. **False red** on a different LightGBM build; **false
green** because `delta > 0.0` can be satisfied by fit noise while the gate feature contributes nothing.
**Evidence:** Read lines 146-162. The test *already* contains the direct assertion on the line above:
`any("gate_" in n for n in names_on)` -- selection, not fit score. That is exactly the reframing the
0.94-AUC case in this audit's brief was fixed with.
**Suggested fix:** Keep the selection assertion as the contract and demote `delta > 0.0` to a printed
diagnostic, or replace it with a selection-side statement (the gate composite outranks every raw operand
in the selector's own score ordering). `auc_on >= 0.999` can stay as a coarse sanity floor.

### NT-18 [P2] adversarial-auc-0.6-0.7-band-on-identical-splits
**File:** tests/reporting/test_diagnostics_dispatch.py:283, tests/reporting/test_diagnostics_dispatch.py:286
**Summary:** `auc_same <= 0.6` proves two identically-distributed samples are indistinguishable;
`auc_shift >= 0.7` proves a mean-shifted feature is detectable.
**Failure scenario:** `auc_same` is a null statistic whose sampling distribution around 0.5 depends on the
adversarial model's capacity and its CV fold split. At n=4000 with an overfitting-prone model, a
different sklearn default (e.g. a changed `n_estimators` or a different fold shuffle) can push a null AUC
past 0.6: **false red**. Less likely but possible: a broken `adversarial_auc` returning a constant 0.5
passes `auc_same <= 0.6` trivially -- but then fails the shift arm, so the pair is partially
self-protecting.
**Evidence:** Read lines 273-287. The comment states measured ~0.5 / ~0.95, so margins are currently wide;
this is closer to P3 than the AUC-margin findings above, but the 0.6 cut on a null statistic has no
distributional justification.
**Suggested fix:** Derive the null band from the data rather than pinning it: compute `adversarial_auc`
on several label-permuted or independently-resampled identical pairs and assert the real `auc_same` sits
inside that empirical null's range, while `auc_shift` sits outside it. That is a permutation test, which
travels exactly.

### NT-19 [P2] pure-noise-transform-score-band-0.3-0.7
**File:** tests/preprocessing/test_auto_transform_select_fold_leakage.py:81, tests/preprocessing/test_auto_transform_select_fold_leakage.py:106
**Summary:** every transform's CV score on a pure-noise column must land in `[0.3, 0.7]`, used as a proxy
for "no fold saw its own test rows' statistics".
**Failure scenario:** n=400 with `n_splits=4` means 100 rows per held-out fold; a noise-column AUC on 100
rows has a standard deviation around 0.06, so a 4-fold aggregate straying outside [0.3, 0.7] is not
remarkable under a different `random_state` interpretation or a changed CV shuffle: **false red**. The
**false green** is more serious and the test's own comment concedes it: a genuine leak on a 400-row
noise column may only lift the score to ~0.72 at the extreme fold, and a mild leak stays inside the band
entirely.
**Evidence:** Read lines 65-107. The comment at lines 69-72 explicitly says "an exact leak-magnitude
assertion would be fixture-fragile; a bounded max-AUC check is a robust proxy" -- the proxy was chosen
knowingly.
**Suggested fix:** Assert fold-locality structurally, which is cheap here: monkeypatch the scaler/imputer
fit entry point and record the row indices it was fitted on for each fold, then assert that set is
disjoint from that fold's test indices. That is the contract verbatim, with no statistical margin at all.
The same technique closes the imputation-median case at line 106, where the leaked statistic is a single
median and the noise band is even less able to see it.

### NT-20 [P3] cross-product-lift-bounded-by-0.02-auc
**File:** tests/feature_selection/biz_val/test_biz_val_stratified_subsample.py:317
**Summary:** `auc_with_cross - auc_no_cross <= 0.02` is used to establish that "the pairs subsample is not
the end-to-end lever".
**Failure scenario:** `auc_no_cross` is already >= 0.95, so the headroom above it is under 0.05 -- the
0.02 bound occupies most of the remaining range and is therefore nearly unfalsifiable: **false green**
by construction. In the other direction a change to `LogisticRegression`'s solver/`max_iter` convergence
on the 5-feature arm could add 0.02: **false red**.
**Evidence:** Read lines 295-320. Both arms are the same LogisticRegression on nested feature sets.
**Suggested fix:** Compare against the ceiling rather than an absolute delta: assert the cross term
closes less than some fraction of the *available* gap, `(auc_with - auc_no) / (1.0 - auc_no) <= 0.4`.
That normalises out the fit quality of the base arm, which is the part that does not travel.

### NT-21 [P3] duration-only-auc-below-0.55-for-a-constant-feature
**File:** tests/feature_engineering/test_biz_val_state_duration.py:203
**Summary:** `auc_duration_only < 0.55` proves a constant single feature cannot separate the classes.
**Failure scenario:** If the feature is truly constant, the AUC is *exactly* 0.5 by construction -- the
model degenerates to the class prior, as the comment at lines 193-195 states. Asserting `< 0.55` on a
quantity that is analytically 0.5 is harmless but hides the real claim; and if the feature stops being
constant (the actual regression this guards) the AUC could still land at 0.54 and pass: **false green**.
**Evidence:** Read lines 190-208.
**Suggested fix:** Assert the structural property named in the comment: `np.unique(cancellation_duration).size == 1`
(the feature is constant), and that the predicted probabilities are all equal. Both are exact.

### NT-22 [P3] causal-rank-auc-band-0.65-to-0.90
**File:** tests/feature_engineering/test_biz_val_per_group_rank_causal.py:63
**Summary:** `0.65 <= auc_causal <= 0.90` -- a two-sided band on a fit score used to prove the causal
variant neither leaks (upper bound) nor is useless (lower bound).
**Failure scenario:** The upper bound is the leakage guard and it is only 0.02 below the "plain" arm's
0.92 floor asserted on the line above; a small change to the tie-handling in `per_group_rank` moves
`auc_causal` past 0.90: **false red**. The band is derived from one seed (11), stated in the comment as
"5-15% below/above the measured values".
**Evidence:** Read lines 47-65. The third assertion (`auc_plain - auc_causal >= 0.08`) is the paired one
and is the better shape -- it is already present.
**Suggested fix:** Keep the paired gap assertion, drop the absolute upper bound. Then add the direct
no-leak property: for a given row, `rank_causal` must be computable from strictly-prior rows only --
assert that mutating a *later* row within the group leaves that row's causal rank unchanged. That is an
exact invariance test of the causal property and needs no fit at all.

### NT-23 [P2] sleep-based-registry-clock-assertions
**File:** tests/training/test_phase_summary_accounting.py:68, tests/training/test_phase_summary_accounting.py:53, tests/training/test_phase_summary_accounting.py:61
**Summary:** `registry_elapsed() < 0.03` after `reset_phase_registry()` -- and the two sibling
lower-bound checks after `time.sleep(0.02)` / `sleep(0.03)`.
**Failure scenario:** Line 68 asserts that less than 30ms of wall time elapses between a
`reset_phase_registry()` call and the immediately-following `registry_elapsed()` call. Windows'
default timer granularity is ~15.6ms and a single scheduler preemption on a contended box (this suite
runs under xdist on a shared machine) trivially exceeds 30ms: **false red**. The two lower-bound
assertions are safe (`time.sleep` guarantees at least the requested duration).
**Evidence:** Read lines 46-68. The contract being tested -- "a later reset restarts the clock" -- is not
about 30ms.
**Suggested fix:** Assert monotonic ordering instead of an absolute bound: capture `e1 = registry_elapsed()`
before the second reset and `e2` after, and assert `e2 < e1`. That is the restart property exactly, and
it holds at any host speed.

### NT-24 [P2] kaleido-recovery-90s-and-15s-budgets
**File:** tests/reporting/test_kaleido_recovery.py:79, tests/reporting/test_kaleido_recovery.py:138
**Summary:** `elapsed < 90.0` proves the oneshot fallback fired rather than deadlocking; `elapsed < 15.0`
proves a later save reused the restarted persistent server rather than falling back to oneshot.
**Failure scenario:** Both numbers are Chromium process-spawn times, the least portable quantity in the
suite: the comments themselves list "~30-40s cold, ~12-15s warm, ~8s cold-restart, ~13s oneshot" -- four
overlapping ranges on one host. Line 138's 15s bound cannot separate "cold restart of the persistent
server" (~8s, the pass case) from "oneshot" (~13s, the fail case) on a machine 1.5x slower: **false
green** for the exact regression named in its own failure message, and **false red** on a slow or
cold-disk CI box.
**Evidence:** Read lines 60-141. The bound at line 79 is a genuine hang guard (fine in principle, though
it should use `perf_time_budget`); the one at 138 is being asked to discriminate two code paths.
**Suggested fix:** For line 138, assert the path directly -- the module already tracks oneshot statistics
(`get_kaleido_oneshot_stats()`, exercised at
tests/reporting/test_plotly_kaleido_module_split_inv57.py:46): reset the counter after recovery and
assert the oneshot call count is **0** for the second save. That is an exact statement of "did not fall
back to oneshot".

### NT-25 [P2] cached-call-overhead-microbudgets
**File:** tests/training/neural/test_lightning_callback_cache.py:37, tests/training/test_caching_pipeline_cache_obs.py:68, tests/training/test_preprocessing_fastpath_bench.py:64, tests/training/test_schema_drift_perf.py:56, tests/training/feature_selection/test_mrmr_identity_cache_and_monres_autoknot.py:144
**Summary:** Five sub-100ms budgets (`elapsed_warm < 0.05`, `elapsed/n < 1e-5`, `elapsed < 0.05`,
`elapsed_ms < 100.0`, `elapsed < 0.5`) each standing in for "the cache / short-circuit / fast path is in
effect".
**Failure scenario:** These are the tightest absolute budgets in the suite. 50ms for 1000 calls is 50us
per call; under `coverage.py` (the nightly job traces every line, including with JIT disabled) or under
a debugger or on a contended worker, per-call Python overhead multiplies severalfold: **false red**,
predictably, on the coverage job. Because they measure aggregate loops of trivial work, they are also
the least likely to catch a real regression that is not order-of-magnitude.
**Evidence:** Read all five. `test_preprocessing_fastpath_bench.py:64-68` names the exact structural
cause it is guarding: "someone likely added work before the `if config is None: return ...`
short-circuit". None uses `perf_time_budget`.
**Suggested fix:** All five have an exact alternative. For the short-circuit
(test_preprocessing_fastpath_bench.py) and the identity cache
(test_mrmr_identity_cache_and_monres_autoknot.py), monkeypatch the first function *past* the
short-circuit and assert it is never called. For the callback cache and the pipeline cache, assert the
underlying loader is invoked exactly once across N calls. For `_warn_on_schema_drift`, spy on the pandas
comparison path the comment names and assert it is not entered. Call counts are exact and free.

### NT-26 [P3] fuzz-isolate-runner-reap-bounds
**File:** tests/training/fuzz/test_fuzz_isolate_runner.py:75, tests/training/fuzz/test_fuzz_isolate_runner.py:123
**Summary:** `elapsed < 5.0` proves `_reap_bounded(p, 1.0)` respected its 1s bound; `elapsed < 45.0`
proves `_run_one_combo` did not wedge on a pipe-holding child.
**Failure scenario:** These are subprocess-lifecycle bounds with 5x and 22x headroom over the parameter
they guard. Windows process teardown under memory pressure can be slow, but the margins make this
unlikely: currently safe. The 5.0s bound would not distinguish a bound of 1s from a bound of 4s, so a
regression that widens `_reap_bounded`'s effective timeout would pass: **false green** for a narrow
class.
**Evidence:** Read lines 66-125.
**Suggested fix:** Bound relative to the parameter: `elapsed < timeout_arg * 3`. It keeps the same
portability slack while actually tracking the value under test.

### NT-27 [P3] stress-suite-5s-budgets-on-unseeded-fixtures
**File:** tests/training/test_stress.py:213, tests/training/test_stress.py:248, tests/training/test_stress.py:249, tests/training/test_utils.py:549, tests/training/test_feature_selection.py:741, tests/training/test_memory_usage_polars_fastpath.py:96, tests/training/composite/cache/test_composite_update_ring_buffer.py:208, tests/training/core/test_training_core_a_fixes.py:310, tests/training/neural/test_neural_high_severity_regressions.py:357, tests/training/neural/test_neural_medium_severity_regressions.py:246, tests/feature_selection/wrappers/test_wrappers_invariants.py:450, tests/feature_selection/filters/test_qs_mah_edge_stress.py:200, tests/reporting/test_renderers_vocabulary.py:201, tests/preprocessing/test_preprocessing.py:233, tests/preprocessing/test_preprocessing.py:383, tests/training/test_confidence_analysis_fixes.py:303, tests/training/test_regression_drift_psi_array_cells.py:93, tests/data_valuation/test_biz_val_training_weight_adapter.py:103, tests/feature_selection/mrmr/core/test_mrmr_sis_screen.py:259, tests/feature_selection/mrmr/biz_val/test_biz_value_mrmr_regression_union/test_regression_diff_vs_l52.py:217
**Summary:** Twenty remaining absolute wall budgets (0.5s to 180s) used as coarse "did not become
pathologically slow" sensors, none wrapped in `perf_time_budget`.
**Failure scenario:** **False red** under xdist, host contention, or `NUMBA_DISABLE_JIT=1`. Several
already carry comments recording a past widening forced by exactly that
(test_training_overhead_integration_fixes.py:265-269 documents 10.0 -> 20.0 after CI hit 10.27s;
test_preprocessing.py:377-382 documents 2.0 -> 3.0; test_charts_prediction_stability.py:274-278
documents 5.0 -> 8.0). Each widening trades away sensitivity permanently.
**Evidence:** AST scan plus source reading of 14 of the 20. `tests/training/test_stress.py` additionally
builds its fixtures from the *unseeded global* numpy RNG (29 unseeded `np.random.*` calls in that file),
so the workload itself varies run to run -- the timing and the data both fail to travel.
**Suggested fix:** These are legitimately coarse regression sensors and the right fix is mechanical, not
architectural: wrap each base value in `perf_time_budget(...)`. The helper preserves the tight budget on a
quiet box and widens it exactly where the measurement stops being meaningful, which is what every one of
the hand-written widenings above was trying to approximate. Separately, seed the fixtures in
test_stress.py.

### NT-28 [P3] percentage-speedup-floor-on-early-stopping
**File:** tests/training/test_bizvalue_outliers_earlystop.py:374
**Summary:** `speedup_pct >= 15.0` -- a percentage wall-clock saving from early stopping.
**Failure scenario:** Early stopping's saving is a function of how many iterations are skipped, which is
a discrete, host-independent quantity; expressing it as a percentage of wall time reintroduces host
dependence for no gain. **False red** under contention.
**Evidence:** Flagged by the AST scan; the file imports no timing helper.
**Suggested fix:** Assert the iteration count directly (`best_iteration_` / `n_iterations_` on the
stopped arm vs the unstopped arm). The suite already asserts iteration counts this way elsewhere
(tests/training/test_gate_return_deadline_and_dead_param.py:122,
tests/training/neural/test_history_recorder.py:33).

---

## Summary table

| ID | Sev | Slug | Files | Direct property available? |
|---|---|---|---|---|
| NT-01 | P2 | prewarm-cache-budget-dies-under-NUMBA_DISABLE_JIT | test_prewarm.py | Yes -- dispatcher `.signatures` table |
| NT-02 | P2 | brier-logloss-prewarm-50ms-proxy | test_prewarm_bool_dtype.py | Yes -- dispatcher `.signatures` table |
| NT-03 | P2 | gpu-shared-fused-kernel-2s-budget | test_batch_pair_mi_shared_fused.py | Yes -- launch geometry via `launch_config` |
| NT-04 | P2 | cmim-hotpath-5s-budget | test_cmim_hotpath_perf.py | Yes -- cache hit / call count |
| NT-05 | P2 | cprofile-wrapped-wall-budgets | 4 files | Yes -- scaling exponent, or the cap value itself |
| NT-06 | P2 | loky-pool-prewarm-15s-vs-26s-baseline | test_polynom_loky_pool_prewarm.py | Yes -- worker PID identity |
| NT-07 | P2 | mrmr-high-dim-embedding-wall-budgets | test_high_dim_embedding.py (6 sites) | Partly -- scaling ratio; else `perf_time_budget` |
| NT-08 | P2 | shap-preflight-25s-30s-budgets | test_shap_proxy_preflight.py | Yes -- booster `max_depth` / `n_estimators` |
| NT-09 | P3 | shap-preflight-additive-ratio-thresholds | test_shap_proxy_preflight.py | Yes -- paired ordering instead of two cuts |
| NT-10 | P2 | speedup-floors-without-perf_speedup_floor | 23 sites, 21 files | Partly -- cache/call counts; else paired trials |
| NT-11 | P2 | gil-release-threading-speedup-1.2x | test_gil_release_threading_speedup.py | Yes -- `nogil` dispatcher attribute |
| NT-12 | P3 | robust-fourier-axis-50x-spread-ratio | test_biz_val_filters_robust_basis_axis.py | Yes -- two separate spread assertions |
| NT-13 | P2 | gpu-k-chunk-precondition-depends-on-device-vram | test_gpu_resident_fe.py | Yes -- explicit `free_bytes`, assert chunk count |
| NT-14 | P2 | brute-force-n-chunks-depends-on-core-count | test_shap_proxy_search.py | Yes -- drop it; invariance loop is the contract |
| NT-15 | **P1** | naive-bayes-log-odds-honest-negative-0.0005-auc-margin | test_biz_val_naive_bayes_log_odds.py | Yes -- over-confidence / calibration statement |
| NT-16 | **P1** | gmm-honest-negative-0.005-auc-margin-at-ceiling | test_biz_val_gmm_classifier.py | Yes -- difference-in-differences across fixtures |
| NT-17 | **P1** | conditional-gate-e2e-auc-0.999-and-delta-above-zero | test_biz_val_e2e_operator_model_lift.py | Yes -- selection assertion already on the line above |
| NT-18 | P2 | adversarial-auc-0.6-0.7-band-on-identical-splits | test_diagnostics_dispatch.py | Yes -- empirical permutation null |
| NT-19 | P2 | pure-noise-transform-score-band-0.3-0.7 | test_auto_transform_select_fold_leakage.py | Yes -- fold-index disjointness via spy |
| NT-20 | P3 | cross-product-lift-bounded-by-0.02-auc | test_biz_val_stratified_subsample.py | Yes -- fraction of available headroom |
| NT-21 | P3 | duration-only-auc-below-0.55-for-a-constant-feature | test_biz_val_state_duration.py | Yes -- `np.unique(...).size == 1` |
| NT-22 | P3 | causal-rank-auc-band-0.65-to-0.90 | test_biz_val_per_group_rank_causal.py | Yes -- future-row mutation invariance |
| NT-23 | P2 | sleep-based-registry-clock-assertions | test_phase_summary_accounting.py | Yes -- monotonic ordering across the reset |
| NT-24 | P2 | kaleido-recovery-90s-and-15s-budgets | test_kaleido_recovery.py | Yes -- `get_kaleido_oneshot_stats()` count == 0 |
| NT-25 | P2 | cached-call-overhead-microbudgets | 5 files | Yes -- call counts on the guarded entry point |
| NT-26 | P3 | fuzz-isolate-runner-reap-bounds | test_fuzz_isolate_runner.py | Partly -- bound relative to the timeout arg |
| NT-27 | P3 | stress-suite-5s-budgets-on-unseeded-fixtures | 20 sites, 20 files | No -- coarse sensors; use `perf_time_budget` + seed |
| NT-28 | P3 | percentage-speedup-floor-on-early-stopping | test_bizvalue_outliers_earlystop.py | Yes -- iteration counts |

**Totals:** 3 P1, 15 P2, 10 P3 across roughly 70 individual assertion sites.

**The one structural recommendation.** Three of the 28 findings (NT-15/16/17) are false-greens caused by
a model-fit score standing in for a structural claim, and every one of them has the structural assertion
either already present in the same test or available for free. The other 25 are dominated by a single
mechanical gap: 280 of 281 wall-clock and speedup comparison sites do not use the widening helpers that
`tests/conftest.py` provides. A `flake8`/`ruff`-style custom check -- "an `assert` whose comparison
mentions `elapsed`/`wall`/`speedup` must have a `perf_time_budget(`/`perf_speedup_floor(` call in its
expression" -- would ratchet this shut and stop the pattern of hand-widening individual literals after
each CI failure, which is visible in at least four files' comments and costs sensitivity every time.
