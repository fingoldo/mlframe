# Edge-Case Proposals

1 findings, 53 proposals.

## Findings

### [P2] bug -- src/mlframe/feature_selection/filters/friend_graph.py:287

**_inf_fs_centrality builds the adjacency matrix from `edges` by direct assignment `A[i,j]=A[j,i]=float(e.mi)`; if `edges` ever contains a duplicate (a,b) pair (e.g. a future caller that unions two edge lists before calling build_friend_graph), the later edge silently overwrites the earlier one instead of raising or summing/max-ing, with no test pinning which behavior is intended.**

A future refactor that merges DCD-cluster edges and ordinary MI edges into one `edges` list before centrality could pass through a duplicate (a,b) with two different `mi` values (e.g. stale vs fresh score); centrality silently uses whichever happened to be appended last, changing `low_centrality` membership with no error and no regression coverage.

## Proposals

### (edge_case) compute_selection_gate: seed=101 regression pinned directly

File: src/mlframe/feature_selection/filters/_screen_predictors_gate.py, compute_selection_gate(). Build a synthetic predictors list mirroring the inline comment's documented case: first-selected feature has raw gain 0.328 that Miller-Madow-corrects down to 0.088 (high-cardinality user_id-like column, e.g. factors_nbins=500), second candidate has raw gain 0.187 correcting to 0.185 (low-cardinality genuine signal, e.g. nbins=10). Call compute_selection_gate with min_relevance_gain_relative_to_first=0.05 and assert the relative floor computed is based on the RUNNING MAX (0.185*0.05=0.00925), not the first-selected value (0.088*0.05=0.0044) -- i.e. construct a third candidate with corrected gain 0.005 that must be REJECTED under the max-based floor but would PASS under a first-based floor. Directly targets the exact bug class the module's own comment (lines 126-136) warns a future edit could reintroduce; currently only reachable via a ~600-test e2e suite that may never hit this exact numeric configuration.

### (edge_case) compute_selection_gate: MM-bias sign flip regression

File: _screen_predictors_gate.py, lines 113-125. Unit-test compute_selection_gate directly (bypassing MRMR.fit) with cardinality_bias_correction=True, a single-column best_candidate=(3,), factors_nbins=[..., 50 (for col 3), ..., 4 (for y)], n=1000, best_gain=0.10. Assert _best_gain_for_gate == best_gain - (50-1)*(4-1)/(2*1000) exactly (bias SUBTRACTED, not added). A sign flip (+ instead of -) would silently make high-cardinality columns MORE likely to pass rather than less -- the opposite of the correction's purpose -- and nothing currently isolates the sign from the aggregate e2e selection outcome.

### (edge_case) compute_selection_gate: maxT floor uses cached marginal MI, not conditional gain

File: _screen_predictors_gate.py, lines 157-184. Directly test compute_selection_gate with fdr_gain_floor=0.05, interactions_order=1, best_candidate=(7,), cached_MIs={(7,): 0.02} (marginal MI BELOW the floor) but best_gain=0.20 (conditional/Fleuret gain ABOVE the floor, simulating the exact 'noise inflates under deep conditioning' scenario the comment describes). Assert the gate REJECTS (fdr_pass False) because it must read cached_MIs[(7,)]=0.02, not best_gain=0.20. This is the precise regression the inline comment (lines 157-172) warns a future edit could break by flooring on `_best_gain_for_gate` instead of `cached_MIs`.

### (edge_case) compute_selection_gate: joint (order>=2) candidates skip MM correction entirely

File: _screen_predictors_gate.py, lines 106-113, 173. Call with interactions_order=2, cardinality_bias_correction=True, best_candidate=(2,5) with high per-column nbins (39,39 as cited in the comment). Assert _best_gain_for_gate == best_gain unchanged (no MM subtraction applied) and _fdr_floor_eff == 0.0 (FDR floor is a no-op for joints). Pins the documented 'explicit MM correction on joints is double-counting' design decision against an accidental future generalization of the order==1 branch to order>=2.

### (edge_case) warn_accuracy_suboptimal_params: fires exactly once per estimator instance

File: src/mlframe/feature_selection/filters/_param_accuracy_warnings.py, warn_accuracy_suboptimal_params(). Construct a stub object with min_features_fallback=0 (a registered bad value). Call warn_accuracy_suboptimal_params(stub) twice inside pytest.warns(UserWarning) context and assert exactly ONE warning is captured across both calls (second call is a no-op because `_accuracy_caveats_warned_` is now True). Directly targets the fire-once guard described in the coverage gap; a broken guard (e.g. reset on each fit, or never set) would spam warnings on repeated .fit() calls or never re-arm across separate estimator instances -- untested today.

### (edge_case) warn_accuracy_suboptimal_params: silent on default config

File: _param_accuracy_warnings.py. Build a stub object with every ACCURACY_SUBOPTIMAL attr set to its documented GOOD default (fe_discrete_structural_operators_enable=True, dcd_enable=True, fe_accuracy_gate=True, min_features_fallback=1, quantization_nbins=10, fe_confirm_undersample_rows_per_cell=5.0, etc). Assert warn_accuracy_suboptimal_params(stub) with pytest.warns(None)-equivalent (recwarn) captures ZERO UserWarnings. Regression-pins 'default-config fit is silent' -- a predicate typo (e.g. `_eq(True)` instead of `_eq(False)`) would flip this and every real-world default fit would start emitting spurious warnings, which nothing catches today.

### (edge_case) warn_accuracy_suboptimal_params: quantization_nbins boundary at exactly 5

File: _param_accuracy_warnings.py, line 91: `lambda v: isinstance(v, int) and v < 5`. Parametrize stub.quantization_nbins over {3, 4, 5, 6} and assert the warning fires for {3,4} and NOT for {5,6} -- pins the fencepost (`< 5`, not `<= 5`) in the lambda; also test quantization_nbins=5.0 (float, not int) to confirm isinstance(v,int) guard means a float 4.0 does NOT trigger the caveat (a real footgun: a user passing quantization_nbins=4.0 gets no warning despite the same bad discretization).

### (edge_case) warn_accuracy_suboptimal_params: missing attribute is skipped, not raised

File: _param_accuracy_warnings.py, lines 106-113 (`if not hasattr` / `except Exception: continue`). Construct a stub object missing 3 of the registered attrs (e.g. no `dcd_enable`) and with one attr whose `getattr` raises (a property that throws). Call warn_accuracy_suboptimal_params(stub) and assert it does not raise, and still correctly warns about the OTHER triggered caveats present on the object. Pins the 'never raises' contract explicitly promised in the docstring; a future edit tightening the except clause could turn a partially-stubbed test double (common in the MRMR test suite) into a hard crash at fit time.

### (edge_case) fe_deadline: cleared deadline does not leak into next fit on same thread

File: src/mlframe/feature_selection/filters/_fe_deadline.py. In one test: set_fe_deadline(timer()-1) (already expired), assert fe_deadline_passed() is True, then call clear_fe_deadline(), then assert fe_deadline_passed() is False and fe_budget_active() is False -- simulating two successive MRMR.fit() calls on the SAME thread where the first used a budget and the second did not. Directly regression-pins the 'never leak into the next fit' contract stated in clear_fe_deadline's own docstring, called out by name in the coverage gap as untested.

### (edge_case) fe_deadline: thread-local does NOT cross threads (documents current limitation)

File: _fe_deadline.py. In the main thread call set_fe_deadline(timer()+0.001); sleep past it; spawn a `threading.Thread` that calls fe_deadline_passed() and stores the result; assert the WORKER thread's fe_deadline_passed() returns False (deadline invisible) even though the main thread's own fe_deadline_passed() now returns True. This is the exact 'threading.local does not propagate to worker threads' property the module's docstring calls load-bearing; a test asserting current (correct) isolation also acts as a tripwire if someone later moves an enrichment generator into a joblib worker without forwarding the deadline explicitly, per the docstring's own warning.

### (edge_case) fe_deadline: no-deadline path never gates (regression for the common case)

File: _fe_deadline.py. Without calling set_fe_deadline at all (or after clear_fe_deadline()), assert fe_deadline_passed() is False and fe_budget_active() is False. Trivial-looking but load-bearing: fe_budget_active()'s comment says it must stay False on 'the common no-budget path' so KTC sweeps still run per-host; a bug that defaults `_state.deadline` truthy (e.g. via getattr default swap) would silently disable per-host GPU/CPU crossover tuning on every normal fit.

### (edge_case) fe_deadline: negative / already-past deadline set directly

File: _fe_deadline.py. set_fe_deadline(0.0) and set_fe_deadline(-1.0) (values in the past relative to any timer() reading) -- assert fe_deadline_passed() is immediately True on the next call, i.e. an already-elapsed budget aborts on the very first check rather than requiring a full monotonic-clock wraparound or being misread as 'unset' via a falsy-value bug (0.0 is falsy in Python; a bug written as `if dl:` instead of `if dl is not None:` would silently treat a 0.0 deadline as disabled).

### (edge_case) renyi_alpha: alpha exactly 1.0 does not raise ZeroDivisionError

File: src/mlframe/feature_selection/filters/_renyi_alpha.py, _renyi_entropy_from_gram() line 109: `np.log2(s) / (1.0 - alpha)`. Call renyi_alpha_mi(x, y, alpha=1.0) and assert it either raises a clear, documented error OR returns inf/nan that the caller can detect -- currently a bare division by (1.0-1.0)=0.0 with floats produces inf silently (no ZeroDivisionError for floats), which would propagate a silent inf into MRMR's gain computation if a caller ever passes alpha=1.0 despite the docstring's 'close to 1, not exactly 1' guidance. No test currently pins behavior at the documented singularity.

### (edge_case) renyi_alpha: constant column (zero variance) does not NaN the Gram matrix

File: _renyi_alpha.py, _silverman_sigma() lines 66-73 and _rbf_gram(). Call renyi_alpha_mi(x=np.full(200, 5.0), y=rng.normal(size=200)) (X is fully constant). _silverman_sigma returns 1.0 via the `std<=0.0` guard, so the Gram matrix should degrade to all-ones (K=exp(0)=1 for every pair since d2=0 everywhere), giving S_x≈0 entropy and MI≈0 -- assert the result is finite, >=0, and near 0 rather than NaN/inf. This is exactly the 'single unique value in X' degenerate-input class called out in the task brief, applied to the newly-added 2026-07 estimator which has no such test today.

### (edge_case) renyi_alpha: NaN/Inf in input raises or is guarded, not silently propagated

File: _renyi_alpha.py, renyi_alpha_mi / _rbf_gram. Call renyi_alpha_mi(x=np.array([1.0,2.0,np.nan,4.0,...]), y=rng.normal(size=n)) and assert either a clear ValueError is raised OR the function documents/guards NaN propagation -- currently `_as_2d` does `np.asarray(x, dtype=np.float64)` with no `np.isfinite` check, so a NaN silently poisons the squared-distance matrix (`sq[:,None]+sq[None,:]-2*x@x.T` becomes NaN in an entire row/column), then the eigendecomposition and entropy silently return NaN with no diagnostic -- exactly the 'NaN/inf not guarded before a kernel that assumes finite input' pattern called out in the audit checklist.

### (edge_case) renyi_alpha_cmi: multivariate z (n,k) with k>1 conditioning set

File: _renyi_alpha.py, renyi_alpha_cmi(). Call with z of shape (n,3) (three already-selected MRMR variables stacked), verifying the Hadamard-product multivariate extension the module's docstring specifically advertises as the reason conditioning on several variables is cheap. Assert the result is finite, >=0, and that I(X;Y|Z1,Z2,Z3) collapses toward the independence floor when Z fully determines Y (e.g. Z's first column IS y). No existing test exercises z.ndim==2 with k>1; the existing suite's z is always a single column.

### (edge_case) renyi_alpha: n=1 and n=2 degenerate sample sizes

File: _renyi_alpha.py. renyi_alpha_mi with x,y of length 1 and length 2. At n=1, the (1,1) Gram matrix K=[[1.0]], A=[[1.0]], eigvalsh gives eigval 1.0 > 1e-12, entropy = log2(1)/(1-alpha) = 0/(1-alpha) = 0.0 -- assert this returns exactly 0.0, finite, not NaN. At n=2, verify no crash in the eigendecomposition or the max_n subsampling path (n<=max_n so no-op) and result is finite. Directly the 'n=1, n=2' degenerate-input class from the task brief, unexercised for this estimator.

### (edge_case) renyi_alpha: max_n subsampling determinism across repeated calls with same random_state

File: _renyi_alpha.py, _maybe_subsample() lines 112-119. Call renyi_alpha_mi(x, y, max_n=100, random_state=42) twice on the same n=500 arrays and assert byte-identical results (the rng.choice draw is deterministic per random_state) -- then call with random_state=43 and assert a (generally) different result, confirming random_state actually threads through to the subsample rather than being ignored. Also test random_state left at its default 0 across two separate MRMR-style callers to ensure two features scored via renyi_alpha in the same fit get the SAME subsample row indices (needed for cross-feature CMI comparability), not independently-drawn subsamples that would make gains incomparable.

### (edge_case) Inf-FS centrality (_inf_fs_centrality): single selected feature and empty edges

File: src/mlframe/feature_selection/filters/friend_graph.py, _inf_fs_centrality() lines 267-298. Call with sel=[5] (n=1) and sel=[5,6] with edges=[] (n=2, no edges) -- both must return {} per the documented 'Returns an empty dict for <2 nodes or an all-zero adjacency' contract (line 278). Then call with sel=[5,6], edges=[FriendGraphEdge(5,6,mi=0.0)] (an explicit zero-weight edge, distinct from 'no edges') and assert lambda_max<=0.0 path returns {5:0.0, 6:0.0} rather than {} -- pins the distinction between the two different degenerate returns (empty dict vs all-zero dict) which downstream `centrality.get(i,0.0)` treats identically but `if centrality:` at call site (line 555) treats DIFFERENTLY (empty dict is falsy, skipping the percentile cutoff entirely; the all-zero dict is truthy and computes a percentile of all-zeros).

### (edge_case) Inf-FS centrality: (I - alpha*A) invertibility on a near-degenerate adjacency

File: friend_graph.py, _inf_fs_centrality() line 292: `np.linalg.inv(np.eye(n) - alpha*A)`. Construct an adjacency where two nodes have IDENTICAL edge weights to every other node (a symmetric, rank-deficient-looking A) and assert np.linalg.inv does not raise LinAlgError -- alpha=0.5/lambda_max mathematically guarantees invertibility (spectral radius of alpha*A is exactly 0.5), but this is proven only in the docstring's prose, never asserted by a test; a future edit changing the 0.5 safety margin constant (e.g. tightening it to 0.9 'for a stronger centrality signal') could silently produce a near-singular matrix and NaN-laden centrality scores on some real graphs with nothing catching it.

### (edge_case) Inf-FS centrality: centrality_percentile cutoff at boundary values 0 and 100

File: friend_graph.py, build_friend_graph() lines 555-557. Fit with centrality_percentile=0.0 (nothing should be excluded as low_centrality, since np.percentile at 0 is the min, and only strictly-<=-min nodes qualify) and centrality_percentile=100.0 (potentially ALL nodes below-or-equal the max qualify as low_centrality, i.e. everyone gets flagged) -- assert low_centrality's length in each case behaves as documented rather than crashing on the percentile computation for a single-element `centrality.values()` list.

### (edge_case) random_seed threading: both random_seed and random_state set to conflicting values

File: src/mlframe/feature_selection/filters/mrmr/_mrmr_class.py, lines 2950-2966 in __init__. Construct MRMR(random_seed=7, random_state=99) and assert (a) a UserWarning fires naming both values (line ~2958), (b) `_effective_random_seed()` resolves to 99 (random_state wins), and (c) fitting twice with this exact config produces IDENTICAL selections (i.e. the conflict-resolution warning path doesn't silently randomize which value actually gets used run-to-run). Also test random_seed=7, random_state=7 (same value, no conflict) asserts NO warning fires -- pins the `random_seed != random_state` guard on line 2950 against a future edit that warns even on equal values.

### (edge_case) random_seed threading: clone() before fit preserves unresolved random_seed/random_state pair

File: mrmr/_mrmr_class.py, __getstate__/__setstate__ (lines 3069-3157) and the constructor's lazy resolution comment (lines 2932-2935: '_effective_random_seed resolved LAZILY at fit time, NOT here'). sklearn.base.clone(MRMR(random_seed=5)) BEFORE any fit, then check the clone's raw `random_seed`/`random_state` attrs (via get_params()) still show the pre-resolution values (random_seed=5, random_state=None) rather than the clone silently promoting one into the other -- pins the documented invariant that mutating at construction time would break `get_params()` round-tripping (sklearn's clone contract) which the comment explicitly calls out as the reason resolution is deferred to fit time.

### (edge_case) random_seed threading: identical selections across repeated seeded fits despite intervening unseeded fits on numpy global RNG

File: mrmr/_mrmr_class.py, lines 3719-3746 (`_preserve_global_numpy_rng_state` context, 'reseeds numpy/numba/cupy'). Fit MRMR(random_state=42) on a fixture, capture support_; then fit a DIFFERENT unseeded MRMR() (perturbing the global numpy RNG state via its internal draws); then fit the ORIGINAL MRMR(random_state=42) config again on the same data and assert support_ is IDENTICAL to the first seeded fit. Directly targets the comment's claim that 'even a SEEDED fit was NON-[deterministic before this fix]' -- the regression is specifically about global RNG state leaking BETWEEN fits, which requires an intervening unseeded fit in the test to actually exercise the `_preserve_global_numpy_rng_state` save/restore rather than just seeding once in isolation.

### (edge_case) random_seed threading: random_state=None (entropy-seeded) fits are non-reproducible but internally consistent

File: mrmr/_mrmr_class_config.py, _effective_random_seed() line 236-246 ('Returns None when neither is set (entropy-seeded)'). Fit MRMR(random_state=None) twice on the same data and assert the two support_ sets are ALLOWED to differ (no reproducibility guarantee) but that a SINGLE fit's internal randomized sub-steps (e.g. synergy detection at _mrmr_class.py:3575 `detect_synergy(..., random_seed=int(self._effective_random_seed() or 0))`) don't silently coerce None to 0 and become secretly deterministic while random_state=None elsewhere in the same fit stays entropy-seeded -- an inconsistency where 'unseeded' partially means 'seeded with 0' would be a subtle reproducibility footgun for a user who explicitly wants non-determinism for a bagging/ensemble use case.

### (edge_case) degenerate: n=1 sample fit

File: mrmr/_mrmr_class.py, MRMR.fit(). Fit on X with a single row (n=1), y a single scalar label. Expect either a clear, documented ValueError/warning (not a silent empty support_ or a divide-by-zero deep in the MI/binning machinery) -- exercise both classification (y=[0]) and regression (y=[3.14]) targets. Standard degenerate-input class from the task brief; MI estimation, quantile binning, and the Miller-Madow bias term (`2*n_samples` in the denominator at _screen_predictors_gate.py:121/150/183) all divide by n, so n=1 is a concrete division-by-near-zero risk path, not just a generic 'small n' smoke test.

### (edge_case) degenerate: all-NaN column in X survives to a documented outcome

File: mrmr/_mrmr_class.py fit / discretization pipeline. Construct X with one column entirely np.nan (all 200 rows), other columns normal signal. Fit and assert the all-NaN column is EXCLUDED from support_ (not silently NaN-poisoning a joint MI computation for a different column via a shared binning pass) and that no warning/exception is swallowed silently -- verify via the FE-family bare except-Exception audit pattern: an all-NaN column is exactly the kind of edge case likely to trip one of the 178 unguarded `except Exception: pass` sites in _fit_impl_core.py without leaving a diagnostic trail.

### (edge_case) degenerate: duplicate columns in X (bit-identical twins)

File: mrmr/_mrmr_class.py + _fe_raw_redundancy_anchors.py (build_raw_redundancy_anchors). Construct X with column 'a' and an exact copy 'a_dup' (both signal-carrying and correlated to y). Fit and assert MRMR selects at most ONE of {a, a_dup} (redundancy gate correctly identifies bit-identical duplicates as maximally redundant, MI(a;a_dup)=H(a) exactly) -- this exercises build_raw_redundancy_anchors's host binning path on a boundary case (correlation exactly 1.0, not just 'highly correlated'), which per the coverage gap has zero direct test and whose GPU-resident twin (_quantile_bin_gpu_resident) is claimed-but-unverified to tie-break identically on this exact degenerate case.

### (edge_case) degenerate: duplicate rows do not bias MI estimation asymmetrically vs GPU path

File: _fe_raw_redundancy_anchors.py + core MRMR fit. Construct X,y where 50% of rows are exact duplicates of the other 50% (a common real-world artifact from upstream joins). Run the SAME fit twice: once forced CPU (use_gpu=False) and once forced GPU (use_gpu=True, skipped/xfail gracefully if no CUDA device present) and assert selection-equivalent support_ -- ties in the quantile-bin boundary computation (many identical values landing on the same bin edge) are exactly where a CPU vs GPU divergence in tie-breaking would first surface, per the coverage gap's cpu_gpu_parity concern for radix-select and quantile binning.

### (edge_case) degenerate: extreme class imbalance, single minority-class sample

File: mrmr/_mrmr_class.py MRMR.fit (classification). y has 999 samples of class 0 and exactly 1 sample of class 1. Fit and assert: no crash in MI/entropy estimation (a single-sample stratum has zero within-class variance by construction), stratified resampling paths (e.g. cv_shuffle, group_aware_mi's per-group estimator) degrade gracefully rather than raising an unguarded IndexError/ValueError, and the result is a valid (possibly small) support_ rather than an empty one given min_features_fallback's documented never-empty floor.

### (edge_case) degenerate: constant y (zero-variance target)

File: mrmr/_mrmr_class.py MRMR.fit. y is a constant array (all rows same value). Every candidate's MI(X;y)=0 exactly, so every gain is 0.0 and should fail the abs floor in compute_selection_gate() -- assert the fit does NOT crash (e.g. divide-by-zero in a normalized-MI ratio somewhere) and, per min_features_fallback's documented contract (_param_accuracy_warnings.py line 86), still returns a non-empty support_ of size >= min_features_fallback rather than an empty one, since 'a fit whose gates reject every candidate' is exactly this scenario.

### (edge_case) GPU path: use_gpu=True on a machine with no CUDA device falls back cleanly with a single clear warning

File: mrmr/_mrmr_class.py + _gpu_policy.py / _gpu_hw_launch.py. Force cupy import to fail (monkeypatch sys.modules['cupy']=None or run on a genuinely CPU-only CI box) and fit MRMR(use_gpu=True). Assert: fit completes, selection matches use_gpu=False on the same data (selection-equivalence, not necessarily bit-identical), and at most ONE fallback warning/log line is emitted per fit (not one per candidate/per-FE-family, which would spam logs across hundreds of internal GPU-gated call sites like _gpu_resident_radix_ktc.py, _usability_gpu.py, _resident_bincount.py etc, several of which currently have zero test coverage per the batch gap finding).

### (edge_case) GPU path: circuit breaker recovers cleanly on the NEXT fit after a simulated CUDA OOM mid-fit

File: _fe_gpu_vram.py / _fe_additive_fusion.py (cp.cuda.memory.OutOfMemoryError catch sites) + _fe_pure_form_retention_gpu_resident.py:106. Monkeypatch one GPU kernel call to raise cp.cuda.memory.OutOfMemoryError on the FIRST fit only (a stateful mock: raises once, then behaves normally), fit MRMR(use_gpu=True) and assert (a) fit 1 completes via graceful CPU fallback rather than propagating the OOM, and (b) a SECOND, immediately-following fit on fresh data with the mock now behaving normally runs the GPU path again rather than being permanently pinned to CPU by any leftover module-level 'GPU is broken' flag -- this is the actual test gap named in the task brief ('does the circuit breaker actually recover the next fit cleanly') and nothing in the current suite exercises the two-fits-in-a-row sequence.

### (edge_case) GPU path: a GPU-resident kernel handed a CPU (numpy, not cupy) array by mistake fails loudly, not silently wrong

File: _gpu_resident_radix_ktc.py / _resident_bincount.py / _resident_raw_mi.py (any of the '_dev_from_cont'/'_dev_from_codes' style resident helpers). Directly call one of the GPU-resident kernel wrapper functions (bypassing the normal dispatch that guarantees a cupy array) with a plain numpy.ndarray and assert it either (a) raises a clear TypeError/AttributeError immediately (cupy-specific calls like `.device` fail fast on ndarray) rather than (b) silently producing a wrong numeric result via numpy's duck-typed broadcasting mistaking the array for something it can partially operate on. This targets the coverage gap's explicit ask for 'a GPU-resident kernel handed a CPU array by mistake' and is runnable on CPU-only CI since it tests the type-mismatch failure mode, not the GPU compute path itself.

### (edge_case) radix_select_threads / radix_select_f32_variant: KTC lookup-miss and malformed .choose() return fall back to a safe default

File: _gpu_resident_radix_ktc.py lines 76-110 and 233-244 (`_RADIX_THREADS_SPEC = None` on tuner-init failure). Directly call radix_select_threads(n=10_000) with `_RADIX_THREADS_SPEC` monkeypatched to None and assert it returns the documented CPU-safe default rather than raising; then monkeypatch `_RADIX_THREADS_SPEC.choose` to raise an arbitrary Exception and assert the SAME default fallback fires with a WARNING logged (not silently); then monkeypatch `.choose()` to return a malformed value (e.g. a string where an int block-size is expected, or a negative number) and assert the function does NOT propagate that malformed value into a kernel launch unchecked. All three are plain-Python, no-CUDA-required paths the coverage gap explicitly flags as untested despite needing no hardware.

### (edge_case) radix_select_f32_variant: regression test pinning the 2026-07-18 wrong-module-binding fix

File: _gpu_resident_radix_ktc.py, ~line 139 comment referencing the shipped-silently bug where the sweep probe wrote its override onto a re-export alias instead of the real module binding. Write a test that sets the variant override through the documented public entry point, then calls radix_select_f32_variant(n) and asserts the OVERRIDDEN value is actually returned (not the pre-override default) -- i.e. assert the override round-trips through the SAME binding the getter reads, which is precisely what silently failed to happen in the shipped bug. This is a named historical bug with zero regression test today per the coverage gap.

### (edge_case) multi-output y: _mrmr_y_columns on a polars.DataFrame (duck-typed branch)

File: mrmr/_mrmr_class_shared.py, _mrmr_y_columns() lines 21-24: `str(type(y).__module__).startswith('polars')`. Directly call _mrmr_y_columns(polars_df) with a real 2-column polars.DataFrame and assert the yielded (label, array) pairs match column names and values exactly, AND separately construct a duck-typed fake object whose `type(y).__module__` is a string starting with 'polars' but is NOT an actual polars.DataFrame (e.g. a test double) to confirm the isinstance-avoidance is intentional and doesn't crash on `.columns`/`[col]` access it doesn't support -- covers the exact named gap ('duck-typed via module-name-startswith rather than isinstance') and its stated risk ('polars renaming its module path').

### (edge_case) multi-output y: _mrmr_y_columns on a raw 2D ndarray uses correct 0-indexed labels

File: mrmr/_mrmr_class_shared.py, lines 25-27: `for k in range(arr.shape[1]): yield f'y{k}', arr[:, k]`. Call _mrmr_y_columns(np.column_stack([a,b,c])) (3 columns) and assert labels are exactly ['y0','y1','y2'] in that order AND that each yielded array is bit-identical to the corresponding source column (no off-by-one column selection, e.g. arr[:,k+1] instead of arr[:,k]) -- directly targets the named 'off-by-one in the ndarray label index' risk in the coverage gap, currently only reachable if an e2e multi-output test happens to use a bare ndarray y.

### (edge_case) multi-output y: MRMR.fit end-to-end with 2D y across all three container types produces per-output distinguishable results

File: mrmr/_mrmr_class.py multi-output fit path (_fit_multioutput, referenced near line 3347). Fit the SAME X against y as (a) a 2-column pandas.DataFrame, (b) a 2-column polars.DataFrame, (c) a raw (n,2) ndarray, where the two target columns have DELIBERATELY DIFFERENT signal columns in X (target 0 correlates with X[:,0], target 1 correlates with X[:,5]). Assert all three container types produce the SAME per-output feature attribution (via whatever provenance/per-output support MRMR exposes) -- catches a regression where the polars branch's duck-typed dispatch silently falls through to the wrong column-extraction path and scrambles which target's signal gets attributed to which output.

### (edge_case) partial_fit resumption across mismatched dtypes

File: src/mlframe/feature_selection/filters/_mrmr_partial_fit.py, partial_fit() line 159, and _to_dataframe/_to_series helpers (lines 64-95). Call MRMR().partial_fit(X_batch1) where X_batch1's numeric column is float32, then partial_fit(X_batch2) on the SAME column as float64 (or int64), and assert the accumulated rolling-window state (_apply_rolling_window, line 115) does not silently upcast/downcast in a way that changes binning edges between batches -- e.g. a float32 batch quantized against float64-computed bin edges from a prior batch could shift which bin a boundary value falls into. Assert either a clear dtype-coercion (documented, consistent) or an explicit warning, not silent divergent behavior across batches.

### (edge_case) pickling mid-fit vs post-fit

File: mrmr/_mrmr_class.py, __getstate__/__setstate__ (lines 3069-3157). Test 1: pickle.dumps(mrmr) AFTER a completed .fit() and pickle.loads it back; assert support_ / transform() output matches the original exactly (existing __getstate__ already exists so this pins current behavior). Test 2 (the actual gap): use a monkeypatched hook to interrupt fit MID-WAY (e.g. patch an internal FE-family function to raise after partial state like `self._fit_sample_weight_` (line 3418) is set but before `self.support_` is finalized), catch the exception, then attempt `pickle.dumps(mrmr)` on this PARTIALLY-fitted, exception-raised instance and assert it either pickles cleanly with a well-defined 'not fitted' state or raises a clear error -- not a cryptic AttributeError from __getstate__ assuming fully-fit attributes exist that were never set.

### (edge_case) clone() before vs after fit differ in fitted-state but agree on get_params()

File: mrmr/_mrmr_class.py + sklearn.base.clone contract. clone(mrmr_before_fit) and clone(mrmr_after_fit) (both starting from the SAME constructor params) must produce `get_params()` dicts that are IDENTICAL to each other and to the original's pre-fit params (clone never carries over `support_`/fitted attrs) -- assert hasattr(cloned, 'support_') is False for both clones even though the after-fit source object DOES have support_. This is the standard sklearn clone contract but is worth pinning explicitly here because the constructor's lazy random_seed/random_state resolution (lines 2932-2966) is EXACTLY the kind of stateful-looking-but-must-not-leak attribute that a naive clone() implementation could get wrong.

### (edge_case) groups= without group_aware_mi=True raises NotImplementedError under strict_groups default

File: mrmr/_mrmr_class.py, lines 1221-1226 (`strict_groups` default True, finding #20 in comments). Fit MRMR().fit(X, y, groups=group_array) with strict_groups left at its default and group_aware_mi=False; assert NotImplementedError is raised (not silently ignored, not just a UserWarning) -- then fit with strict_groups=False explicitly and assert the legacy warn-only behavior (groups accepted but not consumed, UserWarning emitted) fires instead. Pins the documented behavior-flip between the two strict_groups values, a leakage-guard correctness gap the code comments explicitly call out as a 'correctness gap' when silently degraded.

### (edge_case) groups= combined with non-uniform sample_weight disables group_aware_mi with a warning, not silent misalignment

File: mrmr/_mrmr_class.py, lines 3608-3631. Fit with group_aware_mi=True, groups=group_array, AND a non-uniform sample_weight simultaneously; assert group_aware_mi is force-disabled for this fit (per the documented 'resampling rows would misalign them against groups' guard) and a warning naming exactly this reason is emitted -- then verify the resulting selection is NOT silently corrupted by rows being resampled (via _maybe_resample_for_sample_weight) while `groups` labels stay in their original, now-misaligned order. This is the concrete leakage-adjacent bug class ('sample_weight edge cases' x 'groups=' interaction) named in the task brief's coverage matrix.

### (edge_case) sample_weight: all-zero weights raise a clear ValueError, not a silent uniform-weight fallback

File: mrmr/_mrmr_class_fit_helpers.py, _maybe_resample_for_sample_weight() lines 292-317. Call MRMR().fit(X, y, sample_weight=np.zeros(len(X))) and assert ValueError('sample_weight sums to zero') is raised (line ~313) rather than the resampling silently treating all-zero as all-equal and falling through to the uniform-legacy no-op path -- this is the exact 'all-zero weights' edge case named in the task brief; the code already has an explicit guard (`if total <= 0: raise`) but it has no dedicated unit test isolating this branch from the rest of _maybe_resample_for_sample_weight's logic.

### (edge_case) sample_weight: one dominant weight concentrates resampled rows without crashing on near-duplicate binning

File: _mrmr_class_fit_helpers.py, _maybe_resample_for_sample_weight(). sample_weight where row 0 has weight 1e6 and all other 999 rows have weight 1.0 -- assert the resample (with-replacement draw proportional to w_i/sum(w)) produces a result where row 0 is massively overrepresented (statistically, e.g. >90% of drawn rows), the fit completes without error even though the resampled X now has extreme row-duplication (interacts with the 'duplicate rows' degenerate-input class above), and the seeded resample (via the documented `int(seed or 0)` convention at line ~317) is reproducible across two runs with the same random_state.

### (edge_case) sample_weight: negative weight raises ValueError with correct message

File: _mrmr_class_fit_helpers.py, line 309: `if not np.all(np.isfinite(sw)) or (sw < 0).any(): raise ValueError(...)`. Parametrize over sample_weight containing (a) a single -0.001 entry among otherwise-valid positive weights, (b) a np.inf entry, (c) a np.nan entry -- assert all three raise the SAME ValueError('sample_weight must be finite and non-negative') rather than propagating NaN/inf silently into the probability-normalization (`w_i/sum(w)`) a few lines later, which for the NaN case would poison EVERY row's draw probability, not just the offending row's.

### (edge_case) FE family double-counting: fe_pairwise_modular and fe_conditional_gate both engineer the same underlying signal

Files: _cat_pair_fe.py / _pairwise_modular_fe.py + _conditional_gate_fe.py, both reachable when fe_discrete_structural_operators_enable=True (the default per _param_accuracy_warnings.py's registry, which explicitly groups these four families as toggled together). Construct a synthetic where the true generative signal is a simple threshold-gated interaction (y depends on `a if c>tau else b`) that BOTH the conditional-gate family (by design) and, incidentally, a pairwise-modular family could partially approximate via a different basis. Fit and assert MRMR's redundancy gate (not just the FE accuracy gate) drops the SECOND, weaker-conditional-gain engineered duplicate rather than selecting both near-redundant engineered columns and inflating apparent support_ size -- probes the task brief's named risk ('two FE families that could double-count the same signal') using two families the codebase's own registry already documents as commonly co-enabled.

### (edge_case) FE family requiring GPU with use_gpu=False falls back to CPU or is cleanly skipped, never silently no-ops with wrong output

Files: any GPU-only-labeled family, e.g. _fe_pure_form_retention_gpu_resident.py / _hinge_detect_gpu_resident.py / _fe_additive_fusion_gpu_resident.py. Fit MRMR(use_gpu=False, <that family's enable flag>=True) and assert the family either (a) transparently runs its CPU-equivalent sibling (e.g. _fe_pure_form_retention.py, the non-'_gpu_resident' counterpart) producing the SAME engineered columns as it would GPU-resident, or (b) is explicitly skipped with a debug-level log naming why -- but never silently returns an empty contribution while use_gpu=False is misinterpreted deep in the family as 'GPU unavailable, produce nothing' without surfacing that as a a coverage decision the caller can see. Directly probes the task brief's named FE-interaction risk.

### (edge_case) FE hyperparameter at extreme: degree=0 for a polynomial/orthogonal-degree family

File: _orthogonal_adaptive_degree_fe.py / polynom_pair_fe.py (whichever family exposes an explicit `degree` knob). Fit with the family's degree parameter forced to 0 and assert it either raises a clear ValueError (degree=0 is meaningless for a polynomial basis) OR degrades to a documented identity/no-op (emits zero new features, not a degenerate constant-1 column that silently passes every relevance gate because its MI(constant;y)=0 exactly matches the abs_floor=0.0 edge in compute_selection_gate -- a genuine boundary interaction between this family's degenerate output and the gate module's own `>=` comparisons).

### (edge_case) FE hyperparameter at extreme: top_k=0 for a top-k-selecting family (e.g. Inf-FS centrality-based pruning, synergy screen)

Files: _fe_synergy_screen.py / any family exposing a `top_k` selection knob, plus friend_graph.py's centrality_percentile as an analog. Fit with top_k=0 and assert the family emits ZERO candidates (not an off-by-one that emits 1, and not a crash from an empty top-k array feeding into a downstream `np.argpartition` or similar that assumes a positive k) -- and that the overall MRMR fit still succeeds via other families / core screening rather than the zero-candidate family propagating a shape-(0,) array into a later concatenation that silently drops or corrupts other families' outputs.

### (edge_case) FE hyperparameter at extreme: alpha=0 threaded specifically into renyi_alpha_mi's alpha (not just Rényi as an FE knob)

File: _renyi_alpha.py, renyi_alpha_mi(x, y, alpha=0.0). At alpha=0, `S_alpha = 1/(1-0) * log2(sum(lambda_i^0))` = `log2(count of eigenvalues > 1e-12)` -- i.e. Rényi-0 entropy degenerates to log2(matrix rank), a well-defined but qualitatively DIFFERENT quantity from the default alpha=1.01's near-Shannon entropy. Assert the function returns a finite, non-negative value at alpha=0 (not a crash, not the same value as alpha=1.01) and that callers who accidentally pass alpha=0 (e.g. a config typo) get a value that at least doesn't silently masquerade as a valid MI on the same numeric scale the default alpha produces -- a concrete 'hyperparameter at its extreme' scenario for the newly-added 2026-07 estimator, distinct from the alpha=1.0 singularity test above.

### (edge_case) clone() interaction with random_state resolution: cloning after a fit that resolved random_state=-1 to a derived seed

File: mrmr/_mrmr_class_config.py, line 252-253 ('a pickled/cloned estimator re-resolves -1 against...' -- derives a deterministic 6-hex seed, line 289-294). Fit MRMR(random_state=-1) (the documented 'derive-a-fresh-seed' sentinel), capture the resolved `_effective_random_seed()` value used internally, then clone() the estimator and fit the CLONE -- assert the clone independently RE-DERIVES its own -1 resolution (per get_params() showing random_state still -1, not the resolved value) and that this can legitimately differ from the original's resolved seed, since re-resolving is the documented contract -- versus a bug where clone() accidentally carries over the ALREADY-RESOLVED concrete seed, making two supposedly-independent clones spuriously identical when the user's whole intent behind -1 was 'give me an independently-fresh seed per clone'.
