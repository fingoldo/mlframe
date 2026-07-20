# GPU Residency Architect

13 findings, 10 proposals.

## Findings

### [P1] gpu_residency -- src/mlframe/feature_selection/filters/gpu.py:614

**mi_direct_gpu's legacy per-permutation loop does a blocking cp.ndarray.get() every iteration, and this is the ONLY code path whenever return_null_mean=True.**

The batched-dispatch fast path (mi_direct_gpu_batched, gpu.py:503-520) is explicitly excluded whenever return_null_mean=True (`and not return_null_mean`). But return_null_mean=True also forces `npermutations = max(npermutations, _NULL_MEAN_MIN_PERMS)` (>=32) and `max_failed = npermutations` (early-stop disabled), so this path ALWAYS runs the full >=32-iteration for-loop at line 614, and every iteration ends with `mi = totals.get()[0]` (line 628) -- a blocking D2H sync per permutation. This is exactly the anti-pattern the sibling file _gpu_batched.py (same package, lines 158-245) already fixed for the batched path ("stage each batch's failure-count ... sync ONCE at end-of-loop"), but the fix was never back-ported to this fallback loop. Any caller that needs the GPU relevance null-mean/p-value for significance-gated debiasing (the documented purpose of return_null_mean) pays npermutations blocking syncs per candidate pair for the whole fit.

### [P1] gpu_residency -- src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py:370

**_CMI_RESIDENT_CACHE evicts by clearing the WHOLE cache once more than 16 distinct y/z roles are uploaded in a fit, instead of an LRU coldest-only eviction.**

`if len(_CMI_RESIDENT_CACHE) > 16: _CMI_RESIDENT_CACHE.clear()` (line 370-371) drops every cached device upload, not just the oldest one, as soon as a greedy-CMI round has touched more than 16 distinct (factors_data, column, nbins) roles -- routine on any fit scanning more than 16 candidate columns. The sibling cache in _fe_resident_operands.py explicitly documents why this exact pattern is wrong: "the single COLDEST entry is evicted on overflow, NOT the whole table -- a clear-all forces a re-upload storm of the still-hot operands" (its own module docstring), and implements a real OrderedDict LRU. _cmi_cuda._resident_upload was never updated to match, so the CMI-greedy hot path this cache exists to protect (documented in the same file as ~120 y/z upload sites/fit) periodically re-uploads everything it just cached, directly undermining the H2D-avoidance the cache was built for.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_gpu_strict_fe/_entry.py:167

**5 of the 10 DEFAULT-ON device_born_* residency flags (BINAGG, DISPERSION, DUAL_UPLIFT, WAVELET, and EXTRA_BASIS-toggle) have zero test that flips their env var and compares host vs device-born output.**

Grepping every test under tests/ for the literal env-var names MLFRAME_FE_GPU_DEVICE_BORN_BINAGG, _DISPERSION, _DUAL_UPLIFT, and _WAVELET returns zero hits anywhere in the repository; MLFRAME_FE_GPU_DEVICE_BORN_EXTRA_BASIS appears in exactly one test (test_extra_basis_device_born_parity.py) which exercises the function but never toggles the flag on/off for an explicit before/after comparison the way test_resident_311_residual_parity.py does for gate/modular/raw_baseline/uplift_univariate. tests/feature_selection/fe/gpu/test_conditional_dispersion_resident.py and test_binned_numeric_agg_resident.py both exist but neither references DEVICE_BORN_DISPERSION/DEVICE_BORN_BINAGG, so they test a different (older) resident code path, not the specific H2D-avoidance mechanism these flags gate. Each of these flags is DEFAULT ON in production and each carries a specific numeric bit-identity claim in its docstring (e.g. dispersion: "only the per-row f64 divide differs at ~1e-10 ULP") that is currently unverified by any automated test.

### [P1] test_gap -- src/mlframe/feature_selection/filters/_wavelet_basis_fe.py

**fe_gpu_device_born_wavelet_enabled's docstring cites a pinning test, test_wavelet_batched_mi_parity, that does not exist anywhere in the repository.**

The docstring in _gpu_strict_fe/_entry.py (fe_gpu_device_born_wavelet_enabled, ~line 178) states the device/host partition-equivalence is "pinned by test_wavelet_batched_mi_parity". Grepping the entire repo for `def test_wavelet_batched_mi_parity` and for the bare string `test_wavelet_batched_mi_parity` (case-sensitive, all files) finds zero function definitions -- the string appears only inside the docstrings of _wavelet_basis_fe.py, _wavelet_basis_fe_batched.py, and _gpu_strict_fe/_entry.py themselves, never in a tests/ file. tests/feature_selection/fe/basis/test_wavelet_basis_fe.py exists but contains no DEVICE_BORN toggle and no function of that name. The DEFAULT-ON wavelet device-born mechanism's bit-identity claim is therefore currently backed by no regression test, contrary to what its own gating docstring asserts.

### [P1] test_gap -- tests/feature_selection/gpu/test_cmi_residency_traffic.py:1

**The repository's only byte-level residency_audit() regression test covers one FE family (greedy_cmi_fe_construct) that is opt-in and off by default; the ~10 DEFAULT-ON device_born_* families that actually deliver residency on a normal MRMR.fit() have no byte-traffic audit at all, and no test audits a full multi-family MRMR.fit() call.**

test_cmi_residency_traffic.py wraps greedy_cmi_fe_construct (via the wired greedy loop) in residency_audit() and asserts zero bulk D2H/H2D. That code path is reached from MRMR.fit only when `self.fe_mi_greedy_cmi_enable` is True, which defaults to False (`getattr(self, "fe_mi_greedy_cmi_enable", False)` in _mrmr_fit_impl/_fit_impl_core.py:2612) -- so the ONE audited path is not exercised on a default fit. Meanwhile the gate/binagg/dispersion/crossbasis/dual_uplift/wavelet/raw_baseline/uplift_univariate/extra_basis/modular families -- all DEFAULT ON under STRICT-residency and each individually claiming to collapse a specific 20-300MB host upload -- are checked only for selection-equivalence (where checked at all, see the wavelet/binagg/dispersion/dual_uplift gaps above), never for actual transferred-byte volume. A regression that silently reintroduces a bulk upload in any of these (e.g. a future edit to `assemble_resident_matrix`'s fallback branch, _fe_resident_operands.py:193-199) would pass every existing test and still balloon back to the pre-fix H2D volume the docstrings describe fixing.

### [P2] bug -- src/mlframe/feature_selection/filters/_fe_gpu_strict.py:40

**The module docstring's "RESIDENCY GAP FOUND AND CLOSED" note incorrectly claims greedy_cmi_fe_construct is called ONLY from a standalone benchmark script and never from MRMR.fit.**

greedy_cmi_fe_construct IS reachable from MRMR.fit: _mrmr_fit_impl/_fit_impl_core.py:2615 imports and calls `greedy_cmi_fe_construct_with_recipes` (gated on `fe_mi_greedy_cmi_enable`), and greedy_cmi_fe_construct_with_recipes (_mi_greedy_cmi_fe.py:1826) calls `greedy_cmi_fe_construct` directly -- the exact function the docstring says is benchmark-only. A maintainer reading this docstring to triage residency risk would incorrectly conclude a bug found in this code path (or a regression of the fix) is production-inert, when it actually affects any fit run with fe_mi_greedy_cmi_enable=True. Low severity because the underlying fix itself is real and tested, but the risk-assessment claim in the docstring is factually wrong and should be corrected before anyone relies on it.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/friend_graph_gpu.py:215

**friend_graph_gpu's O(k^2) edge pass D2Hs the raw integer joint/marginal histograms every tile (lines 215, 246, 309) so entropy can be reduced on the bit-exact CPU path -- a self-imposed design choice, not a hardware necessity, and the single largest remaining structural residency gap in this cluster.**

The module docstring states this is "non-negotiable": entropy is computed by the CPU njit `entropy()` specifically so results are bit-identical to the CPU build, forcing counts (not MI) to cross the bus at cp.asnumpy(...) on every node-marginal pass (215), every relevance pass (246), and every pairwise-edge tile (309, the dominant O(n*k^2) cost this whole module exists to accelerate). This is LOAD-BEARING under the current contract, but the constraint is self-imposed at a stricter bar (literal CPU bit-identity) than every other GPU/CPU parity claim in this codebase, which tolerates ~1e-9-1e-7 FP-reorder divergence as "selection-equivalent" (see _orth_mi_backends parity tests, rtol=1e-7/1e-9). Porting `_entropy_from_counts`'s formula (`-(log(freqs)*freqs).sum()`) to a cupy elementwise+reduction kernel, validated to the same tolerance the rest of the codebase already accepts, would remove this D2H entirely -- but doing so requires a new, deliberately-validated parity test, not a mechanical code change, so it is flagged as AVOIDABLE-with-validation rather than a drop-in fix.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/_fe_resident_operands.py:130

**resident_operand's own docstring documents an unfixed residual inefficiency: fit-constant y/z are re-hashed (O(n) content hash) on every call from every role instead of being uploaded and handed to callers by reference once per fit.**

The docstring at line 130 states verbatim: "the fit-constant y / z are re-hashed on every role's call; a caller-side upload-dedup (upload y/z ONCE and pass the resident handle) would skip resident_operand entirely for them, but that is a caller change outside this file." Every one of the ~9 documented roles (cmi_y, card_y, fixedyz_y, y_mi_classif, orth_uni_y, cmi_greedy_y_fixed, cmi_z, card_z, fixedyz_z) still pays a copy-free-but-nonzero xxh3_64 hash over the full y/z buffer on every call to confirm the cache hit, rather than the fit-step dispatcher handing every consumer the same already-resident cupy array by reference. Individually cheap (hash is O(n), no H2D), but it is the one gap in this file the authors explicitly flagged as real and unaddressed, and it sits on the hottest per-candidate-batch call path in the whole STRICT-resident FE pipeline.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/_fe_gpu_batch/_executor.py:93

**gpu_fe_batch_mi defaults free_blocks=True, forcing a real cudaFree+cudaMalloc on every VRAM-chunked candidate-matrix batch instead of reusing cupy's memory pool; the code comment documents this as a known, unresolved cost.**

The in-code comment at lines 93-100 states this is "a genuine cost" and that verifying pool-retention is safe "needs live multi-GPU profiling this wave didn't run; not changed without it." gpu_fe_batch_mi is reached once per candidate-matrix batch, many batches per fit, and concurrently across devices from multi_gpu_fe_batch_mi's ThreadPoolExecutor -- so the forced allocator churn recurs on every chunk of every batch of every FE family that reaches this executor. This is exactly the kind of avoidable-but-deferred item the task asks to surface even when already known: it is real, measured-as-real by the author, and simply pending the profiling evidence needed to flip the default safely.

### [P2] design -- src/mlframe/feature_selection/filters/_gpu_strict_fe/_state.py:23

**ResidentFEState / run_fe_step_gpu_strict is dead scaffold code with zero production callers, confirmed by grep across the entire src tree.**

`grep -r "run_fe_step_gpu_strict(\|ResidentFEState" src/` outside the two files defining them returns nothing; run_fe_step_gpu_strict (_entry.py:271) unconditionally raises NotImplementedError and ResidentFEState.build (_state.py:43) has no caller anywhere in src/. The _entry.py module docstring itself says this was superseded by the per-family device_born_* mechanism and explicitly warns "Do not wire this stub up." Not a functional bug (nothing breaks), but ~430 lines of maintained, type-annotated, tested (test_gpu_strict_resident_scaffold.py) infrastructure exist purely as a documented dead end -- worth an explicit decision (delete, or archive under a clearer "historical/rejected" marker) so a future contributor scanning for "what enforces residency" doesn't have to read the whole class to learn it does nothing.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters

**Clean: no bare/unlogged exception-swallowing found among the ~35 GPU-residency call sites reviewed across this cluster.**

No issues found in this cluster for this angle -- every `except Exception:` / `except:` encountered while reviewing gpu.py, the _gpu_strict_fe package, _fe_gpu_batch, _fe_resident_operands.py, _cmi_cuda.py, friend_graph_gpu.py, _gpu_resident_basis.py, _mi_greedy_cmi_fe.py, and the ~20 other GPU-resident modules grep-scanned for H2D/D2H patterns is either tagged `# nosec B110` with an inline rationale or routes through `logger.debug(...)` naming the exception before falling back, matching the project convention.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters

**Clean (post-fix): the stale-cache-invalidation risk class (identity-only keys aliasing a recycled id() with different content) is guarded everywhere reviewed, including the one place it was historically missing.**

No issues found in this cluster for this angle beyond the eviction-policy bug already reported separately (info_theory/_cmi_cuda.py:370). Both device caches reviewed -- _fe_resident_operands._FE_RESIDENT_OPERANDS (content hash: shape+dtype+xxh3_64/njit-word-hash) and info_theory._cmi_cuda._CMI_RESIDENT_CACHE (content hash: shape+dtype+hash(tobytes())) -- fold a full content fingerprint into the cache key specifically to guard against id() recycling; the _cmi_cuda module's own comment documents that this content-hash guard was added retroactively after the identity-only key was identified as a real correctness risk ("a recycled id with the SAME column index + shape + dtype but DIFFERENT values would return a STALE device copy").

### [P2] cpu_gpu_parity -- tests/feature_selection/gpu/test_resident_311_residual_parity.py:154

**Clean (historical, already fixed and regression-tested): the one silent CPU/GPU divergence found in this cluster's history -- _mi_classif_batch_numba's _orth_mi_gpu_enabled gate ignoring an explicit rank_binning=True request and silently swapping RANK binning for EDGE binning under MLFRAME_CMI_GPU=1 -- was found 2026-07-16 and is now pinned by test_sf2_combiner_baseline_and_residue_mi_byte_identical.**

No currently-unresolved instance of this bug class was found in the files reviewed. Documenting it here because its test docstring is itself useful evidence for the roadmap: this exact bug class (a shape-blind or flag-blind GPU dispatch gate silently choosing a different binning/estimator than the caller requested, producing a ~6% MI divergence rather than the expected ~1e-9 FP-reorder noise) is a real, recurring risk pattern in this codebase and motivates a class-wide audit of every `_*_gpu_enabled()` / `_should_use_cuda` gate for the same blind spot, not just a one-off fix.

## Proposals

### (residency_step) 1. Fix _CMI_RESIDENT_CACHE's whole-cache-clear eviction to a real LRU

Swap the raw dict at info_theory/_cmi_cuda.py:338 for an OrderedDict with coldest-only eviction (`popitem(last=False)`), mirroring _fe_resident_operands._FE_RESIDENT_OPERANDS exactly. Cheapest, highest-confidence item on this list: a ~5-line change to code that already has the content-hash correctness guard in place, removes a self-inflicted re-upload storm on any greedy-CMI round scanning >16 candidates, and the file already has a template (the sibling cache) to copy.

### (residency_step) 2. Batch the mi_direct_gpu return_null_mean permutation loop's per-iteration .get()

In gpu.py's mi_direct_gpu (the loop starting at line 614), stage each permutation's `mi` scalar into a resident (npermutations,) cupy buffer instead of calling totals.get()[0] every iteration, and do ONE D2H (plus the nfailed/_null_sum reduction) after the loop -- exactly the pattern _gpu_batched.py (same package) already proved out for the batched path (lines 158-245, 'single end-of-loop sync'). Since return_null_mean=True disables early-stop (max_failed=npermutations) this loop always runs to completion anyway, so batching the readback costs nothing in lost early-stop opportunity and removes npermutations blocking syncs per relevance-debiasing call.

### (residency_step) 3. Add residency_audit()-based parity+traffic tests for the 5 undertested device_born_* flags

Using tests/feature_selection/gpu/test_resident_311_residual_parity.py as the template (env-var toggle + before/after byte-identical/rtol=1e-9 comparison), add explicit tests that flip MLFRAME_FE_GPU_DEVICE_BORN_BINAGG, _DISPERSION, _DUAL_UPLIFT, _WAVELET, and add an explicit toggle to the existing extra_basis test. Wrap each in residency_audit() and assert the specific host-upload site each docstring names (e.g. _binned_numeric_agg_fe.py:360, _extra_fe_families_dispersion.py:563/489) is NOT in the bulk-H2D list when the flag is on. Cheap relative to value: the pattern, the fixtures, and the numeric-tolerance conventions are all already established in this codebase.

### (residency_step) 4. Fix or remove the stale test_wavelet_batched_mi_parity docstring reference

Either restore/rename a test with that exact name that actually toggles MLFRAME_FE_GPU_DEVICE_BORN_WAVELET and pins partition-equivalence, or edit the fe_gpu_device_born_wavelet_enabled docstring to stop citing a nonexistent test. Do this before or alongside item 3 so the wavelet family's coverage claim and its actual coverage agree.

### (residency_step) 5. Wire a full-fit residency_audit() regression test around a real STRICT-resident MRMR.fit() call

Currently the only byte-traffic audit test (test_cmi_residency_traffic.py) covers one FE family that is off by default. Add a test that runs a real MRMR.fit() at n>=100k (clearing the AUTO-STRICT gate in _fe_gpu_strict.py) with MLFRAME_FE_GPU_STRICT=1 wrapped in residency_audit(), and asserts bulk H2D/D2H stays within a small fixed budget (e.g. a handful of per-device operand uploads, not O(rounds*families)). This is the only mechanism that can catch a cross-family or cache-eviction-order leak that no single-family test can see (exactly the class of bug item 1's fix addresses for one specific cache).

### (residency_step) 6. Thread a single per-fit resident y/z handle instead of re-deriving via content hash at every call site

Set the resident y/z cupy handle once at FE-step entry (alongside the existing _gpu_strict_fe.set_auto_fit_n(n, p) call in MRMR.fit) and have the ~9 documented roles in _fe_resident_operands.py / _cmi_cuda.py consume that handle directly instead of re-hashing the host array's content on every call. Converts 'upload once, guaranteed by a cache hit' into 'upload once, guaranteed by construction' -- removes the O(n) hash recompute from the hottest per-candidate-batch path, at the cost of one new caller-side plumbing change (correctly scoped as future work per the file's own docstring note).

### (residency_step) 7. Correct the _fe_gpu_strict.py docstring's incorrect MRMR.fit reachability claim

Update the 'RESIDENCY GAP FOUND AND CLOSED' note (line 40) to reflect that greedy_cmi_fe_construct IS reachable from MRMR.fit (via greedy_cmi_fe_construct_with_recipes when fe_mi_greedy_cmi_enable=True), not benchmark-only. Low effort, prevents a future maintainer from under-prioritizing a real residency regression in that path.

### (residency_step) 8. Port friend_graph_gpu's entropy-from-counts reduction on-device, relaxed to the codebase's standard ~1e-9 tolerance

The largest remaining STRUCTURAL residency gap in this cluster: friend_graph_gpu.py D2Hs raw integer counts every tile specifically to preserve literal CPU bit-identity, a stricter bar than every other GPU/CPU parity claim in the codebase. Write a cupy elementwise+reduction twin of _entropy_from_counts, validate it against the CPU path at the same rtol=1e-9/argmax-match bar test_resident_311_residual_parity.py already uses elsewhere, add a dedicated test_friend_graph_gpu_entropy_device_parity.py, and gate the new fully-resident path behind an opt-in flag until validated -- then flip it default-on. Biggest remaining win in this cluster, but correctly the LAST item because it requires new validation work, not a mechanical change.

### (residency_step) 9. Resolve the free_blocks=True cudaFree/cudaMalloc churn in _fe_gpu_batch/_executor.py with real multi-GPU profiling

The code's own comment says this needs 'live multi-GPU profiling this wave didn't run' before changing the default. Run that profiling (concurrent multi-device gpu_fe_batch_mi calls via multi_gpu_fe_batch_mi's ThreadPoolExecutor, watching peak VRAM understatement risk across devices) and either flip free_blocks default to False with a measured-safe VRAM margin, or keep it True with the measurement now on record instead of deferred.

### (residency_step) 10. Decide the fate of the dead _gpu_strict_fe/_state.py ResidentFEState scaffold

Either delete ResidentFEState / run_fe_step_gpu_strict (and their test) now that the per-family device_born_* mechanism has fully superseded them, or convert the module into an explicit short 'historical/rejected, see device_born_* instead' pointer instead of ~430 lines of live-looking, type-annotated, tested code. Pure cleanup, zero residency impact, but removes a real source of future confusion about which mechanism actually enforces residency.
