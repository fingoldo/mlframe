# GPU residency / dispatch infrastructure

11 findings, 5 proposals.

## Findings

### [P0] gpu_residency -- src/mlframe/feature_selection/filters/batch_pair_mi_gpu.py:372

**dispatch_batch_pair_mi / dispatch_batch_pair_mi_chunked never check MLFRAME_DISABLE_GPU or CUDA_VISIBLE_DEVICES="" (the documented single mlframe GPU opt-out convention); they gate purely on _CUPY_AVAIL/_CUDA_AVAIL, which are booleans memoized once at import time from a raw `import cupy` / numba probe.**

A user sets MLFRAME_DISABLE_GPU=1 (the project's documented "force CPU" convention, per _gpu_policy.py's own docstring which calls the un-honored case "catastrophic") before an MRMR fit that reaches the FE pair-search (_mrmr_fe_step/_step_pairmi.py calls dispatch_batch_pair_mi_chunked with no opt-out check of its own). _CUPY_AVAIL stays True (cupy import succeeded at process start) so dispatch_batch_pair_mi still routes to batch_pair_mi_cupy/batch_pair_mi_cuda on a large-enough shape, silently ignoring the user's explicit opt-out -- the exact failure mode _gpu_resident_fe.py/_gpu_resident_basis.py and the sibling _pairs_dispatch._dispatch_batch_mi_with_noise_gate were specifically patched to prevent (the latter has a dedicated regression test, test_noise_gate_gpu_optout_regression.py, that this file has no equivalent of).

### [P0] gpu_residency -- src/mlframe/feature_selection/filters/friend_graph_gpu.py:691

**dispatch_friend_graph_stats has the identical gap as batch_pair_mi_gpu: no MLFRAME_DISABLE_GPU / CUDA_VISIBLE_DEVICES="" check anywhere in the file, and its only production caller (friend_graph.py:402) doesn't check either -- only raw _CUDA_AVAIL/_CUPY_AVAIL (memoized at import) gate the choice.**

MLFRAME_DISABLE_GPU=1 set before a fit that reaches friend-graph debiasing statistics (friend_graph.py -> dispatch_friend_graph_stats) still dispatches to the cupy/cuda backend on a large-enough shape, exactly reproducing the incident _gpu_policy.py was created to fix, but in a file that incident's fix never reached.

### [P1] gpu_residency -- src/mlframe/feature_selection/filters/gpu.py:461

**mi_direct_gpu declares a `use_gpu: bool = True` parameter that is never referenced anywhere in the function body -- a dead/no-op parameter. The function also never checks MLFRAME_DISABLE_GPU / CUDA_VISIBLE_DEVICES itself.**

Any caller (existing or future) that calls mi_direct_gpu(..., use_gpu=False) expecting CPU execution (the obvious, standard meaning of that parameter name elsewhere in the codebase, e.g. feature_engineering/transformer/random_features.py's real "use_gpu=False -> CPU regardless" contract) gets silently routed through the GPU code path anyway (`import cupy as cp` executes unconditionally). Verified by grep: `use_gpu` appears only once in gpu.py (the signature line); every existing caller happens to gate the *decision to call* mi_direct_gpu externally before invoking it (e.g. _confirm_predictor.py inlines its own CUDA_VISIBLE_DEVICES/MLFRAME_DISABLE_GPU check before calling), so today's callers are accidentally safe, but the public parameter itself is a footgun. Additionally, when reached via permutation.py's screen (`prefer_gpu and npermutations>=32`) the only live gate is pyutilz's is_cuda_available() (confirmed via reading its source: wraps `numba.cuda.is_available()`, which honors CUDA_VISIBLE_DEVICES but has no knowledge of the mlframe-specific MLFRAME_DISABLE_GPU convention), so MLFRAME_DISABLE_GPU=1 alone does not stop this call path either.

### [P2] test_gap -- tests/feature_selection/gpu/test_noise_gate_gpu_optout_regression.py

**The only regression test pinning the MLFRAME_DISABLE_GPU/CUDA_VISIBLE_DEVICES opt-out contract covers exclusively the noise-gate dispatcher (_pairs_dispatch.py, outside this cluster); no equivalent test exists for batch_pair_mi_gpu.dispatch_batch_pair_mi, friend_graph_gpu.dispatch_friend_graph_stats, or gpu.mi_direct_gpu despite all three living in this cluster and sharing the identical gap (findings above).**

Because this exact bug class was hit once in production and fixed for one dispatcher with a named regression test, but never generalized to a suite-wide check or the other GPU dispatch entry points, the P0 findings above shipped and would keep shipping through CI silently: nothing in the existing 79-file gpu test suite would fail if a future change further widened this gap.

### [P2] design -- src/mlframe/feature_selection/filters/_gpu_policy.py:21

**gpu_globally_disabled()'s docstring claims to be "the single source of truth every GPU dispatch should consult," but grep confirms zero files in this cluster (gpu.py, batch_pair_mi_gpu.py, friend_graph_gpu.py, batch_mi_noise_gate_gpu.py, _gpu_batched.py, _gpu_pairs.py, all _gpu_resident_*.py, all _batch_*_cuda_*.py, all _resident_*.py) call it -- it is only consulted by files outside the cluster (_permutation_null.py, _usability_gpu.py, _hermite_fe_mi.py, etc).**

Two of the cluster's files (_gpu_resident_fe.py, _gpu_resident_basis.py) reimplement the same two env checks inline via _env_gpu_default_on rather than calling the shared function; if the policy's semantics are ever extended (e.g. a new opt-out mechanism, a per-device allowlist), every inline duplicate -- and every dispatcher with no check at all (findings 1-3) -- silently keeps the old, narrower behavior. This is the root architectural cause underlying findings 1-3, not just an isolated typo.

### [P2] design -- src/mlframe/feature_selection/filters/_gpu_resident_fe.py:1

**gpu_resident_pair_candidate_mi, gpu_resident_pair_candidate_mi_fast, grand_fused_pair_mi, grand_fused_pair_mi_fused, gpu_resident_pair_recipes, and pair_candidate_mi_dispatch (spread across _gpu_resident_fe.py and _gpu_resident_basis.py) are self-documented in the module docstring as "prototype (gated, un-wired)" and "imported by nothing in the production FE path." Confirmed via grep: their only callers are each other, tests, and one standalone benchmark script (_benchmarks/bench_grand_fusion_scaling.py) -- never MRMR.fit or any production FE call path.**

Not a runtime bug (the module's own "PROTOTYPE-ONLY" note at lines 79-90 explains why: production's candidate-MI needs the (n,K) float matrix materialized for downstream survivor/usability reads, which the grand-fusion recompute-not-store design structurally cannot provide), but it is ~450 LOC of dead, heavily-optimized, heavily-tested scaffolding from an abandoned residency push living in the audit's most complex file (1599 LOC, over the project's own 1k-LOC carve threshold partly because of this dead weight) with no code marker (deprecation warning, `_UNUSED`/`_prototype` prefix, or test skip) signaling it to a future reader who might otherwise treat its "DEFAULT ON" env flags (e.g. MLFRAME_FE_GPU_GRAND_FUSION=1) as meaningful in production.

### [P2] design -- src/mlframe/feature_selection/filters/_gpu_strict_fe/_state.py:1

**ResidentFEState (the class this whole subpackage's __init__.py re-exports as its headline public name) and _entry.py's run_fe_step_gpu_strict (a Phase-0 stub that unconditionally raises NotImplementedError) are self-documented as SUPERSEDED: "ResidentFEState/run_fe_step_gpu_strict Phases 1-3 were never implemented ... that alternate mechanism [the fe_gpu_device_born_*_enabled predicates] is live ... ResidentFEState itself has zero production callers." Confirmed via grep: ResidentFEState appears only in its own definition, the _entry.py docstring explaining it is dead, and the __init__.py re-export.**

This directly answers the cluster's specific audit question ("is the _gpu_strict_fe audit/entry/state machinery actually enforced everywhere GPU-strict mode claims to guarantee residency?"): the class that GAVE the subpackage its name and public __init__ export is confirmed-dead scaffolding; the REAL residency enforcement for MLFRAME_FE_GPU_STRICT lives entirely in the ten fe_gpu_device_born_*_enabled predicates in _entry.py (each independently opt-out-able, each falling back to the host path on any failure per-family) -- a reader following the __init__.py export list would reasonably but wrongly conclude ResidentFEState is the mechanism to inspect/extend.

### [P2] gpu_residency -- src/mlframe/feature_selection/filters/_gpu_strict_fe/_audit.py:47

**The GPU-strict residency contract's only enforcement mechanism (residency_audit(), which monkeypatches cp.asarray/cp.asnumpy/cp.ndarray.get to tally bulk vs scalar transfers) is confirmed via grep to be invoked ONLY from 3 test files -- never from any production code path, CI-wide gate, or fit-time assertion.**

Each of the ten fe_gpu_device_born_*_enabled families' "selection-equivalent, zero-bulk-D2H" residency guarantee is validated only by whichever specific pinned test (e.g. test_device_born_cross_basis_parity.py) happens to exercise that family under residency_audit(); there is no blanket assertion that ALL device-born families stay resident simultaneously on a real multi-family fit. A future edit that reintroduces a bulk transfer in an untested code path (or an interaction between two families not covered by any single existing test) would silently regress performance without any test failing -- the audit finding the cluster prompt specifically asked for ("is it actually enforced everywhere, or are there silent fallback gaps") is: enforcement exists but is opt-in-per-test, not blanket.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_cupy_polynom_optimizer.py:176

**run_cupy_kernel_search has zero test coverage anywhere in the repository (confirmed via grep across tests/) despite its own module docstring stating it "is the DEFAULT since 2026-07-15" optimizer for single-pair hermite-FE polynomial search (production wellbore-100k validation cited inline). It also lacks input validation: if a caller supplies batch_size <= elitism_k (both public kwargs), the generation loop computes a negative n_perturb.**

At line 248, `n_perturb = batch_size - elitism_k - max(1, batch_size // 4)`; with e.g. batch_size=8, elitism_k=10, n_perturb=-4, and `rng.integers(0, elitism_k, size=n_perturb)` (line 249) raises `ValueError: negative dimensions are not allowed`. Even short of that, batch_size < elitism_k makes `elites = pop[order[:elitism_k]]` (line 247) silently return fewer than elitism_k rows, so the same rng.integers call can draw an out-of-range index into elites and raise IndexError. Not reachable today (the sole call site, _hermite_fe_optimise_pair.py:631, never overrides the batch_size=100/elitism_k=10 defaults), but the public function offers no guard, and being the reported default optimizer with zero tests means a regression here (or a future caller/config surface that does vary batch_size) has no safety net at all -- not even a smoke test.

### [P2] cpu_gpu_parity -- src/mlframe/feature_selection/filters/_gpu_resident_basis.py:113

**No open CPU/GPU parity issues found in this cluster's core kernels beyond one already-fixed, self-documented historical case: the Laguerre GPU Clenshaw recurrence was previously WRONG (L_2(0) gave -0.5 vs the correct 1) and went uncaught because the canonical prewarp path defaults to the Chebyshev basis, so no production pin exercised Laguerre until test_gpu_basis_column_parity was added specifically to sweep all 4 bases.**

No current failure -- the fix (forward recurrence matching the host _lagval_njit) is in place and pinned by test_gpu_basis_column_parity. Flagged per the audit's explicit instruction to report a clean angle explicitly: the cluster's CPU/GPU parity discipline (maxdiff-0 pins, dual-kernel-retention, selection-equivalence documentation) is broadly excellent across the ~20 kernel files read/scanned, and this historical near-miss is the evidence for why (a per-basis/per-formula sweep test, not a single-basis smoke test, is what actually caught the class of bug this audit angle is looking for).

### [P2] bug -- src/mlframe/feature_selection/filters/_batch_pair_mi_cuda_kernels.py:594

**Cluster-wide sweep of bare `except Exception:` blocks (dozens found): the large majority are legitimate, commented best-effort hardware-probe fallbacks (matching project convention), but a minority silently pick a conservative default with no debug/warning log at all -- e.g. this file's free-VRAM probe (line 594, falls back to a hardcoded 512MB with no log), its threads-per-block probe (line 120, silently returns 128), its shared-mem-budget probe (line 357, silently returns 49152), and batch_mi_noise_gate_gpu.py's threads_per_block fallback (line 552-553, silently sets 128).**

All of these are perf-tuning fallbacks (a wrong guess only picks a slower/more-conservative kernel launch config, never a wrong numeric answer), so this is not a correctness bug, but the total silence (no logger.debug call, unlike most of this same file's other except blocks which do log) means a persistently-failing hardware probe on some host would never surface in logs even at DEBUG level, making a real perf regression on that host undiagnosable without reading source.

## Proposals

### (residency_step) Wire the shared GPU opt-out check into every dispatch entry point in this cluster

Add a `if gpu_globally_disabled(): return <cpu path>` (or equivalent inline MLFRAME_DISABLE_GPU / CUDA_VISIBLE_DEVICES check, matching _pairs_dispatch.py's already-fixed pattern) at the top of batch_pair_mi_gpu.dispatch_batch_pair_mi, batch_pair_mi_gpu.dispatch_batch_pair_mi_chunked, friend_graph_gpu.dispatch_friend_graph_stats, and gpu.mi_direct_gpu (also make it consult its own currently-dead `use_gpu` parameter). This closes findings 1-3 in one consistent pass and retires the duplicated inline checks in _gpu_resident_fe.py/_gpu_resident_basis.py in favor of the shared _gpu_policy.gpu_globally_disabled() function.

### (coverage_gap) Generalize test_noise_gate_gpu_optout_regression.py's pattern to the other 3 GPU entry points

Add sibling regression tests (spy on the GPU-backend function, assert it's never entered under CUDA_VISIBLE_DEVICES="" and MLFRAME_DISABLE_GPU=1) for dispatch_batch_pair_mi, dispatch_friend_graph_stats, and mi_direct_gpu, so the bug class this audit found cannot silently regress or reappear.

### (other) Mark or remove the confirmed-dead GPU-resident scaffolding

gpu_resident_pair_candidate_mi(_fast), grand_fused_pair_mi(_fused), gpu_resident_pair_recipes, and pair_candidate_mi_dispatch (_gpu_resident_fe.py / _gpu_resident_basis.py, ~450 LOC) and ResidentFEState / run_fe_step_gpu_strict (_gpu_strict_fe/_state.py + _entry.py) are both self-documented dead code with zero production callers. Either delete them (git history preserves them per the project's REJECTED != DELETED convention only for validated-but-rejected perf attempts, not abandoned architectural forks) or add an explicit `# UNREACHED FROM PRODUCTION -- see <doc>` module-level marker plus a `pytest.mark.skip`-free but clearly-labeled test file so a future reader doesn't mistake DEFAULT-ON env flags like MLFRAME_FE_GPU_GRAND_FUSION for something production actually exercises.

### (edge_case) Validate batch_size vs elitism_k in run_cupy_kernel_search

Add `elitism_k = min(elitism_k, max(1, batch_size - 1))` (or raise a clear ValueError) at the top of run_cupy_kernel_search before the generation loop, plus a unit test at batch_size below elitism_k to pin the fix, and at least one parity/smoke test against run_numba_kernel_search given this is reportedly the default optimizer in production.

### (coverage_gap) Add a blanket cross-family residency assertion for MLFRAME_FE_GPU_STRICT

Beyond the per-family pinned tests, add one test that runs a real multi-family STRICT fit under _gpu_strict_fe._audit.residency_audit() and asserts zero bulk_h2d/bulk_d2h across the WHOLE fit (not one family in isolation), closing the gap between "enforcement exists per-test" and "enforcement is guaranteed end-to-end."
