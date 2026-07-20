# Info-theory kernels (CPU + GPU) and MI estimator dispatch

10 findings, 7 proposals.

## Findings

### [P1] bug -- src/mlframe/feature_selection/filters/_renyi_alpha.py:109

**_renyi_entropy_from_gram divides by (1-alpha) with no guard against alpha==1.0, and the module's own docstring names alpha=1 as the mathematically singular point -- yet no caller-facing validation exists.**

Empirically verified: renyi_alpha_mi(x, y, alpha=1.0) on x=N(0,1), y=x+0.1*noise (a strongly correlated pair that correctly scores MI~2.23 at the documented default alpha=1.01) returns exactly 0.0 with 3 unlogged 'RuntimeWarning: divide by zero encountered' warnings. Root cause: at alpha=1.0 each per-block entropy S(A)=log2(s)/(1-alpha) hits 0/0-adjacent division -> -inf for Sx, Sy, Sxy; Sx+Sy-Sxy evaluates to -inf + -inf - (-inf) = nan; Python's max(0.0, nan) silently returns 0.0 (nan is never '>' 0.0) instead of raising or propagating the nan. A user or future MRMR integration passing alpha=1.0 (the natural Shannon-limit choice, explicitly discussed in this file's own docstring) silently gets told 'no relationship' for a highly-dependent pair instead of an error or the correct value. Reachable today via score_pair_mi(x, y, estimator='renyi_alpha', estimator_kwargs={'alpha': 1.0}).

### [P1] bug -- src/mlframe/feature_selection/filters/_renyi_alpha.py:109

**renyi_alpha_mi/renyi_alpha_cmi return matrix-based Renyi MI in BITS (log2-based formula), but score_pair_mi's documented contract ('Returns: I(X; Y) in nats') and every other estimator branch (plug_in, mixed_ksg, ksg_lnc, mine, infonet, mist, fastmi, median, genie) return nats -- a silent ~1.4427x unit mismatch for estimator='renyi_alpha'.**

Any caller that follows score_pair_mi's documented nats contract and compares/combines a renyi_alpha score with any other estimator's score (e.g. a fixed absolute-nats significance floor, or a future MRMR per-candidate greedy-loop integration the module docstring says is planned) will silently over-value renyi_alpha-scored candidates by ~44%. test_mi_dispatch_contract.py's FAST_ESTIMATORS list includes 'renyi_alpha' and asserts only relative signal>noise separation, which cannot detect a constant unit-scale error, so this ships green through the existing test suite.

### [P1] test_gap -- tests/feature_selection/mrmr/test_gpu_circuit_breaker_rearm_on_fit.py:68

**test_breaker_reset_is_resilient_to_missing_gpu_modules monkeypatches the wrong reference (the _cmi_cuda module attribute) and therefore never exercises the except-block it claims to regression-test in _rearm_gpu_circuit_breakers.**

_mrmr_class_fit_helpers.py does `from ..info_theory._cmi_cuda import reset_cmi_gpu_circuit_breaker` (import-by-value), so `_rearm_gpu_circuit_breakers()` calls its OWN locally-bound name, not `_cmi_cuda.reset_cmi_gpu_circuit_breaker`. Empirically verified: after `monkeypatch.setattr(_cmi_cuda, 'reset_cmi_gpu_circuit_breaker', _boom)`, `fh.reset_cmi_gpu_circuit_breaker is _boom` is False -- the real (never-raising) function still runs. The test always passes regardless of whether the surrounding `except Exception as exc: logger.debug(...)` in `_rearm_gpu_circuit_breakers` actually works; a regression there (e.g. narrowing the except type, or accidentally re-raising) would go undetected by this suite.

### [P1] cpu_gpu_parity -- src/mlframe/feature_selection/filters/_ksg.py:490

**ksg_mi_dispatch (prefer_gpu=True by default) wraps its GPU call in a bare `except ImportError: pass` only -- unlike every other GPU path in this package (_cmi_cuda, permutation.py mi_direct, _permutation_null_pair_resident pair-maxT), it has no circuit breaker / general-exception fallback for a real runtime GPU fault.**

If cupy IS importable but mixed_ksg_mi_gpu faults at runtime (illegal-address, CUDA OOM, transient driver hiccup -- exactly the fault class that motivated the process-global circuit breakers documented at length in _cmi_cuda.py, citing a real 'wellbore 1515-retry cascade' incident), the exception propagates uncaught out of ksg_mi_dispatch/score_pair_mi and crashes the caller instead of falling back to the CPU mixed_ksg_mi the way every sibling GPU path in this cluster now does. _fastmi.py's optional GPU branch (prefer_gpu default False, lower exposure) has the identical narrow-except gap at line 209.

### [P2] bug -- src/mlframe/feature_selection/filters/info_theory/_cmi_cuda_ktc.py:136

**Module-level `except Exception: _CMI_SPEC = None` (and the narrower excepts at lines 53 and 60) silently disable the entire per-host CMI CPU<->CUDA kernel_tuning_cache crossover -- the mechanism this file exists to implement per the repo's 'never hardcode CUDA thresholds' rule -- with zero log line on any failure.**

A bug in pyutilz.performance.kernel_tuning.registry, a signature mismatch in the kernel_tuner(...) registration call, or any other import/registration-time exception permanently and silently degrades every MRMR fit in the process to the hand-coded bootstrap heuristic (n*p>=1e6 and p>=64) this module is meant to supersede -- undetectable without manually inspecting `_CMI_SPEC is None` in a debugger, since no warning/error is ever logged.

### [P2] bug -- src/mlframe/feature_selection/filters/info_theory/_batch_kernels.py:879

**select_batch_mi_kernel's `except Exception: choice = 'v2'` (line 879) and _run_batch_mi_kernel_sweep's `except Exception: return []` (line 891) are bare swallows with no logging, violating the project's 'WARNING log naming the exception' bar for silent-error classes.**

Any genuine bug in KernelTuningCache.load_or_create().get_or_tune(...) or in the sweep harness (not just a missing optional dependency) is silently absorbed; the kernel selection quietly falls back to the hardcoded 'v2' default with no diagnostic trail, so a real regression in the per-host tuning path for the batched FE-candidate MI kernel would never surface in logs.

### [P2] test_gap -- tests/feature_selection/mrmr/core/test_mrmr_concurrency_fixes.py:31

**The fit-count-gated GPU circuit-breaker re-arm logic (_enter_active_fit_scope/_ACTIVE_FIT_COUNT, the exact mechanism the cluster brief asked to verify under 'real concurrency') is tested only via sequential simulated calls on bare mixin instances, never via genuinely concurrent threading.Thread execution.**

_rearm_gpu_circuit_breakers() is invoked OUTSIDE the _ACTIVE_FIT_COUNT_LOCK after the 0->1 transition is decided (_mrmr_class_fit_helpers.py lines 78-83), so a second thread's _enter_active_fit_scope() (a genuine 1->2 transition, correctly skipping its own rearm) can begin GPU work before the first thread's rearm call has actually cleared _CMI_GPU_FAILED/_MI_DIRECT_GPU_FAILED/_PAIR_MAXT_GPU_FAILED. No test exercises this interleaving, so the docstring's claim that the gating 'prevents cross-fit un-poisoning' under real thread concurrency is asserted but not demonstrated; today this window only costs a few spurious CPU-fallback calls (self-healing, not a correctness bug), but a future refactor that assumes the rearm is synchronous-with-the-lock would not be caught by any existing test.

### [P2] design -- src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py

**GPU residency in this cluster is clean: no wasteful host<->device round trips found.**

No issues found in this cluster for this angle -- a targeted grep of every file for cp.asnumpy/.get()/copy_to_host/synchronize() across the whole cluster turns up only load-bearing final-result extractions (the fused CMI kernel's terminal cp.asnumpy(cmi_g), _ksg.py's terminal .get() after the GPU eps-radius count, _fastmi.py's terminal cp.asnumpy(density_gpu) after FFT). _cmi_cuda.py additionally ships three purpose-built, weakref-identity-guarded resident caches (y/z upload cache, whole-factors_data device-resident cache with on-device candidate gather, F-order CPU view cache) specifically to eliminate this class of redundant PCIe traffic across the MRMR greedy loop -- already a mature implementation, nothing further to flag.

### [P2] design -- src/mlframe/feature_selection/filters/info_theory/_entropy_kernels.py

**Bias-direction and >=0 clamping conventions are consistently correct across every estimator variant in this cluster.**

No issues found in this cluster for this angle -- Miller-Madow (entropy corrected UP via +(k-1)/2n, MI corrected DOWN via the closed-form product-of-occupied-bins term), Chao-Shen (coverage-adjusted entropy, MI floored at 0), PID/BUR/JMIM/RelaxMRMR/group-MI (all floor their final scores at 0 with the correct sign on every subtraction), and CSU/SU (numerator floored before the ratio) were all spot-checked against their cited references and found directionally correct and consistently clamped; no sign-flip or missing-floor bugs found.

### [P2] test_gap -- src/mlframe/feature_selection/filters/_renyi_alpha.py

**No test covers _renyi_alpha.py's compute-cost guard boundary: a caller-supplied max_n far above the documented default (1500) triggers an unbounded O(n^3) dense eigendecomposition with no sanity cap or clear error.**

score_pair_mi(x, y, estimator='renyi_alpha', estimator_kwargs={'max_n': 50000}) on n>=50000 data silently attempts np.linalg.eigvalsh on a ~50000x50000 dense Gram matrix (tens of GB, effectively hangs/OOMs) rather than raising a clear error -- unlike _ksg.py's max_input_n and _neural_mi.py's max_input_n, which are the same class of guard but this module has no test pinning any upper bound behavior.

## Proposals

### (edge_case) Guard alpha near/at 1.0 in _renyi_alpha.py

Add an explicit check in _renyi_entropy_from_gram (or the two public entry points) that rejects or nudges alpha away from 1.0 (e.g. raise ValueError, or clamp to 1.0 +/- 1e-6 with a logged warning) instead of letting the 0/0-adjacent division silently produce -inf/nan that max(0.0, ...) then swallows into a misleading 0.0.

### (other) Fix the bits-vs-nats unit mismatch for estimator='renyi_alpha'

Either convert _renyi_entropy_from_gram's output to nats (multiply by ln(2), or use natural-log Rényi entropy directly) so it matches score_pair_mi's documented 'in nats' contract and every sibling estimator, or explicitly document renyi_alpha as bits-scaled and exclude it from any future direct comparison/aggregation with nats-based estimators (median/genie panels, absolute significance floors).

### (coverage_gap) Fix the ineffective monkeypatch in test_breaker_reset_is_resilient_to_missing_gpu_modules

Patch the name actually consulted at the call site -- `monkeypatch.setattr(fh, 'reset_cmi_gpu_circuit_breaker', _boom)` on the _mrmr_class_fit_helpers module (or monkeypatch all three imported-by-value reset functions there) -- so the test genuinely exercises the try/except resilience path in _rearm_gpu_circuit_breakers instead of always running the real, never-raising implementation.

### (coverage_gap) Add a genuinely concurrent (threading.Thread) test for the GPU circuit-breaker fit-count gate

Spawn a real background thread that trips _CMI_GPU_FAILED mid-fit (e.g. via a monkeypatched GPU call that raises once), start a second concurrent fit() from the main thread while the first is still in-flight, and assert the second fit still observes the tripped breaker (routes to CPU) rather than being silently un-poisoned -- closing the gap between the current sequential-simulation test and the real concurrency the mechanism is designed for.

### (coverage_gap) Add a units/scale regression test for renyi_alpha against a known-nats analytic MI

Use the Gaussian-copula analytic MI formula (-0.5*log(1-rho^2), already used as ground truth for MIST calibration in _neural_mi.py) to pin renyi_alpha_mi's output against the EXPECTED nats scale at a known correlation, which would have caught the bits/nats mismatch immediately.

### (edge_case) Add an upper sanity cap on max_n in _renyi_alpha.py

Mirror _ksg.py's max_input_n / _neural_mi.py's max_input_n pattern: raise a clear ValueError (or log+clamp) when a caller-supplied max_n would make the O(n^3) eigendecomposition exceed a documented memory/time budget, instead of silently attempting it.

### (other) Add logging to the bare except-Exception swallows in _batch_kernels.py and _cmi_cuda_ktc.py

Add a logger.debug/warning naming the caught exception in select_batch_mi_kernel, _run_batch_mi_kernel_sweep, and the three silent excepts in _cmi_cuda_ktc.py (lines 53, 60, 136) so a genuine registration/tuning bug leaves a diagnostic trail instead of silently degrading to the bootstrap heuristic forever.
