# Cross-cutting audit: process-global state that latches into a degraded mode

**Date:** 2026-09-05
**Scope:** `src/mlframe` (2466 files), READ-ONLY.
**Method:** AST scan for the six shapes (module-level `_CACHE`/`_AVAILABLE`/`_BACKEND` flags written under `global`
inside/around `try`; broad `except Exception` handlers that assign a module global; `lru_cache`/`cache` decorators;
`threading.local()` / `ContextVar` construction; `__getstate__` vs runtime attrs on `self`; global RNG seeding),
then manual reads of ~95 candidate files. Additionally cross-checked every `reset_*` / `clear_*` for a real
non-test call site, and every `id()`-keyed cache for a liveness/content guard.

**Headline:** this codebase has clearly already been through this bug class. The three confirmed historical
instances (`_select_mi_backend`, the kernel-tuning singleton, `_fe_deadline`) are all fixed, with the reasoning
documented in-line, and most `id()`-keyed device caches carry weakref or content-hash guards. The remaining
findings are (a) one genuine re-introduction of the `_fe_deadline` leak in a **loky worker**, (b) three GPU
availability probes that still latch on a broad exception at `debug` level with no reset path, and (c) two
circuit breakers left out of the fit-entry re-arm their three siblings get.

---

### LATCH-01 [P1] fe-deadline-republished-in-loky-worker-never-cleared
**File:** src/mlframe/feature_selection/filters/polynom_pair_fe.py:388
**Summary:** `_eval_one_pair_impl` re-publishes MRMR's FE wall-clock deadline into the worker's thread-local
(`set_fe_deadline(fe_deadline)`) because the deadline cannot cross the process boundary, but there is no
matching `clear_fe_deadline()` on any exit path: no `try/finally`, no clear after the seed loop. The
`_fe_deadline` thread-local latches into "budget already expired" inside the worker.
**Failure scenario:** TRIGGER - any MRMR fit with `max_runtime_mins` set that runs the polynom-pair FE step.
`run_polynom_pair_fe` dispatches with `Parallel(n_jobs=..., backend=<loky>)` (polynom_pair_fe.py:507-508; the
comment at :446-452 records the deliberate switch from `threading` to `loky`). loky **reuses** its worker
processes across `Parallel(...)` invocations, so the absolute `timer()` timestamp written at :388 survives in
that worker after the function returns. BLAST RADIUS - for the rest of that loky worker's life (across
subsequent FE steps and across subsequent `MRMR.fit()` calls, including ones that pass no budget at all), every
`fe_deadline_passed()` consumer executing in that worker returns True once the stale timestamp elapses, silently
truncating enrichment loops that were never given a budget. The main-thread guard added for the original bug
(`_mrmr_class.py:4001`, `_clear_fe_deadline()` in the `finally`) does not reach worker processes.
**Evidence:** polynom_pair_fe.py:385-392 (import + `set_fe_deadline`, then straight into the `for seed_offset`
loop); no `clear_fe_deadline` anywhere in the file (grep over the file returns nothing);
polynom_pair_fe.py:446-452 + :507-508 confirm the loky (process-pool, worker-reusing) backend;
`_fe_deadline.py:29` documents this exact defect as the original regression.
**Suggested fix:** clear on exit - wrap the `if fe_deadline is not None: set_fe_deadline(fe_deadline)` block and
the seed loop in `try/finally: clear_fe_deadline()`, mirroring `_mrmr_class.py:4001`. A `contextmanager` in
`_fe_deadline.py` (`with fe_deadline_scope(fe_deadline):`) would make the pairing un-droppable for future
worker-side republishers.

---

### LATCH-02 [P1] gpu-metrics-availability-latched-on-broad-except
**File:** src/mlframe/metrics/_gpu_metrics.py:72
**Summary:** `is_gpu_metrics_available()` memoises `_GPU_AVAILABLE` for the process lifetime. The probe body
catches bare `Exception` and writes `False`, logging only at `debug`. There is no reset/re-probe entry point in
the module (unlike `_utils.reset_gpu_probe`).
**Failure scenario:** TRIGGER - the probe does `cp.cuda.runtime.getDeviceCount()` **and** an NVRTC compile
(`cp.asarray([1.0]).sum().item()`). Both raise on transient conditions that are not facts about the machine: a
concurrent process holding the device, a CUDA OOM at probe time, a WDDM TDR reset, a driver hiccup. The first
one to land during the first metrics call in the process wins. BLAST RADIUS - process lifetime. Every
`compute_batch_aucs` / `compute_batch_rmse` dispatch falls to CPU for the rest of the run; this module's own
header records that path as ~32s of a ~55s suite wall on a 1M-row binary run, i.e. the loss is the same order as
the historical `_select_mi_backend` regression, and equally invisible (debug).
**Evidence:** _gpu_metrics.py:53 (`_GPU_AVAILABLE: Optional[bool] = None`), :95-127 (probe thread, bare
`except Exception` -> `result["available"] = False`, then `_GPU_AVAILABLE = bool(...)`); no `reset` function in
the module. Contrast with `feature_engineering/transformer/_utils.py:127` which does expose `reset_gpu_probe()`.
**Suggested fix:** narrow the catch. `ImportError` (cupy absent) is the only genuinely permanent verdict and
should latch; anything else should be logged at `warning` and re-probed on the next call under a bounded attempt
counter, exactly the discipline `_kernel_tuning.get_kernel_tuning_cache` (`_MAX_INIT_ATTEMPTS`) and
`_cb_pool._cb_gpu_usable` (`_CB_GPU_PROBE_ATTEMPTS`) already use in this repo. Keep the existing hang/timeout
branch latched as-is: a 20s hang per call is a real cost and that branch is correctly reasoned.

---

### LATCH-03 [P1] metrics-argsort-gpu-availability-latched-on-broad-except
**File:** src/mlframe/metrics/_core_auc_brier.py:126
**Summary:** `_gpu_argsort_available()` caches `_GPU_ARGSORT_AVAILABLE` once per process; the probe wraps
`cp.cuda.runtime.getDeviceCount()` in a bare `except Exception` -> `False`, `logger.debug` only. No reset path.
**Failure scenario:** TRIGGER - a `CUDARuntimeError` from `getDeviceCount()` while another process momentarily
holds the card, or during a driver reset, on the first large-N metric call. BLAST RADIUS - process lifetime:
every metric argsort at `N >= _GPU_ARGSORT_MIN_N` (50k) stays on CPU. The file's own A/B header quantifies the
GPU path as a consistent ~10% end-to-end win at 200k (CPU 9.37/8.29s vs GPU 7.98/7.89s), so this silently gives
back a measured, deliberately-tuned win with a debug-level trace as the only evidence.
**Evidence:** _core_auc_brier.py:42 (`_GPU_ARGSORT_AVAILABLE: "bool | None" = None`), :126-136 (probe, broad
except, `_GPU_ARGSORT_AVAILABLE = False`), :140-146 (the consumer gate).
**Suggested fix:** narrow to `ImportError` for the latch; re-attempt on other exceptions (bounded), warn on the
first non-ImportError. Same shape of fix as LATCH-02.

---

### LATCH-04 [P1] cluster-su-gpu-availability-latched-on-broad-except
**File:** src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_cluster_su.py:64
**Summary:** `cluster_su_gpu_available()` caches `_GPU_AVAILABLE_CACHE` for the process. The whole probe
(`import cupy` + `getDeviceCount()` + a 4-element alloc round-trip) sits under one bare `except Exception` that
writes `False` at `debug`. No reset function.
**Failure scenario:** TRIGGER - the tiny `cp.zeros(4)` allocation or its `.sum().get()` faulting because another
mlframe stage (or another process) has the VRAM at that instant: a momentary allocation failure, not a missing
device. BLAST RADIUS - process lifetime: the entire ShapProxiedFS cluster-SU pair loop stays on CPU for every
later `fit()` in that process. The docstring itself justifies the cache by the cost of the *first* probe, which
argues for retrying rather than latching a failure.
**Evidence:** _shap_proxy_cluster_su.py:61 (`_GPU_AVAILABLE_CACHE: bool | None = None`), :64-92 (probe;
`except Exception` -> `logger.debug(...)`, `_GPU_AVAILABLE_CACHE = False`). Sibling
`_shap_proxy_prefilter.gpu_model_available` in the same package already narrows to `ImportError` (:208, :229)
and ships `reset_gpu_model_available_cache()` (:251) - this one does neither.
**Suggested fix:** narrow the catch. Latch only on `ImportError` / `getDeviceCount() <= 0`; treat an allocation
or kernel fault as transient, warn once and re-probe next call. Add a `reset_cluster_su_gpu_cache()` to match the
package's own `_shap_proxy_prefilter` / `_shap_proxy_catboost` precedent.

---

### LATCH-05 [P1] two-gpu-circuit-breakers-omitted-from-the-fit-entry-rearm
**File:** src/mlframe/feature_selection/filters/mrmr/_mrmr_class_fit_helpers.py:91
**Summary:** `_rearm_gpu_circuit_breakers()` re-arms exactly three of the five process-global GPU circuit
breakers in this package on the 0->1 in-flight-fit transition. `_ksg._KSG_GPU_FAILED` and
`_permutation_null_resident._ORDER1_MAXT_GPU_FAILED` are not re-armed, so they keep the *un-fixed* behaviour the
docstring at :93-100 explicitly calls out as the defect.
**Failure scenario:** TRIGGER - one CUDA OOM / context fault inside `mixed_ksg_mi_gpu` (`_ksg.py:521`) or inside
the order-1 max-T resident null (`_permutation_null_resident.py:46`), e.g. VRAM contention from a concurrent
stage. BLAST RADIUS - the rest of the process, spanning every LATER fit in a long-lived worker (notebook,
service, CV loop): precisely the scope `_rearm_gpu_circuit_breakers` exists to bound to one fit. KSG MI at
`n >= _KSG_GPU_THRESHOLD` (50k default) and the order-1 max-T null both fall to their CPU floors permanently.
**Evidence:** `_mrmr_class_fit_helpers.py:101-112` calls exactly `reset_cmi_gpu_circuit_breaker`,
`reset_mi_direct_gpu_circuit_breaker`, `reset_pair_maxt_gpu_circuit_breaker`. A repo-wide grep for
`reset_ksg_gpu_circuit_breaker` and `reset_order1_maxt_gpu_circuit_breaker` finds only the definitions, the
`__all__` entries and test call sites: no production caller. Flags at `_ksg.py:45` and
`_permutation_null_resident.py:40`.
**Suggested fix:** re-attempt - add both resets to `_rearm_gpu_circuit_breakers()` with the same
`try/except -> logger.debug` wrapper the other three use. Better still, have each breaker module register itself
in a shared list so a sixth breaker cannot be forgotten the same way.

---

### LATCH-06 [P2] suite-wide-overrides-applied-long-before-their-restore-snapshot-is-recorded
**File:** src/mlframe/training/core/_phase_config_setup.py:208
**Summary:** `setup_configuration` flips four process/thread-wide overrides (residual-audit reporting at :208,
inline display at :229, format subfolders at :246, calibration colormap at :261) and records the prior values
into `ctx.artifacts["_process_flag_prior_*"]` only at :539-545. Nothing spans the two points, so an exception in
between leaves the flags flipped with **no snapshot in existence** for `restore_process_flags` to act on.
**Failure scenario:** TRIGGER - any raise inside the ~280 lines of setup between :208 and :539 (a bad
`feature_handling_config`, a chart-registry import failure, a validation error on a downstream config block).
BLAST RADIUS - the thread's lifetime: `_set_residual_audit_enabled` and the two `renderers.save` thread-locals
and `reporting.colors` stay on the failed suite's values for every later caller on that thread. In a test
process that is every later test, which is the exact symptom `_process_flag_scope.py`'s module docstring records
for the previous instance of this defect ("unrelated later tests failing on a setting nobody in them chose").
**Evidence:** _phase_config_setup.py:207-208, :224-231, :239-246, :251-262 (all four sets), :539-545 (the
snapshot writes); the only `try:` statements in that span are narrow per-import guards, none wrapping the
set-to-record span. `_process_flag_scope.py:8-12` documents the class; `capture_process_flag_snapshot` (:26)
already hardens the *later* half of the chain (a phase rebuilding `ctx.artifacts` wholesale), which shows the
handoff is known to be fragile.
**Suggested fix:** clear on exit - record each prior into `ctx.artifacts` **immediately before** its
corresponding set call, rather than in a batch at the end. That makes the snapshot exist for every path that
could have flipped the flag, and it is a pure reordering with no behaviour change on the success path.

---

### LATCH-07 [P2] lru-cache-over-env-vars-and-the-on-disk-kernel-tuning-cache
**File:** src/mlframe/models/ensembling/member_metrics.py:23
**Summary:** `_per_member_use_numba` is `@lru_cache(maxsize=256)` keyed on `(elements_per_member, n_groups,
ndim)`, but its body reads two environment variables (`MLFRAME_PER_MEMBER_BACKEND`,
`MLFRAME_PER_MEMBER_AUTOTUNE`) and the per-host `KernelTuningCache` file - none of which are in the key - and
ends with a bare `except Exception` that returns the hand-heuristic fallback. Both a stale env read and a
transient tuning-cache failure are memoised permanently for that key. This is shape 3 and shape 2 in one
function.
**Failure scenario:** TRIGGER - `KernelTuningCache.load_or_create().get_or_tune(...)` raising for one of the
momentary reasons `_kernel_tuning.py:29-31` enumerates for the identical constructor (the tuning JSON being
rewritten by a concurrent sweep, a Windows file lock from another mlframe process, a transient nvidia-smi
fault), on the first call at a given shape. BLAST RADIUS - process lifetime for that `(elements, n_groups,
ndim)` key: the measured per-host verdict is discarded and the element-count heuristic
(`>= _PER_MEMBER_NUMBA_FLOOR_ELEMENTS`) is used instead, logged at `debug`. The docstring quantifies the
backends as 5-18x apart across the 2-D regime, so a wrong verdict is not cosmetic. A second-order effect: an env
override set after the first call for a given shape is silently ignored.
**Evidence:** member_metrics.py:23-24 (decorator + signature), :60-63 (env reads inside the cached body), :69-82
(`get_or_tune`), :89-91 (`except Exception` -> `logger.debug` -> heuristic fallback).
`_kernel_tuning.py:29-31` and :110-134 document the same failure modes as transient and retry them.
**Suggested fix:** narrow the catch and re-attempt - do not memoise the exception path. Either fold the env
values into the cache key, or (better, matching `_kernel_tuning`) keep the `lru_cache` for the success path only
and route failures through a bounded-retry helper so a later call can still pick up the real tuning verdict.

---

### LATCH-08 [P2] numba-cuda-metrics-probe-latched-on-broad-except
**File:** src/mlframe/metrics/_gpu_metrics.py:129
**Summary:** `_is_numba_cuda_available()` caches `_NUMBA_CUDA_AVAILABLE` once per process; a bare
`except Exception` around `from numba import cuda; cuda.is_available()` falls through to `False` at `debug`.
**Failure scenario:** TRIGGER - `numba.cuda.is_available()` raising on a transient device condition. This is the
*identical* call the codebase has already hardened twice: `training/_gpu_probe.py:27-34` stays optimistic and
warns, and `_fe_gpu_strict._retry_cuda_available` (:196-215) retries a bounded number of times, both with
comments naming it a transient condition rather than evidence about the machine. This site does neither. BLAST
RADIUS - process lifetime: the numba-CUDA RMSE fast path is disabled and the cupy `ReductionKernel` fallback (or
CPU) is used for every later batch-RMSE call.
**Evidence:** _gpu_metrics.py:57 (`_NUMBA_CUDA_AVAILABLE: Optional[bool] = None`), :129-142; contrast
`_gpu_probe.py:27-34` and `_fe_gpu_strict.py:191-215`.
**Suggested fix:** narrow the catch to `ImportError` and reuse the bounded-retry pattern already written in
`_fe_gpu_strict._retry_cuda_available` rather than a third independent policy for the same probe.

---

### LATCH-09 [P2] cupy-probe-latched-process-wide-with-a-test-only-reset
**File:** src/mlframe/feature_engineering/transformer/_utils.py:88
**Summary:** `is_gpu_available()` memoises `_GPU_AVAILABLE` and deliberately catches broadly (device alloc, D2H,
an NVRTC compile). The breadth is well-argued in the docstring, but the *latch* is not re-attempted: a single
transient failure pins CPU for the process. `reset_gpu_probe()` exists but its own docstring says "Production
callers should not need this", and there is no production call site.
**Failure scenario:** TRIGGER - the `cp.zeros(1).get()` or the `cp.asarray([1.0]).sum().item()` NVRTC compile
faulting on a momentarily contended or resetting device. BLAST RADIUS - process lifetime: every
transformer-package GPU dispatch that gates on this helper falls to CPU. Milder than LATCH-02/03/04 because the
outcome is logged at `INFO` with the exception type and message, so a run at least leaves a trace, and a reset
hook exists.
**Evidence:** _utils.py:88 (flag), :91-125 (probe + `except Exception` -> `logger.info(...)`,
`_GPU_AVAILABLE = False`), :127-133 (`reset_gpu_probe`, documented as tests-only); a repo-wide grep for
`reset_gpu_probe` returns only the definition.
**Suggested fix:** re-attempt - split the handler: `ImportError` latches, everything else warns and re-probes on
the next call under a bounded attempt counter (`_kernel_tuning`'s `_MAX_INIT_ATTEMPTS` pattern). Keep the broad
catch itself; only the latch needs narrowing.

---

### LATCH-10 [P3] kernel-tuning-init-attempts-never-decay
**File:** src/mlframe/feature_selection/filters/_kernel_tuning.py:32
**Summary:** `get_kernel_tuning_cache` correctly retries a failing `KernelTuningCache()` construction, but
`_INIT_ATTEMPTS` is a monotone process-lifetime counter reset only by `_reset_for_tests()`. Three *unrelated*
transient failures spread across a long-lived process exhaust the budget and latch `_CACHE_SINGLETON = False`
permanently.
**Failure scenario:** TRIGGER - three separate momentary faults (three concurrent-sweep rewrites of the 17 KB
JSON over a multi-hour service, say) rather than one persistent one. BLAST RADIUS - process lifetime: all 268
kernel-tuning dispatch sites fall back to hardcoded defaults. Mitigated by the `warning`-level log at exhaustion,
which the historical bug lacked - hence P3 rather than P1.
**Evidence:** _kernel_tuning.py:31-32 (`_MAX_INIT_ATTEMPTS = 3`, `_INIT_ATTEMPTS = 0`), :117-128 (increment and
the terminal latch), :136-142 (`_reset_for_tests` is the only reset). No decay, and no reset on an eventual
success (harmless there, since success returns the singleton).
**Suggested fix:** re-attempt with a time-decayed budget - reset `_INIT_ATTEMPTS` after a cooldown (or count
only *consecutive* failures) so three faults hours apart are not treated as one persistent breakage.

---

### LATCH-11 [P3] numba-global-rng-seeded-without-save-restore
**File:** src/mlframe/training/_iterative_stratification_njit.py:35
**Summary:** `_iterative_stratification_njit` calls `np.random.seed(seed)` in `@njit` scope, which sets **numba's
own** global RNG state for the calling thread. There is no save/restore, so the numba stream of any later njit
kernel on that thread is repositioned by whatever this function consumed.
**Failure scenario:** TRIGGER - any call to this splitter. BLAST RADIUS - the calling thread's numba RNG stream
for the rest of the process, until some other code re-seeds it. Marked P3 because the numba-RNG consumers found
in the package (`_permutation_null._build_shuffle_matrix`, `_mdlp_validated_split`) all re-seed per
row/permutation before drawing, so no current consumer is actually shifted - but the invariant is unguarded and
the next unseeded numba draw added anywhere silently inherits it.
**Evidence:** _iterative_stratification_njit.py:28-35. Contrast `feature_selection/filters/screen.py:50-75`,
which is the codebase's own worked example of this hazard: it snapshots `np.random.get_state()`, derives
entropy-based restoration seeds for numba and cupy precisely because "those exposed no portable get_state and
were not previously restored on exit, leaving the caller's downstream numba/cupy stream shifted", and restores in
a `finally`.
**Suggested fix:** clear on exit - wrap the call site (the Python-level caller, since the restore cannot live
inside `@njit`) in the same entropy-derived save/restore `screen.py` already implements, ideally by extracting
that block into a shared `numba_rng_scope()` context manager.

---

### LATCH-12 [P3] env-derived-cuda-verdict-frozen-in-a-device-availability-cache
**File:** src/mlframe/feature_selection/filters/_fe_gpu_strict.py:75
**Summary:** `_cuda_usable()` is documented as caching only the immutable device probe - "the ENV FLAG is read
LIVE on every call" - but the `CUDA_VISIBLE_DEVICES` / `MLFRAME_DISABLE_GPU` short-circuits at :182-184 write
their verdict into the same process-lifetime `_CUDA_USABLE_CACHE`, so those two env vars *are* frozen at first
call.
**Failure scenario:** TRIGGER - a process that sets `CUDA_VISIBLE_DEVICES=""` for one stage (a CPU-strict
determinism check, a test using the documented "Set CUDA_VISIBLE_DEVICES='' to force CPU" escape hatch) and then
unsets it. BLAST RADIUS - process lifetime: STRICT-resident FE stays off for every later fit, and the module has
no reset hook. The docstring at :186-190 acknowledges the trade ("those are start-of-process device gates, not
the runtime STRICT toggle"), so this is a documented design choice rather than an oversight - flagged because the
comment at :71-75 promising live env reads and the code at :182-184 disagree about which env vars are live.
**Evidence:** _fe_gpu_strict.py:71-75 (the "ENV FLAG is read LIVE" comment), :178-184 (env short-circuit writing
the cache), :186-190 (the acknowledgement). No `reset` function in the module.
**Suggested fix:** re-attempt - evaluate the two env short-circuits live on every call (they are a dict lookup)
and cache only the ~17us pyutilz/numba probe, which is what the module's own comment already claims happens.

---

### LATCH-13 [P3] triton-bootstrap-disabled-on-any-unexpected-error
**File:** src/mlframe/training/neural/_triton_bootstrap.py:103
**Summary:** `ensure_triton_loaded()` caches `_triton_loaded`; the outer handler catches bare `Exception`, sets
`_triton_loaded = False` and returns, so every later `is_triton_available()` reports unavailable without
re-probing.
**Failure scenario:** TRIGGER - `ctypes.WinDLL(pyd, winmode=0x8)` or the `site.getsitepackages()` walk raising
for a momentary reason (a file lock on `libtriton.pyd` from a concurrent install, a transient `OSError` on a
network-mounted site-packages). BLAST RADIUS - process lifetime: all Triton-dependent neural paths use eager
fallbacks. P3 because the common case ("Triton is not installed") is genuinely permanent and correctly latched,
the failure is logged at `warning`, and the eager fallback is a correctness-preserving path.
**Evidence:** _triton_bootstrap.py:94-108 (`except Exception as _bootstrap_err:` -> `_triton_loaded = False`,
`logger.warning`), :111-114 (`is_triton_available` just delegates, no re-probe).
**Suggested fix:** narrow - treat "no candidate `.pyd` found" as the permanent verdict (it already is, at :95)
and let an unexpected exception re-probe once on the next call rather than latching.

---

### LATCH-14 [P3] LEAD: log-suppression flag latches, behaviour does not
**File:** src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_subsetrank.py:50
**Summary:** `_fallback_logged` latches True on the first GPU-unavailable fallback so the warning is emitted
once per process.
**Failure scenario:** TRIGGER - the first GPU fallback. BLAST RADIUS - **no behavioural degradation**: the
dispatch itself is re-attempted on every call (`gpu_available()` is consulted each time; there is no cached
"failed" verdict), so only the operator-visible warning is suppressed. Marked LEAD per the brief: the trigger is
concrete but the blast radius is diagnostic-only. Worth recording because a run that loses the GPU *after* the
first fallback message emits nothing further, so the log understates how long the CPU kernel ran.
**Evidence:** _shap_proxy_subsetrank.py:50 (`_fallback_logged = False`), :217-222 (both `if not
_fallback_logged:` guards). No cached availability verdict anywhere in the function.
**Suggested fix:** none required for correctness; if the operator signal matters, rate-limit the warning by time
rather than latching it, or downgrade the repeats to `info`.

---

## Verified-clean (checked, no finding)

Recorded so the next audit does not re-walk them. All were candidates the scan surfaced under one of the six
shapes and were dismissed only after reading the guard.

| Site | Shape | Why clean |
|---|---|---|
| `_orth_mi_backends._select_mi_backend` (:250) | 1,2 | The documented historical bug. Now returns `"numba"` (optimistic) on any non-`ImportError`, with a `warning`. |
| `_kernel_tuning.get_kernel_tuning_cache` (:110) / `_register_default_tuning_cache` (:45) | 1,2 | The documented historical bug. `ImportError` latches; anything else retries (`_MAX_INIT_ATTEMPTS`) and warns. Residual nit -> LATCH-10. |
| `_fe_deadline` main-thread path | 4 | The documented historical bug. `_clear_fe_deadline()` now called in `_mrmr_class.py:4001`'s `finally`. Worker path -> LATCH-01. |
| `_fe_gpu_strict._cuda_usable` (:180) | 1,2 | `ImportError` latches; a device fault goes through `_retry_cuda_available` (bounded retry + `warning`). Env nit -> LATCH-12. |
| `_cb_pool._cb_gpu_usable` (:600) | 1,2 | Only the wheel-signature string `"Environment for task type [GPU] not found"` latches; transient faults retry `_CB_GPU_PROBE_ATTEMPTS` times with a `warning`. |
| `training/_gpu_probe` (:20-34) | 1,2 | `ImportError` -> False; anything else warns and stays **optimistic** (`CUDA_IS_AVAILABLE = True`), deferring to per-library binary probes. |
| `_cmi_cuda` / `permutation.mi_direct` / `_permutation_null_pair_resident` breakers | 1 | Re-armed per fit by `_mrmr_class_fit_helpers._rearm_gpu_circuit_breakers`. The two omitted siblings -> LATCH-05. |
| `_cmi_cuda._resident_upload` (:353) | 5 | `id(factors_data)`-keyed but folds a content hash `(shape, dtype, hash(tobytes()))` into the entry; a recycled id misses. |
| `_cmi_cuda._resident_factors_device` / `_cmi_forder_view` | 5 | weakref-identity guarded, LRU-bounded, with `clear_*` teardown hooks. |
| `batch_mi_noise_gate_gpu._resident_y_all_device(_cupy)` (:381, :421) | 5 | `(id, weakref)` co-validated - `ref() is classes_y` re-checked on every hit; a dead ref drops the entry. |
| `_mah._get_y_binning` (:96), `_mi_dispatch._get_unique_y` (:77) | 5 | weakref co-validation + dead-entry sweep + LRU. |
| `training/core/predict.py` df cache (:144), `_dummy_baseline_compute` (:107) | 5 | weakref / `(cols, shape)` revalidation on hit with an explicit recycled-id comment. |
| `xgb_shim._DMatrixReuseMixin`, `lgb_shim._DatasetReuseMixin` | 5 | `__getstate__` nulls both pointer and key attrs and stamps the library version; `__setstate__` re-inits. |
| `HonestLossCache` (`_shap_proxy_loss.py:168`) | 5 | `__getstate__` excludes `_lock`; `__setstate__` rebuilds it. |
| `MRMR.__getstate__` (:3236) | 5 | Strips `_fit_reentrancy_lock_`, stamps a schema version, warns on downgrade. Remaining `self._*_cache_` attrs are plain picklable data. |
| `_flat_torch_module` ContextVar (:25-41) | 4 | `copyreg` reducer reduces a ContextVar to a fresh same-named one, so no captured context is pickled. |
| `_process_flag_scope` + `reporting/colors` + `renderers/save` overrides | 4 | Full snapshot/restore chain, idempotent, reachable from both finalize and the suite `finally`. Ordering gap -> LATCH-06. |
| `screen.py` seeding scope (:50-78) | 6 | Saves `np.random.get_state()` and derives entropy-based restore seeds for numba **and** cupy; restores in `finally`. |
| `_permutation_null._build_shuffle_matrix`, `_mdlp_validated_split` | 6 | `np.random.seed` inside `prange` bodies, re-seeded deterministically per row - thread-count independent by construction. |
| `crash_reporting.enable_crash_reporting` (:34) | 1 | `_ENABLED = ok` is set **after** both steps, so a partial failure permits a later retry. |
| `core/stats.get_tukey_fences_multiplier_for_quantile`, `holiday_calendar_features._cached_holiday_dates`, `rfecv._fit_accepts_sample_weight` | 3 | Pure functions of their keys; no mutable external state read inside the cached body. |

## Summary

| ID | Sev | Site | What latches | Blast radius |
|---|---|---|---|---|
| LATCH-01 | P1 | polynom_pair_fe.py:388 | FE wall-clock deadline in a **loky worker** thread-local, never cleared | Worker process lifetime, across later fits |
| LATCH-02 | P1 | _gpu_metrics.py:72 | `_GPU_AVAILABLE = False` on any exception, `debug` only, no reset | Process lifetime; batch AUC/RMSE on CPU |
| LATCH-03 | P1 | _core_auc_brier.py:126 | `_GPU_ARGSORT_AVAILABLE = False` on any exception | Process lifetime; ~10% measured e2e loss at 200k |
| LATCH-04 | P1 | _shap_proxy_cluster_su.py:64 | `_GPU_AVAILABLE_CACHE = False` on an alloc fault | Process lifetime; cluster-SU pair loop on CPU |
| LATCH-05 | P1 | _mrmr_class_fit_helpers.py:91 | KSG + order-1 max-T breakers excluded from the fit-entry re-arm | Process lifetime instead of one fit |
| LATCH-06 | P2 | _phase_config_setup.py:208 | 4 suite overrides flipped ~280 lines before their restore snapshot exists | Thread lifetime on a setup raise |
| LATCH-07 | P2 | member_metrics.py:23 | `lru_cache` memoises an env+KTC-dependent verdict, incl. its failure path | Process lifetime per shape key |
| LATCH-08 | P2 | _gpu_metrics.py:129 | `_NUMBA_CUDA_AVAILABLE = False` on any exception | Process lifetime; numba-CUDA RMSE path off |
| LATCH-09 | P2 | _utils.py:88 | cupy probe latched; `reset_gpu_probe` has no production caller | Process lifetime (logged at INFO) |
| LATCH-10 | P3 | _kernel_tuning.py:32 | `_INIT_ATTEMPTS` never decays; 3 unrelated faults latch | Process lifetime; 268 dispatch sites on defaults |
| LATCH-11 | P3 | _iterative_stratification_njit.py:35 | numba global RNG seeded with no save/restore | Calling thread's numba stream |
| LATCH-12 | P3 | _fe_gpu_strict.py:75 | `CUDA_VISIBLE_DEVICES` verdict frozen in a "device-only" cache | Process lifetime; STRICT FE off |
| LATCH-13 | P3 | _triton_bootstrap.py:103 | `_triton_loaded = False` on any unexpected error | Process lifetime; eager fallbacks |
| LATCH-14 | P3 (LEAD) | _shap_proxy_subsetrank.py:50 | Warning suppression only; dispatch still re-attempted | Diagnostics only |

**Totals:** 5 x P1, 4 x P2, 5 x P3 (one of which is a LEAD). 21 candidate sites read and dismissed, with the
guard that clears each one recorded above.
