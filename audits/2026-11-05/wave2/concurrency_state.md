# Wave 2: concurrency and shared state (2026-11-05)

Scope: `src/mlframe`, read-only. Method: grep for module-level cache dicts/OrderedDicts, `threading.Lock` coverage, and process-global `os.environ` writes; every reported site was read.

## Coverage note

The cache surface is already heavily hardened. Of the ~75 module-level caches found, every one I checked sits in a module that also creates a module-level `threading.Lock`/`RLock` (178 files carry one). The cases I read in full, `_mrmr_fingerprints._MRMR_LAST_X_HASH_CACHE`, `_orth_extra_basis_fe_helpers._ORDER_CACHE`, `training/utils._PD_VIEW_LAST_CACHE` and `_pipeline_cache._LAST_KEY_CACHE`, all do their compare and fetch under one lock. The id-keyed memos fold a content signature or a weakref into the key. `kernel_tuning_fit_policy` is depth-counted under a lock. No unlocked check-then-act on a shared cache was found. What is left is **process-global environment and monkeypatch state**, where a lock around a dict does not help.

## Findings

### CONC-W2-1 (Medium): a kernel-tuning sweep erases the user's forced device setting and flips the backend for concurrent fits
- Files: `src/mlframe/feature_selection/filters/_bootstrap_rank_mi_gpu.py:190-204` and `src/mlframe/feature_selection/filters/_bootstrap_rank_mi_tied_gpu.py:220-229`.
- Statement: the tuner variants set `MLFRAME_BOOTSTRAP_RANK_DEVICE` / `MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE` process-wide. Their `finally` then calls `os.environ.pop(...)` instead of restoring the previous value.
- Interleaving:
  1. The user exports `MLFRAME_BOOTSTRAP_RANK_DEVICE=0`, for example on a contended 4 GB card, and enables `MLFRAME_AUTOTUNE=on`. Under that mode `kernel_tuning_fit_policy` lets sweeps run in the background during fits.
  2. The sweep runs `host_variant`, which sets the variable and then pops it, so the user's `0` is gone for the rest of the process.
  3. Meanwhile, fit thread B calls `use_device_walk()` while the sweep's `device_variant` holds the variable at `"1"`. B takes the device walk regardless of its own setting and the tuner's choice.
  4. The results agree to within `equiv_rtol` (1e-9), so selection does not change. The harm is a silently lost user override and possible VRAM pressure or OOM fallbacks in B.
- Disposition: RESOLVED-needed. Save the prior value and restore it in `finally`, as `_fe_cmi_fused_backend.py:60-68` already does. Better still, pass a `force_device=` argument into `bootstrap_rank_mi` instead of going through the env, so nothing process-global changes. The tied-path module needs the same fix.

### CONC-W2-2 (Low): `_disable_cuda_globally` races on recording the original `torch.cuda.is_available`
- File: `src/mlframe/training/neural/base/_cuda_fallback.py:126-131`.
- Statement: the `if _CUDA_DISABLED_STATE is None` check, the snapshot and the monkeypatch `torch.cuda.is_available = lambda: False` run without a lock.
- Interleaving:
  1. Two estimators in joblib threads both hit the CPU-retry-also-failed path.
  2. T1 passes the None check and is preempted before it assigns the snapshot.
  3. T2 runs the whole function, snapshots the original and installs the lambda.
  4. T1 resumes and overwrites the snapshot, recording T2's lambda as the "original" `is_available`.
  5. `restore_cuda_visibility()` later "restores" the lambda, so the process stays CPU-only, contrary to that function's documented contract. The same applies to `env` if it was already `""`.
- Disposition: RESOLVED-needed. Add a module lock around both `_disable_cuda_globally` and `restore_cuda_visibility`. This is cheap, and the window is real under threaded multi-estimator fits.

### CONC-W2-3 (Low): `_tqdm_off_during_call` save/restore is not re-entrant across threads
- File: `src/mlframe/feature_selection/hybrid_selector.py:169-186`.
- Statement: two hybrid-selector fits running in parallel threads each save the environment variable and restore it on exit, and these saves and restores interleave.
- Interleaving:
  1. A saves `None` and sets the variable to `"1"`.
  2. B saves `"1"`.
  3. A finishes and pops the variable, so B's progress bars reappear (cosmetic).
  4. B finishes and restores `"1"`.
  5. `TQDM_DISABLE=1` now leaks for the rest of the process. This is the exact leak the wrapper's docstring says it fixed.
- Disposition: RESOLVED-needed (cheap). Use a depth counter under a lock, the same pattern as `kernel_tuning_fit_policy`. Alternatively pass `disable=True` to the members' tqdm calls instead of touching the environment.

### CONC-W2-4 (Info): `_fe_cmi_fused_backend` sweep forces `MLFRAME_FE_GPU_FUSE_CMI_ENTROPY=1` process-wide during the sweep
- File: `src/mlframe/feature_selection/filters/_fe_cmi_fused_backend.py:60-68`.
- Statement: the prior value is restored correctly. While a background sweep (`MLFRAME_AUTOTUNE=on`) is running, though, a concurrent fit thread also takes the fused path. It is numerically equivalent, so this is a transient performance effect only.
- Disposition: DOC, or fold into the CONC-W2-1 fix by passing the flag as an argument.

No High-severity finding. No unlocked identity-keyed cache, per-call lock, or check-then-act race on a module cache was found in the sites read.
