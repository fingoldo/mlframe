# Wave 1 (2026-11-05), attack class A8: concurrency, global state, caches and pickling

Read-only audit by one Opus 5.5 low-effort agent. Eight defects verified on the current tree: one High, two Medium, five Low. Nothing was edited or run, and not every concurrency path was traced end to end, so the list is not exhaustive. The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| C8-01 | High | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_finalise.py:659` | The process-wide `MRMR._FIT_CACHE` stores the live fitted instance (`setdefault(_cache_key, self)`), so refitting that instance on other data silently changes what the old key replays. |
| C8-02 | Medium | `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_revalidate/_shap_proxy_loss.py:80-83, 122-123`; `src/mlframe/feature_selection/shap_proxied_fs/_shap_proxy_explain.py:83-84` | Disk-cache keys for cached losses and fitted boosters use `hash_array_summary` (head/tail rows plus column sum/min/max), not the full content. |
| C8-03 | Medium | `src/mlframe/_numba_parallel_guard.py:42-45, 63-81` | The process-wide guard lock is created through `lru_cache`, which does not serialise misses, and the threading-layer probe runs a `parallel=True` kernel with no guard. |
| C8-04 | Low | `src/mlframe/feature_selection/filters/_mrmr_fit_impl/_fit_impl_core.py:35` vs `_fit_impl_core_parts.py:26, 232-233` | Two separate `RLock`s are named `_MRMR_FIT_CACHE_LOCK`, and the one published on the class (`MRMR._FIT_CACHE_LOCK`) is not the one that guards the cache. |
| C8-05 | Low | `src/mlframe/feature_selection/filters/_fe_resident_operands.py:132-149` (used at `:220`, `:342`, `_qbin_block_gpu.py:144`, `_mi_greedy_cmi_fe_gpu.py:314`) | The content-hash memo is keyed on object identity, so it misses in-place mutation and the GPU operand cache can serve stale device data. |
| C8-06 | Low | `src/mlframe/feature_selection/filters/info_theory/_cmi_cuda.py:95-121` | The F-order copy cache is keyed on identity plus a weakref; it misses in-place mutation and holds up to 4 GB per live input with no count limit. |
| C8-07 | Low | `src/mlframe/feature_selection/filters/_gpu_resident_rank_bin.py:80-87` | A device-array cache is keyed on `(n, n_bins)` with no CUDA device id. |
| C8-08 | Low | `src/mlframe/feature_selection/filters/_neural_mi.py:94`; `_confirm_predictor.py:63`; `_orthogonal_univariate_fe/_orth_extra_basis_fe_helpers.py:389` | Environment variables are read once at import: `MLFRAME_NEURAL_MI_DEVICE` is never used at all, and two other switches are frozen at import. |

### C8-01 (High)

A cache hit copies (replays) the cached object's fitted attributes (`_fit_cache.py:155-160`), and the cache keeps a strong reference to the object itself. Nothing removes `self`'s old entry when it starts a new fit.

Scenario: `m = MRMR(); m.fit(X1, y1)` stores key K1 pointing to `m`. Then `m.fit(X2, y2)` overwrites `m`'s fitted state, and K1 still points to `m`. Later `clone(m).fit(X1, y1)`, for example inside `GridSearchCV` or a later suite tier, hits K1 and gets X2's `support_` and engineered recipes: a wrong feature selection with no error. The "locked replay" comment does not help, because the refit writes attributes without holding the lock, so a concurrent replay can also read a half-updated instance.

### C8-02 (Medium)

`disk_cache.py`'s own docstring says this summary lets arrays "that differ only in the middle rows share a key and replay each other's result". `_shap_proxy_explain_cache.py:90` already appends `content_digest`, but these three key builders do not.

Scenario: two `y_tr` vectors that differ only by swapping two labels in middle rows, as a resample or a partial permutation would produce. The count, sum, min, max and head/tail are identical, so the keys match, and the disk cache returns the first run's cached loss or pickled booster, across processes as well as within one.

### C8-03 (Medium)

Under the workqueue threading layer the lock exists precisely to stop concurrent entry.

Scenario: two joblib threads call `parallel_kernel_entry()` at the same time in a cold process. Both miss `_entry_lock`'s cache, each builds its own `RLock`, and both enter their prange kernels at once, which is the workqueue crash the guard was written to prevent. Separately, both threads can miss `_concurrent_entry_ok` and run `_probe` in parallel with no lock. A module-level `RLock()` would fix it.

### C8-04 (Low)

`_finalise` and `_fit_cache` use the `_fit_impl_core` lock, while `_fit_impl_core_parts` publishes its own lock on `MRMR._FIT_CACHE_LOCK`. Any caller that takes `MRMR._FIT_CACHE_LOCK` (the documented "canonical" lock) gets no mutual exclusion with the real cache writes. Nothing reads it today, so this is latent.

### C8-05 (Low)

The memo only checks that `ref() is host` and that shape and dtype match.

Scenario: a caller passes a fit-constant array to `resident_operand`, then writes into it in place (as the DCD swap null does to `factors_data` at `_dcd_swap_null.py:135`) and passes it again. The same hash comes back and the stale device buffer is returned. No current caller was confirmed to mutate an array before passing it again; today it is an unenforced "read-only operands" contract, and nothing makes the arrays read-only.

### C8-06 (Low)

The same mutate-and-restore pattern breaks this cache if `_cpu_cmi_loop` is ever called on `state.factors_data` during the null; today the DCD path calls `conditional_mi` directly. Every live 2-D C-order matrix keeps its own F-order copy, and the only cleanup is a sweep of dead entries on the next miss.

### C8-07 (Low)

`resident_operand` puts `device_id` in its key (`:219-220`), but this cache does not. In a multi-GPU process the first device's boundary array is handed back to code running on another device, which leads to a cross-device access error or an implicit peer copy.

### C8-08 (Low)

- `_NEURAL_MI_DEVICE` is defined and never referenced, so `MLFRAME_NEURAL_MI_DEVICE=cpu` has no effect and neural MI still picks CUDA when it is available.
- `MLFRAME_MRMR_EVAL_POOL` and `MLFRAME_FOURIER_COARSE_GEMV` are read at import. Setting them after `import mlframe` (in a test, a notebook, or `os.environ` inside a script) does nothing, unlike the many per-call env reads elsewhere in the code.

## Checked and found sound (not reported)

The ensembling gate cache (`models/ensembling/predict.py`, full-buffer blake2b), the `_trend` fit cache, `_HEAVY_TAIL_MEMO`, `MatrixKeyedCache`, `ContentMemo`, `_Y_DENSE_MEMO`, the InfoNet model/y-prep caches, both lazy `__getattr__` caches, `_ATTACHED` shared memory, `OptimalJointPlanCache`'s thread-local, and `MRMR.__getstate__` dropping the null-codes closure.

## Not covered

- Fork vs spawn and CUDA-before-fork in the joblib/loky fan-outs were not traced.
- A systematic sweep for estimators that mutate state in `predict`/`transform`; only spot-checked.
- The `__getstate__` contents of the other 47 classes that define one, and pickle round-trips of the neural wrappers.
- The roughly 40 GPU kernel caches (`_*_KERNELS` dicts), checked only for locking patterns and not for per-device keying.
- Thread-local state in `info_theory/_state_and_dispatch.py` crossing joblib thread workers.
- The `pyutilz` side, including `kernel_tuning_cache`.
