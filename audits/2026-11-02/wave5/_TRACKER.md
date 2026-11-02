# Wave 5 audit 2026-11-02 -- tracker

Five read-only reports on `src/mlframe` by Opus 5.5 low-effort agents, one per attack class: CUDA kernel integer overflow, public API validation, output paths and symlinks, global state and determinism, GPU context and device memory. The agents could not write files, so the reports are their returned text, condensed. `pip-audit` 2.10.1 on the local environment reported "No known vulnerabilities found". Dispositions: RESOLVED (safe, or fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | OVF5-1 | cuda_overflow | High | `transformer/_kernels_cupy.py` stage-4 kernel: int offsets now `long long` |
| **RESOLVED** | OVF5-2 | cuda_overflow | Low | same launcher refuses K above 1024 |
| **RESOLVED** | OVF5-3 | cuda_overflow | Low | `_batch_pair_mi_cuda_shared_fused.py:219` already validates `threads_per_block` |
| **DOC** | OVF5-4 | cuda_overflow | Info | fixed 256-entry shared arrays tied to the hard-coded block size |
| **REJECTED** | OVF5-5 | cuda_overflow | Info | int histogram counters, safe below 2^31 rows |
| **RESOLVED** | API5-1 | api_validation | Medium | model name in the `.dump` path (`safe_file_stem`) |
| **RESOLVED** | API5-2 | api_validation | Medium | pre-pipeline and model name in the cached model path |
| **RESOLVED** | API5-3 | api_validation | Low | calibration plot file name |
| **RESOLVED** | API5-4 | api_validation | Medium | OOF probabilities and target truncated to the shorter length |
| **RESOLVED** | FSW5-1 | output_paths | Medium | `embed_website_to_mlflow`: `..` refused, atomic write |
| **RESOLVED** | FSW5-2 | output_paths | Low | stale CatBoost snapshot deletion contained to `train_dir` |
| **RESOLVED** | FSW5-3 | output_paths | Low | CatBoost border files in the private per-user directory |
| **RESOLVED** | FSW5-4 | output_paths | Low | numba stamp renamed into place |
| **DOC** | FSW5-5 | output_paths | Info | `votenrank/stability_exp.py` `filename` is the caller's explicit path |
| **RESOLVED** | STATE5-1 | global_state | Medium | `TQDM_DISABLE` set only during `HybridSelector.fit` |
| **RESOLVED** | STATE5-2 | global_state | Medium | id-keyed caches weak, locked and cleared at the end of a fit |
| **RESOLVED** | STATE5-3 | global_state | Low | pinned-memory warning filter through `install_filter_once` |
| **DOC** | STATE5-4 | global_state | Low | `LOKY_MAX_CPU_COUNT` default set at import |
| **DOC** | STATE5-5 | global_state | Low | prewarm draws from numba's per-thread RNG stream |
| **REJECTED** | STATE5-6 | global_state | Info | vendored `infonet/infer.py` is not imported |
| **RESOLVED** | GPU5-1 | gpu_context | Low-Medium | RFF path releases the device pool on failure too |
| **RESOLVED** | GPU5-2 | gpu_context | Low | row-attention releases pool and pinned buffers in `finally` |
| **RESOLVED** | GPU5-3 | gpu_context | Low | RFF path clears its pinned host buffers |
| **REJECTED** | GPU5-4 | gpu_context | Info | idempotent kernel compile; importing cupy creates no context |

## Counts

24 findings: 1 High, 5 Medium, 1 Low-Medium, 8 Low, 9 Info. Dispositions: 17 RESOLVED, 4 DOC, 3 REJECTED, 0 FUTURE.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| FSW5-1 | Refuse a '..' component in embed_website_to_mlflow and write the file atomically | `tests/integrations/test_mlflow.py` |
| OVF5-1, OVF5-2 | Use 64-bit offsets in the row-attention stage-4 kernel and refuse K above the block limit | `tests/feature_engineering/transformer/test_kernels_cupy_softmax_temp_zero_guard.py` |
| GPU5-1 to GPU5-3 | Release GPU memory and pinned buffers when an RFF or row-attention GPU step fails | `tests/feature_engineering/transformer/test_gpu_memory_released_on_failure.py` |
| API5-1, API5-2 | Keep model and pipeline names from steering the saved file's path | `tests/utils/test_safe_filename.py` |
| API5-3, API5-4 | Skip the honest calibration block on a row-count mismatch and sanitise its plot name | `tests/training/test_honest_calibration_alignment.py` |
| STATE5-1 | Silence progress bars only during HybridSelector.fit, not for the whole process | `tests/feature_selection/test_hybrid_selector_leaves_tqdm_alone.py` |
| STATE5-2 | Hold the id-keyed fit caches' arrays weakly, evict under a lock and clear them when a fit ends | `tests/feature_selection/filters/test_id_keyed_caches_hold_no_arrays.py` |
| FSW5-2 | Delete the stale CatBoost snapshot only inside train_dir and never through a symlink | `tests/training/test_catboost_snapshot_discard_containment.py` |
| FSW5-3, FSW5-4, STATE5-3 | Rename the numba stamp into place, stage CatBoost border files privately, route the pinned-memory filter through install_filter_once | `tests/utils/test_stamp_filter_scratch_hygiene.py` |

## Not covered by this wave

- Host launchers of `_gpu_resident_extval*`, `_gpu_resident_*_ktc`, `_fe_batched_mi_guards`.
- API validation: strings iterated per character where a list is expected, zero or negative counts, `rfecv/_validate.py`, selector `fit` methods, `reporting/` outputs.
- f-string log calls with `{col}` and no `!r` (carried over from wave 4).
- The `_get_pinned_buffer` docstring ("grows but never shrinks") no longer matches the 32-entry LRU cap.
