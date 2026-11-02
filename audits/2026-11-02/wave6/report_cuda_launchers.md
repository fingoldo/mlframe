# Wave 6 audit 2026-11-02: host-side launch code for CUDA kernels

Scope: Python launchers of the cupy and numba.cuda kernels, read-only, Opus 5.5 low effort. Condensed from the agent's returned text. Four files were read in full (`_gpu_resident_extval_fused.py`, `_fe_batched_mi_mm_cm.py`, `_gpu_resident_select.py` lines 120-268, `_gpu_resident_extval.py` lines 55-114); the three `_ktc` files and `_fe_batched_mi_guards.py` were only grepped and are not verified.

| ID | Severity | Location | Disposition |
|---|---|---|---|
| LNC6-1 | Low | `_gpu_resident_select.py:164-173`: the v3 compaction scratch (`K x n/4` uint64) is allocated without a free-VRAM check; an out-of-memory error is the fallback signal, logged at debug | RESOLVED |
| LNC6-2 | Low | `_gpu_resident_select.py:178-180, 255-257`: two fused-kernel fallbacks swallow every exception at debug level | RESOLVED |

LNC6-1: the scratch is skipped, with an INFO line giving its size, n and K, when it would take more than half the free VRAM (`_scratch_fits`); the v2 kernel answers.

LNC6-2: each fallback logs a WARNING with the exception, n and K through `log_throttle`.

Tests: `tests/feature_selection/gpu/test_radix_select_fallback_visible.py`.

Checked and guarded: `fused_extval_candidates` (sizes passed as int64, 1-D grid, the 2^31 x-limit would need 5.5e11 cells), `_candidate_matrix` (column length checked), `binned_mm_mi_from_cm_gpu` (shared memory checked, codes range-checked), `_radix_select_interior_edges` and `_radix_quantiles` (R checked against the kernel cap, shared memory against the device limit). No grid uses y/z, no fallback leaves an output partly filled.
