# `_fe_pure_form_retention` family (`_fe_pure_form_retention`, `_fe_pure_form_sides`): usability-aware recovery of pure single-pair engineered forms

Measured 2026-11-16, GTX 1050 Ti host (sm_61, 112 GB/s), 8 logical cores. One `retain_usable_pure_forms` call (stub MRMR, the fixture of `test_biz_val_filters_fe_pure_form_retention`) on 100000 rows: pair interaction `y = 0.2 a^2 / b + noise` with 12 noise columns, `min_resid_corr=0`; the recovered form is `div(identity(a),sqrt(b))` on every path.

## Whole pass, by path

| path | wall per pass |
|---|---|
| host (`MLFRAME_FE_GPU_STRICT=0`) | 1.75 s |
| AUTO with the fit state of a 100000 x 100 fit | 1.21 s |
| STRICT-GPU | 1.04 s |

## Host profile (cProfile, host path, 1174644 function calls, 2.35 s)

`usability_greedy` 1.27 s cumulative (of which `build_usability_candidate_pool` 1.14 s: the pair pool and its replay check), `_adds_nonlinear_value` 0.25 s over 30 calls (the per-candidate residual fit), `drop_forms_not_beating_their_sides` 0.04 s. Self time: a cupy `sort` 0.49 s over 10 calls (the pool's rank binning on the device even on the host path: the device wait surfacing), numpy reductions 0.12 s, `cov` 0.08 s, scipy `lstsq` 0.05 s. No Python loop dominates (the same finding as the 2026-06-22 note in the function's docstring, which put the pool at 4.56 s of 4.72 s on the F2 100k fit before the njit pair kernels).

## Launches per call and the GPU profile (nvprof, one STRICT pass)

Kernel share of the GPU time: `radix_select_f32_v5` 17.0 % (10 launches of 2.6 ms: the edge selection of the rank binner), `cupy_var_core_float32` 7.1 %, `fused_gen_cm_f32` 5.0 % (10 launches, 0.78 ms: the candidate generation), the cub segmented sort 4.9 % (20 launches), `cupy_mean` 3.1 % (352 tiny launches), `cupy_absolute` 2.8 %. Host side of the STRICT pass: `_cv_candidates` 0.44 s cumulative over 6 calls, `_abscorr_batch_resident` 0.20 s over 87 calls, the blocking `.get()` 0.11 s over 102. Occupancy and bandwidth of `radix_select_f32_v5`, `fused_gen_cm_f32` and the pure-form gate kernels are in `_hermite_fe_mi.md`, `_basis_fused_gpu.md` and `_fe_pure_form_retention_gpu_resident.md`.

## Backend choice and memory

Choice: STRICT / AUTO by fit size for the resident usability scoring, the tuned host/device choice of `_fe_pure_form_retention_gpu_resident` for the relevance gate. Memory: the CV scoring runs on a row block of at most `max_rows` (3000) rows over at most `max_base_features` (14) base columns, so the working set is bounded by those caps, not by the frame; the pool build reads full-length base columns once.
