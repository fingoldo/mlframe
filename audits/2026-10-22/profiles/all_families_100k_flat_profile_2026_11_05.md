# All-families 100k x 100 GPU fit: stage sampler, 2026-11-05

`ss_all.py 100000 100 gpu` (0.2 s main-thread stack sampler, no cProfile), one warm process: wall 78.2 s.

- 35.6 s of the main thread is `run_stages_concurrently` waiting on the stage workers, so the stage list below is the workers' critical path, not extra time.
- No stage above 2.4 s besides `_fe_stage_cascade_early_a_part1` (12.0 s) and `_stage_binned_numeric_agg` (6.2 s); every worker chain is at most 2.4 s.
- Worker chains at the top: `fit_logistic_ridge_newton` (2.4), `mdlp_bin_edges` (2.2), `_vander4_gpu` in Fourier detection (2.0), `_one_num_col` (2.0).
- Default-config warm fits: 78-91 s (target ~100 s met). Remaining candidates are each under ~1 s of wall; none worth a kernel without a fresh A/B.
- A `cudaErrorIllegalAddress` seen once under `CUDA_LAUNCH_BLOCKING=1` did not reproduce in three clean single-fit reruns (nvlddmkm Event 153 at every earlier crash time).
