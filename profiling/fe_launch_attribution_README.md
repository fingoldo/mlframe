# GPU FE launch-per-function attribution (the reliable method)

cProfile call-counts do NOT map to cuLaunchKernel counts (cupy ufuncs launch in C). nsys has no
`--python-backtrace`; `cupy.cuda.nvtx` text ranges are NOT captured by nsys 2025.1.3 (only CUB's
internal NVTX records). The ONE reliable per-callsite GPU-launch attribution in this environment:

1. `pip install nvtx` (the package nsys uses for `--python-functions-trace`).
2. List the GPU entry functions in `fe_launch_attribution_annotations.json` (module + functions).
3. Profile: `nsys profile -t cuda,nvtx --python-functions-trace=<annotations.json> --capture-range=cudaProfilerApi python <harness>.py`
   (the harness wraps the profiled region in cp.cuda.profiler.start/stop, e.g. scratchpad f2_nsys_strict.py).
4. Attribute: for each cuLaunchKernel event (CUPTI_ACTIVITY_KIND_RUNTIME, name like 'cuLaunchKernel'),
   find the INNERMOST traced NVTX range (NVTX_EVENTS, textId; exclude 'cub::*') containing its start ->
   launches-per-function. (innermost = smallest range duration containing the launch start.)

F2 300k STRICT attribution (2026-06-25, after -55% grind): _radix_select_interior_edges 1418,
_gpu_evaluate_basis_matrix 975, _build_best_existing_op_candidates_gpu 770, _gpu_detect_heavy_tail_batched
672, batched_cmi_gpu 562, _gpu_batched_abs_corr 528, build_resident_operand_table 362, joint_counts_gpu 275,
hermite resident 264. Use this (not cProfile) to pick fusion targets.
