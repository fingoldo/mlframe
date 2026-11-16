# `_class_encoding` (class merge and 2-variable joint counts, numba forms)

Measured 2026-11-16, GTX 1050 Ti (sm_61, 112 GB/s). Host shape: 100000 rows x 6 int32 columns of 10 bins; device shape: 100 int64 candidate columns of 10 bins against one target and one conditioning column, 100000 rows.

## Host profile (cProfile, 50 calls of merge_vars with 3 variables)

0.037 s for 50 calls, 3951 function calls, 0.034 s in the njit `merge_vars` itself: 0.25 ms (1 variable), 0.46 ms (2), 0.66 ms (3) per call at 100000 rows; `joint_entropy_2var` 0.14 ms. Nothing else is visible: the Python around it is below the timer resolution.

## Device twin: launches per call and the GPU profile (nvprof, 11 calls of batched_cmi_gpu)

The device form of this counting is `_fe_batched_mi` (kernel-tuning entries and column-block streaming are its own, see its record). Launches per call: 1 `batched_joint_hist2_i32` plus 1 `batched_joint_hist2_i32_smem`, 3 cub reductions and 2 `joint_entropy1` launches, a handful of cupy elementwise copies (about 8 launches per call after the one-off upload). 7.4 ms per call for 100 columns, 74 microseconds per column, against 0.66 ms for one host 3-variable merge.

Metrics (nvprof) of the kernels around the counting: achieved occupancy 0.81 to 0.86 for the cupy elementwise kernels and 0.51 to 0.75 for the cub reduction; device-memory read throughput 34 to 60 GB/s (cupy multiply) and up to 95 GB/s (cub reduction) of 112 GB/s; multiprocessor activity 57 to 74 % average. `batched_joint_hist2_i32` 1.23 ms and its shared-memory twin 0.95 ms per call, 20 % and 16 % of the GPU time; the histogram kernels are atomics / latency bound (see `_fe_batched_mi.md`: 24 GB/s of 112 GB/s measured on the single-launch MI kernel).

## Choice and streaming

Backend choice: not made here. The callers that can use the device ask the `_fe_batched_mi` entries (`cmi_fused_entropy_backend`, the candidate-batch crossover `cmi_candidate_batch_backend`); the host form is used for one merge of a few variables, where a launch costs more than the 0.25 to 0.66 ms of the numba loop. Streaming: the device path takes candidate columns in column blocks sized by the free VRAM; this module builds one merged class array of n int32 at a time.
