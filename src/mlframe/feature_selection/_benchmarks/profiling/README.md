# Profiling the GPU feature-engineering fit

Workflow used to take the F2 example from 15 s to 11 s at 1M rows (and to find that the fit is launch-bound, not compute-bound):

1. `python profile_fit_gpu.py 1000000` - cProfile of one cold-data fit; writes `gpu_fit.prof` to `MLFRAME_PROFILE_DIR` (default: the temp directory).
2. `python profile_bucket_time.py` - self-time by library. mlframe Python, cupy call overhead, numpy and pandas show up separately.
3. `python profile_cupy_callers.py` - which mlframe functions issue the most cupy calls.
4. `python nvprof_autotag.py`, then run the generated `nvprof_auto.py` under `nvprof --profile-from-start off --print-gpu-summary` (put CUPTI's `lib64` on `PATH`) and
   `python nvprof_range_summary.py <log>` - kernel launches and GPU ms per function. High launches with low GPU ms: fuse the function. Low launches with high GPU ms: tune the kernel.
5. `python cupy_call_overheads.py` - host cost of the individual cupy operations, to size what a launch-bound stage costs.
6. `python time_tuner.py <kernel> <seconds>` - how long one kernel-tuning sweep takes and where it is when it exceeds the limit.

Always profile on a quiet machine and compare paired runs: a concurrent GPU process moves every figure by more than most single optimisations gain.
