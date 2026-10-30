# Profiling the GPU feature-engineering fit

Workflow used to take the F2 example from 15 s to 11 s at 1M rows (and to find that the fit is launch-bound, not compute-bound):

1. `python profile_fit_gpu.py 1000000` - cProfile of one cold-data fit; writes `gpu_fit.prof` to `MLFRAME_PROFILE_DIR` (default: the temp directory).
2. `python profile_bucket_time.py` - self-time by library. mlframe Python, cupy call overhead, numpy and pandas show up separately.
3. `python profile_cupy_callers.py` - which mlframe functions issue the most cupy calls.
4. `python nvprof_autotag.py`, then run the generated `nvprof_auto.py` under `nvprof --profile-from-start off --print-gpu-summary` (put CUPTI's `lib64` on `PATH`) and
   `python nvprof_range_summary.py <log>` - kernel launches and GPU ms per function. High launches with low GPU ms: fuse the function. Low launches with high GPU ms: tune the kernel.
5. `python cupy_call_overheads.py` - host cost of the individual cupy operations, to size what a launch-bound stage costs.
6. `python -m mlframe.feature_selection._benchmarks.profiling.duplicate_d2h [rows]` - the device-to-host copies of one warm fit that move the same content more than once (a dense and a narrowed copy of one column count as one), with the call sites of every copy. Total bytes alone do not show this; a second reader of an array that takes its own copy only makes the total larger. Fix the readers so they share one copy (`LazyHostCodes.host_as`), and keep the redundant bytes at the level `test_cmi_residency_traffic.py` asserts.
7. `python time_tuner.py <kernel> <seconds>` - how long one kernel-tuning sweep takes and where it is when it exceeds the limit.

Always profile on a quiet machine and compare paired runs: a concurrent GPU process moves every figure by more than most single optimisations gain.

## Per-family profiles of the MRMR fit

Every `fe_*_enable` family can be measured on its own against a BASE fit (all families off), at any size, on the CPU or the strict-resident GPU path:

- `python -m mlframe.feature_selection._benchmarks.profiling.family_profile <family> --n 100000 --p 100 --mode cpu|gpu --binning mdlp|mdlp_validated|optimal_joint|fd|sturges|none_quantile|none_uniform` - one cProfile of one fit (after a warm-up fit with the same flags), top functions by own and cumulative time, the family walls, and `prof_<family>_<mode>_<binning>.pstats` in the profile directory. `--dump-every 120` prints every thread's stack every 120 s, which shows where a fit that does not finish is spending its time (cProfile does not see the numba threads: a function that waits on them shows as a lock acquire).
- `python -m mlframe.feature_selection._benchmarks.profiling.family_scale run --n 1000000 --p 100 --mode gpu --family <stem>|BASE|DEFAULTS` - one fit in a child process with a time limit, a memory guard and a sample of the CPU used by other processes; the record is appended to a JSONL file and carries `contaminated` when other processes used more than a tenth of the machine (such a timing is not comparable).
- `python -m mlframe.feature_selection._benchmarks.profiling.family_batch <n> <p> <mode> [timeout] [results.jsonl]` - every family one after another, resuming after the records already in the file.

Run one profile at a time and nothing else heavy beside it (tests, a second fit): a concurrent job made a GPU batch 3.4x slower.
