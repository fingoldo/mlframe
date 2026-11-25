# Meta-test durations, 2026-11-24

Full `tests/test_meta` run with `--durations=40` (2295 tests): 23 failed, 2265 passed, 42 skipped, 2648 s total.
The run started on a cold numba cache, so per-test numbers are inflated by compilation. Warm re-runs are the comparable figure.

| Test | Before (full run, cold cache) | After / warm re-run |
|---|---|---|
| `test_x_test_suite_architecture_fixes::test_f8_run_heavy_automl_flag_registered` | 224.3 s | not measured (file is another session's WIP, not touched) |
| `test_gpu_optout_makes_no_device_allocations::test_gpu_optout_allocates_nothing_on_device` | 207.3 s | 43.7 s (warm) |
| `test_code_audit_baseline::test_no_new_code_audit_findings` | 149.0 s | not measured yet |
| `test_code_audit_tests_baseline::test_no_new_test_quality_findings_in_the_test_suite` | 114.3 s | not measured yet |

Profile of the opt-out test's fit (standalone, cProfile, 4000 x 12, `fe_max_steps=2`): 56 s wall with the profiler on.
Most cumulative time is spent waiting on the FE thread pool (`concurrent.futures`, `threading.wait`); no single
hot function dominates the main thread. A speed-up needs the worker-thread profile, not this one.
