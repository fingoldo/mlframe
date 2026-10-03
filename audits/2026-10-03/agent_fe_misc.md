# agent_fe_misc: FE and misc failing tests (2026-10-03)

Runs: -n 1, OMP/NUMBA threads 2, --no-cov, `-p no:randomly` (fixed order). The machine was shared and memory/VRAM saturated (GPU free VRAM 0.00-0.15GB in logs), so runs were 10-20x slower than normal.
A first log (`%TEMP%/r2.log`) was clobbered by another agent writing the same filename; those results were discarded and rerun with session-scoped log names.

| # | Test | Reproduced | Status |
|---|------|-----------|--------|
| 1 | test_fix5_int64_downcast_logged_under_verbose1 | yes | FIXED |
| 2 | test_biz_multimode_beats_single_mode_on_multimode_target[default] | no (passes, in fixed-order run) | not reproduced, bound untouched, no multi-seed measurement done |
| 3 | test_all_modern_fe_mechanisms_f32_parity, test_f32_nameset_matches_f64[orth_univariate] | no (both pass) | not reproduced; [rankgauss] hit a MemoryError (machine memory pressure, not a defect) |
| 4 | TestRebuildFullSurvivorColErrstate::test_overflow_binary... | no (passes in isolation) | not reproduced; order/filter-state leak unverified (needs the full CI order) |
| 5 | test_polyeval_sweep_disables_cuda_when_it_never_wins | no (passes) | not reproduced |
| 6a | test_outlier_detection_improves_regression_rmse (tests/training/test_bizvalue_outliers_earlystop.py) | no (passes; floor in the file is already 6.5%, not 8%) | not reproduced |
| 6b | GPU/CPU selection: test_mrmr_gpu_cpu_selection_identical[clf_binary] | yes | OPEN, root cause not established |
| 6c | cache-hit timing: tests/feature_selection/contracts/test_integration_contract.py (2x speedup on fit cache hit) | no (10 passed in that run incl. it) | not reproduced |

## 1. int64 downcast not logged at INFO
Root cause: `MRMR._coerce_target_dtype` logged the successful int64->int16 downcast with `logger.debug`, so a caplog at INFO never saw it; the test and its docstring (verbose>=1 routes through logger.info) are the contract.
Fix: `logger.info(...)` (still gated on `self.verbose`). File: `src/mlframe/feature_selection/filters/mrmr/_mrmr_class_config.py` (CRLF preserved; single-line sed).
Result: after the fix the test was not re-run on its own (the follow-up runs were targeted at other tests); the pre-fix failure signature matched exactly (only min_relevance_gain message at INFO). The existing test is the regression test.

## 6b. clf_binary GPU vs CPU
Observed: CPU selects ['a', 'div(sin(a),exp(b))']; GPU additionally admits 'add(qubed(c),rint(e))'. 17 of 18 tests in the file pass.
Logs show `batch_pair_mi` / `batch_pair_usability_corr` GPU uploads rejected for the VRAM cushion floor and falling back to CPU while other agents held the GPU, so the GPU path taken during the test depends on free VRAM at that moment. Not proven as the cause; no production change made.
Next step (not done): rerun on a quiet GPU, and if it persists compare the GPU vs CPU MI of the extra candidate against its admission threshold.

## Not done
No new production changes beyond #1. No multi-seed measurement for #2 since it passed. Item 3's rankgauss cell and anything needing the full CI ordering were not exercised.
