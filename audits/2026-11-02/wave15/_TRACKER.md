# Wave 15 audit 2026-11-02 -- tracker

Three read-only reports by Opus 5.5 low-effort agents: train/val/test leakage in the training suite, precision and dtype drift, and the scripts and entry points. The agents could not write files, so the reports are their returned text, condensed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | LEAK15-1 | leakage | High | `LeakageSafeEncoder(time_aware=True)` used unshuffled `KFold`, encoding every fold but the last from later rows; rows a forward-chaining splitter never holds out were left uninitialised (`feature_handling/target_encoders.py`) |
| **RESOLVED** | LEAK15-2 | leakage | High | `target_loo` train encoding is a decreasing function of the row's own label; it now logs a warning and its docstring says it is not leakage-safe for tree models (the method itself is unchanged) |
| **FUTURE** | LEAK15-3 | leakage | Medium | suite OOF predictions cross-validate only the final estimator on columns already selected and encoded with all train labels (`_trainer_train_and_evaluate.py`, `trainer.py`); the fix is to run OOF over `clone(Pipeline([pre_pipeline, model]))` on the untransformed frame, or to flag the OOF as not honest when a supervised step ran |
| **FUTURE** | LEAK15-4 | leakage | Medium | pool-ensemble "held-out" score is the in-sample score of the stepwise selection (`composite/hpo_ensembling.py`); needs nested K-fold selection or a rename to `selection_score` |
| **RESOLVED** | LEAK15-5 | leakage | Medium | `DirectMultiHorizonEnsemble` block-size search used shuffled K-fold over consecutive origins with overlapping targets; `time_ordered=True` now gives contiguous folds with an `n_horizons - 1` purge (default stays shuffled because row order is unknown) |
| **RESOLVED** | LEAK15-6 | leakage | Low | stacked discovery built its `_oof_` columns without the group ids pass 1 honours (`composite/discovery/_stacked.py`) |
| **RESOLVED** | DTYPE15-1 | precision | Medium | `per_group_rolling_reduce` / `per_group_cumulative` accumulated in the float32 output dtype (`feature_engineering/grouped.py`) |
| **FUTURE** | DTYPE15-2 | precision | Medium | GPU DTW keeps series and cost matrix in float32 while the CPU path is float64, so results depend on the backend the size picks (`signal/dtw.py`); needs offset-centred inputs or a float64 cost matrix and a measurement of the GPU cost |
| **RESOLVED** | DTYPE15-3 | precision | Low | EWMA block rounded its input to float32 (`feature_engineering/_timeseries_emit.py`) |
| **RESOLVED** | DTYPE15-4 | precision | Low | crossing count multiplied float32 differences, which underflowed for tiny ones (`feature_engineering/_numerical_counts.py`) |
| **RESOLVED** | CLI15-1 | scripts | High | `sync_and_push.sh` retry staged every modified tracked file with `git add -u`, pushing another session's edits inside the merge commit |
| **RESOLVED** | CLI15-2 | scripts | High | merge failures without a conflict and uncommittable merges were ignored and pushed over; now exit 5 with a message |
| **RESOLVED** | CLI15-3 | scripts | Medium | retry decisions matched localized git text; git now runs with `LC_ALL=C`. The match on the word "rejected" anywhere in hook output is unchanged |
| **RESOLVED** | CLI15-4 | scripts | Medium | a missing `origin/<branch>` was merged and pushed anyway; now exit 2 |
| **RESOLVED** | CLI15-5 | scripts | Medium | coverage scripts used repo-root-relative paths from any directory and still exited 0 |
| **RESOLVED** | CLI15-6 | scripts | Low | PowerShell coverage script split a quoted `-TestSelector` phrase on spaces |
| **RESOLVED** | CLI15-7 | scripts | Low | `prepush_kernel_tuning.py` ignored its argv and read `MLFRAME_SKIP_KERNEL_TUNING_HOOK=false` as skip |

## Counts

17 rows. Dispositions: 14 RESOLVED, 3 FUTURE.

## Implemented

| ID | Commit subject | Regression test |
|---|---|---|
| LEAK15-1 | Encode time-aware target-encoder folds from earlier rows only | `tests/training/feature_handling/test_target_encoder_time_aware_forward_chaining.py` |
| LEAK15-2 | Warn that the leave-one-out target encoding depends on the row's own label | `tests/training/feature_handling/test_target_loo_warns_about_own_label_leak.py` |
| LEAK15-5 | Add purged contiguous folds to the multi-horizon block-size search | `tests/training/composite/test_direct_multi_horizon_time_ordered_folds.py` |
| LEAK15-6 | Pass the group ids to the stacked discovery's OOF predictions | `tests/training/composite/test_stacked_oof_passes_groups.py` |
| DTYPE15-1, 3, 4 | Accumulate grouped sums in float64 and count tiny sign changes | `tests/feature_engineering/test_float32_accumulation_fixes.py` |
| CLI15-1..4 | Make sync_and_push.sh stop on a merge it cannot finish and stage only the merge's own paths | `tests/scripts/test_sync_and_push_script.py` (two new tests) |
| CLI15-5..7 | Run the coverage scripts from the repo root and make the kernel-tuning hook parse its arguments | `tests/scripts/test_prepush_kernel_tuning_hook_arguments.py` (the shell and PowerShell changes have no test) |

## Not covered by this wave

- Leakage: only the training package was traced; discovery sub-steps other than the stacked pass were not.
- Precision: about 1,000 float32 casts in `feature_engineering/transformer/*`, the pair materialisers and `training/neural/*` were not reviewed.
- Scripts: the inline `run:` steps of the 22 workflow files, `__main__` blocks outside the benchmark folders and the `mlframe-tune-kernels` console script were not read.
- Unrelated to this wave but seen in the same runs: `tests/test_meta/test_shared_uncalled_functions.py` fails on functions from other sessions (`fit_logistic_ridge_newton_numpy`, `fits_shared`, `launch_k3`, `get_kernel_k4f`, `masked_abs_spearman_many`, `_bspline_basis_values_numpy`) and on `uninstall_log_sanitizer`, a test-only helper of the log sanitizer from wave 6 that has no production caller.
