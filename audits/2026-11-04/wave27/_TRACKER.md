# Wave 27 audit 2026-11-04 -- tracker

Three read-only reports by Opus 5.5 low-effort agents on code no earlier wave read: the shared helpers and out-of-fold modules of `feature_engineering/transformer`, the transformers from `active_virtual` to `ks_shift`, and the pair-scoring pipeline of the MRMR filters. The agents could not write files, so the reports are their returned text, condensed; each finding was re-checked on HEAD before it was fixed. Dispositions: RESOLVED (fixed with a test), FUTURE, DOC, REJECTED.

| Disposition | ID | Class | Severity | Location |
|---|---|---|---|---|
| **RESOLVED** | W27KB-1 | cache | High | the key-bank cache fingerprint left out `y_train`, so the same X with another target loaded a bank holding the first target y and supervised projections; `_key_bank_fingerprint` now hashes `y_train` (optional argument, the legacy key is unchanged without it) and `build_key_bank` passes it |
| **FUTURE** | W27KB-2 | backend | Medium | `knn_search` honours `metric` on the hnswlib path only; the sklearn fallback is always Euclidean, and the hnswlib branch takes a sqrt of non-squared distances for the non-`l2` spaces |
| **FUTURE** | W27KB-3 | nan | Medium | the numba kernels compile with `fastmath=True`, which may fold the `isfinite` guards the CUDA kernels evaluate; `kfold_attention_loop` does not validate its input |
| **FUTURE** | W27KB-4 | leakage | Medium | `_oof.kfold_attention_loop` splits with `splitter.split(X_train)` and never passes groups or labels, so a group-aware splitter raises and the default shuffled KFold leaks within a group |
| **FUTURE** | W27KB-5 | leakage | Medium | the inner baseline folds of `_baseline_oof` / `_residual_oof` are shuffled and group-blind, so duplicates of a row sit in the training folds of its own baseline |
| **FUTURE** | W27KB-6 | stats | Low | `class_or_quantile_slice` overlaps the positive and negative regression clouds when the two quantiles coincide on tied targets |
| **FUTURE** | W27KB-7 | smote | Low | the SMOTE neighbour kernels drop the first neighbour as "self", which is wrong with exact duplicate rows |
| **FUTURE** | W27KB-8 | smote | Low | empty-minority SMOTE returns a 1-column array and an `n_min == 1` minority returns one row instead of `n_synthetic` |
| **FUTURE** | W27KB-9 | smote | Low | `_bgmm_sample` raises on an empty minority |
| **RESOLVED** | W27KS-1 | stats | High | `ks_shift` Wasserstein-1 used the right-endpoint CDF on each interval and ignored all mass outside the neighbours range (local [0,1] vs global [0.5] gave 0, true 0.5); the kernel now integrates the piecewise-constant CDFs exactly (prefix sums of the centred global sample, split where F_g crosses the local level, tails outside the global range included) and matches `scipy.stats.wasserstein_distance` |
| **RESOLVED** | W27KS-2 | stats | Medium | `ks_shift` KS distance only checked the right limit of each local sample, underestimating the sup by up to 1/k; both limits are checked and the value matches `scipy.stats.ks_2samp` |
| **RESOLVED** | W27IB-1 | leakage | High | `ib_baseline_codes` fits each baseline on the train rows and predicts the same rows, so its median cut-points and per-code target statistics come from overfit predictions while queries are coded from out-of-sample ones |; the train-side predictions are now inner-KFold(3) out-of-fold (query predictions still come from the full-train fit), a single-class inner fold falls back to the prior |
| **FUTURE** | W27FW-1 | crash | Medium | `fisher_weighted_residual` with a one-row query: `broadcast_to(...).reshape` returns a read-only view and the following `+=` raises |
| **FUTURE** | W27FW-2 | scale | Medium | `fisher_weighted_residual` bands the query rows on a different quantity (prediction entropy, abs(p - median y)) than the train thresholds were computed from (cross-entropy, abs(y - p)) |
| **FUTURE** | W27AF-1 | leakage | Medium | `adversarial_flip` takes the regression flip threshold from the in-sample train residual decile, which is biased low |
| **FUTURE** | W27AF-2 | convention | Low | class_distance / active_virtual / adasyn return zero distances for a class with fewer than 2 rows, the inverse of the 1e6 far sentinel used elsewhere |
| **FUTURE** | W27PR-1 | precision | High | the pair operand table is float32, so pairs are scored on float32 values while the emitted column is rebuilt in the original dtype (epoch-second operands lose the difference) |
| **FUTURE** | W27PR-2 | gate | Medium | the family-wise gate is skipped on the per-pair batched path because `var_pairs_codes` is only filled on the chunk path |
| **FUTURE** | W27PR-3 | gate | Medium | the Bonferroni divisor is `0.001 / len(prospective_pairs)` per call, so the threshold depends on how pairs were split across calls |
| **FUTURE** | W27PR-4 | crash | Medium | verbose mode formats a `None` correlation after a `_config_corr` exception and aborts the pair search |
| **FUTURE** | W27PR-5 | gate | Medium | a demoted prewarp winner is promoted again at its uncapped MI because `best_prewarp_mi` is not capped with `var_pairs_perf` |
| **FUTURE** | W27PR-6 | ledger | Low | joint-gate rejections are all labelled `operand_uplift_floor` because `ratio_ok` is never passed |
| **FUTURE** | W27PR-7 | ledger | Low | pairs rejected by the family gate leave no rejection record |
| **FUTURE** | W27PR-8 | debias | Low | on the RAM-fallback recompute path the Miller-Madow debias falls back to the nominal bin count |
| **FUTURE** | W27PR-9 | log | Low | the progress bar best field shows the pair own best, not the running maximum |
| **FUTURE** | W27PR-10 | dead | Low | dead NaN write on the per-pair materialise failure path |

## Not covered by this wave

- transformer package: the modules from `l` to `z` (`local_*`, `multi_*`, `neighbor_aggregate_features`, `per_column_rff`, `random_features`, `residual_*`, `row_attention`, `smote_*`, `sign_residual_baseline`, ...), `autoencoder.py`, `aux_mlp.py`, `boosted_attention.py`, `focal_lgb.py`, `lda_projection.py`.
- pair scoring: `_pairs_score_step4.py` (lines 20-289 only), `_pairs_dispatch.py`, `_pairs_chunks.py`, `_pairs_emit_extval.py`, `_pairs_gpu_gates.py`, `_pairs_abs_corr_gpu.py`, `_pairs_analytic_gpu.py`, `_pairs_chunk_codes.py`.

## Implemented

| ID | Regression test |
|---|---|
| W27KB-1 | `tests/feature_engineering/transformer/test_key_bank_cache_key_includes_the_target.py` |
| W27KS-1, 2 | `tests/feature_engineering/transformer/test_ks_shift_matches_scipy_exact_distances.py` |
| W27IB-1 | `tests/feature_engineering/transformer/test_ib_baseline_codes_train_predictions_are_out_of_fold.py` |
