# Wave 2: silent-failure audit (mlframe, src/mlframe)

Scope: broad `except` clauses, debug-only logging on paths that change a result, ambiguous `None`/default returns, silent truncation, and fallbacks nobody is told about.
Method: grep for `except Exception` under `src/mlframe` (benchmarks excluded), then read the code path behind each candidate in training, inference, ensembling, calibration, evaluation and feature_selection.
Excluded as already handled: `_predict_guards` NaN probes (warn and let the real error surface), the `HybridSelector.transform` replay failure (raises `ValueError` on missing columns), `bootstrap.py` resample failures (warns above 25%), and the items in CLAUDE.md / CHANGELOG "Fixed".

| id | sev | site | disposition |
|---|---|---|---|
| SIL-W2-1 | Medium | reporting/_diagnostics_dispatch_extra.py:568, models/ensembling/process_method.py:526-537 | RESOLVED-needed |
| SIL-W2-2 | Medium | training/_feature_importances.py:166-182 | RESOLVED-needed |
| SIL-W2-3 | Medium | training/_trainer_train_and_evaluate_parts.py:84 + training/reporting/_reporting_regression/_sensors.py:48-59 | RESOLVED-needed |
| SIL-W2-4 | Medium | evaluation/_bootstrap_clusters.py:102-116 | RESOLVED-needed |
| SIL-W2-5 | Medium | inference/predict.py:250-277 | RESOLVED-needed |
| SIL-W2-6 | Low | training/_training_loop_refit.py:184-190 | RESOLVED-needed |
| SIL-W2-7 | Low | evaluation/_bootstrap_jackknife.py:128-132 | RESOLVED-needed |
| SIL-W2-8 | Low | calibration/policy.py:613-623 | DOC |
| SIL-W2-9 | Low | models/ensembling/_correlation.py:45-54 | RESOLVED-needed |

---

## SIL-W2-1 (Medium): ensemble member predictions and target are aligned by prefix truncation, so the per-member chart compares the wrong rows

**Sites:** `src/mlframe/reporting/_diagnostics_dispatch_extra.py:568-571`; `src/mlframe/models/ensembling/process_method.py:526-537`

**Statement:** When `y_true` and `member_preds` differ in length, both are cut to the shorter length (`yt[:n]`, `mp[:n]`) with no log. Before that, `_stack_member_predictions` has already skipped `None` members and cut every member to the shortest one (`c[:_w]`), also with no log.

**Scenario:** A coarse ensemble re-scoring pass covers more rows than the target subsample, which is the case the comment itself describes. Row i of the predictions is then paired with a different row of the target. The prediction-stability panel's per-member absolute errors and its spread-vs-error relation are computed on mismatched rows. The figure saves successfully and looks valid. Separately, if one member's test preds are `None`, the stacked matrix has fewer columns than there are members, and nothing records which member is missing.

**Disposition:** RESOLVED-needed. Align by index or by an explicit row mapping. If that is not possible, skip the panel with a WARNING naming both lengths. Log in `_stack_member_predictions` when it skips a member or truncates, and record the member order on the entry.

## SIL-W2-2 (Medium): FI input replay failures are logged at DEBUG, and Captum then attributes on un-preprocessed input

**Site:** `src/mlframe/training/_feature_importances.py:166-182` (`_net_input_frame`)

**Statement:** If `pre_pipeline.transform`, `pipeline[:-1].transform` or `_apply_cat_codes` raises, the function logs at DEBUG and passes on the input of the failed step.

**Scenario:** An all-numeric MLP whose stamped pre_pipeline fails to replay (column-order drift, an unfitted stage) gets raw, unscaled and unimputed X. Captum's IntegratedGradients (and the CUDA permutation kernel) then run without error on inputs the net never saw. The importances come back finite and plausible, but they are wrong, and at default log levels nothing says so. This is the same class as the 2026-10-11 Captum fix, but that fix only covered the crash case; here nothing crashes.

**Disposition:** RESOLVED-needed. Raise the log to a throttled WARNING. When a replay step fails, return a sentinel so the caller skips net-direct FI, or uses the permutation path through `model.predict`, which does its own preprocessing, instead of attributing on the wrong representation.

## SIL-W2-3 (Medium): a train-envelope failure is logged at DEBUG, and the reported metrics are then clipped using the evaluated split's own labels

**Sites:** `src/mlframe/training/_trainer_train_and_evaluate_parts.py:84-86`; `src/mlframe/training/reporting/_reporting_regression/_sensors.py:48-59`

**Statement:** If `compute_train_envelope_stats` raises, the failure is logged at DEBUG and `None` is returned. The regression reporter then builds the clipping envelope from `targets_arr`, which is the TEST/VAL labels being scored, and clips predictions to it before computing metrics.

**Scenario:** A model extrapolates catastrophically on the test split. With train stats available it is clipped at 3 sigma of the train range. Without them it is clipped at 10 sigma of the test labels' own min/max. Clipping with the held-out labels improves the reported test RMSE/R^2 using information from the very labels being scored, and the switch is visible only as the suffix `[eval-fallback]` in a clip message, and only if clipping occurs.

**Disposition:** RESOLVED-needed. Log the train-stats failure at WARNING. In the eval-fallback branch, either do not clip the metrics (keep the sensor warning only) or record `envelope_source` in the metrics dict so the report shows that the number was produced with label-derived clipping.

## SIL-W2-4 (Medium): the clustered/block bootstrap drops failed and non-finite resamples without counting or reporting them

**Site:** `src/mlframe/evaluation/_bootstrap_clusters.py:102-116`

**Statement:** A resample whose metric raises (DEBUG log only) or comes out non-finite is dropped. The CI is built from the survivors, and the result dict carries no failure count. Only the all-failed case is reported.

**Scenario:** AUC with cluster resampling over a few large groups. Resamples that draw only negative-class clusters raise or return NaN and are dropped. The surviving resamples are conditioned on containing both classes, so the percentile CI is biased (typically too narrow) and is returned as a normal result. `bootstrap.py:494` warns when more than 25% of resamples fail; this sibling does not.

**Disposition:** RESOLVED-needed. Count failures per metric, add `n_failed` to each result, and apply the same 25% WARNING threshold as `bootstrap_metrics`.

## SIL-W2-5 (Medium): `read_trained_models` drops unreadable or mismatched models from the returned set, and the caller only learns of it from a log line

**Site:** `src/mlframe/inference/predict.py:250-277`

**Statement:** A sha256 mismatch, an unpickling failure (`except Exception` at 270) or a failed feature-order check leads to `continue`. The returned `models` dict simply has fewer entries, and nothing in the return value says a model was skipped.

**Scenario:** One of five models in an `infer/<featureset>` folder fails to unpickle after a library upgrade. `get_models_raw_predictions` and the caller's averaging then serve a 4-model ensemble as if it were the full one. The comment at line 151 says "a skipped model changes the ensemble, so none may be dropped", but the code does drop it; only a WARNING/ERROR log line says so.

**Disposition:** RESOLVED-needed. Add `strict: bool = True`, which raises on any skipped model file. Alternatively, return or attach a `skipped` list so callers can check the count. Keep the lenient mode as an explicit opt-out.

## SIL-W2-6 (Low): the loss-fallback rollback failure is logged at DEBUG, so `get_params()` can report a loss the model was not trained with

**Site:** `src/mlframe/training/_training_loop_refit.py:184-190`

**Statement:** After `set_params(RMSE)` succeeds and both refit paths fail, restoring the original params can itself fail, and that failure is logged only at DEBUG. The WARNING that follows says the fit is kept "with its original loss".

**Scenario:** A Huber-trained degenerate CatBoost model (set_params after fit raises in some versions, as the comment notes; a mixed state is possible on other wrappers) ends up with its params saying RMSE. The saved metadata, the reports and any later `clone` use RMSE, while the fitted trees were trained under Huber. The WARNING tells the operator the opposite.

**Disposition:** RESOLVED-needed. Log the restore failure at WARNING and adjust the following message so it states that the params now differ from the fitted state.

## SIL-W2-7 (Low): generic jackknife LOO failures are logged at DEBUG and dropped, which skews the BCa acceleration

**Site:** `src/mlframe/evaluation/_bootstrap_jackknife.py:128-132`

**Statement:** In `_jackknife_metric`, a leave-one-out evaluation that raises is skipped with a DEBUG log. The index-based sibling (`_jackknife_metric_idx`, line 169) uses a throttled WARNING.

**Scenario:** On a rare-positive AUC where removing a single positive leaves one class, those LOO points fail. They are exactly the influential points, so dropping them shrinks the acceleration estimate and shifts the BCa interval. Nothing above DEBUG reports this.

**Disposition:** RESOLVED-needed. Match the idx variant: use a throttled WARNING and return the dropped count so `_ci_from_samples` callers can fall back to percentile intervals when many points were dropped.

## SIL-W2-8 (Low): the calibrator-selection inner CV averages each candidate over whichever folds succeeded

**Site:** `src/mlframe/calibration/policy.py:613-623`

**Statement:** A fold whose predict raises (throttled WARNING) or whose ECE is non-finite is skipped. The candidate's mean ECE is taken over fewer folds and then compared with candidates scored on all folds.

**Scenario:** Spline fails on the one hard fold, so its average excludes that fold and comes out lower. It can then win the selection over Isotonic, which was scored on every fold. The docstring states the skip, but the comparison bias is not mentioned.

**Disposition:** DOC. Document the bias, or, preferably, require a candidate to succeed on every fold to stay eligible (or compare only over the folds all candidates share).

## SIL-W2-9 (Low): the GPU correlation failure falls back to numpy and logs only at DEBUG

**Site:** `src/mlframe/models/ensembling/_correlation.py:45-54`

**Statement:** For K>50 or N>1M the cupy path is tried, and any exception drops to `np.corrcoef` with only a DEBUG log. The DEBUG message also names the wrong file ("base.py").

**Scenario:** The result is correct but slower on large stacks, and the slowdown is invisible. This violates the project rule "Say so when a gate makes a run slower".

**Disposition:** RESOLVED-needed. Use `log_throttle(..., logging.WARNING, ...)` with K, N and the exception, fix the message text, and add a caplog test.

---

No High-severity silent failure was found in the paths read: the predict path, the save path and the HybridSelector transform either raise or log at WARNING on the failures that would change predictions.
