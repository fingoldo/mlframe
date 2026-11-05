# Wave 1 (2026-11-05), attack class A9: error handling and silent failures

Read-only audit by one Opus 5.5 low-effort agent. Eight defects; the cited code was read on the current tree for each. No file was edited and no tests were run. The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| X9-01 | High | `src/mlframe/feature_selection/hybrid_selector.py:525-527`, `:915-916` | If replaying the fitted feature-engineering recipe fails at transform time, the engineered columns are dropped. `transform` then quietly drops any selected column that is missing, and if none are left it returns the frame's first column. |
| X9-02 | Medium | `src/mlframe/models/ensembling/score_gate.py:306-310`, `:408-412`; also `:285-292` | When the catastrophic-member drop check raises, the run carries on without dropping anyone and logs only at debug unless `verbose`. A drop that does happen is also logged only when `verbose`. |
| X9-03 | Medium | `src/mlframe/training/_training_loop_objectives.py:136-159` | Wrapping a model in `MultiOutputClassifier` turns off early stopping and callbacks with no log line. If that `set_params` patch fails, it is logged at debug only. |
| X9-04 | Medium | `src/mlframe/inference/predict.py:228`, `:247`, `:250-251` | Models skipped for a sha256 mismatch or a read failure are logged through `log_throttle` with one fixed key per failure kind. The key allows at most 5 messages per process, so later skips are silent and the ensemble just has fewer members. |
| X9-05 | Medium | `src/mlframe/training/_training_loop_refit.py:167-192` (same pattern at `:336-352`) | The loss-fallback refit calls `set_params` on the live model before `fit`. If both refit paths fail, the warning says "Keeping the degenerate fit", but the model already carries the new loss and metric params. |
| X9-06 | Low | `src/mlframe/training/honest_diagnostics.py:592-594`, `:249-251`, `:278-280`; `:619-621` | When a bootstrap-CI or RMSE block fails it becomes `status: skipped` and is logged at debug. The decision-threshold block (`:619`) also logs at debug, but its result line is printed at info afterwards, so only the others are truly quiet. |
| X9-07 | Low | `src/mlframe/inference/predict.py:233-240` | If the library-version check on a model's `.meta.json` sidecar raises, the model is loaded anyway and the failure is logged at debug. |
| X9-08 | Low | `src/mlframe/training/honest_diagnostics.py:236-238`; `src/mlframe/calibration/ensembling.py:161-163`, `:181-183` | Moves to a slower path are logged at debug, not through `log_throttle`, which breaks the project's own "gates that make a run slower must log" rule. The numbers come out the same. |

### X9-01 (High)

If `self._mrmr_member.transform(X)` raises on serving data (a dtype change, a new category, a GPU fault), `_augment` emits `warnings.warn` and returns the raw frame without the `eng_N` columns. `transform` then builds `keep = [c for c in self.raw_selected_ if c in X_aug.columns]`, which silently drops every engineered column that was selected. If all selected columns were engineered, it returns `X_aug.iloc[:, :1]`, an arbitrary raw column. The tree-product columns are dropped the same way when their raw source columns are absent (`:531`). A model fed by position (numpy arrays, a `Pipeline` with no name check) then predicts on fewer or shifted columns, and the user gets wrong predictions, not an error. The warning shows once per call site under Python's default filter and is hidden in most serving setups. `transform` should raise when a selected column cannot be produced.

### X9-02 (Medium)

The K>2 check (absolute MAE against the target) and the K=2 check sit in a broad `except` that logs `"proceeding without drop"` at debug when `verbose=False`. Any failure in that block (a bad target array, a reshape error, a mismatch in tag length) leaves a member that is 10x+ worse inside the blend, and nothing at warning level shows it. The reported ensemble score is degraded with no visible cause. The drop itself, which changes which members make up the ensemble, is also logged only under `verbose`; it is recorded in `res` under `_`-prefixed keys that callers filter out.

### X9-03 (Medium)

For multilabel targets the inner model's `early_stopping`, `callbacks` and `early_stopping_rounds` are cleared before wrapping, and nothing is logged. A user who configured early stopping, say with `n_estimators=5000`, gets each per-label model trained for the full budget: overfitting and long fits, and the logs say nothing. If `get_params`/`set_params` raises, the debug-only `suppressed:` line lets the wrapper go ahead with early stopping still on, which is the very failure mode the block was written to prevent.

### X9-04 (Medium)

`log_throttle` is pyutilz's `log_throttled`, capped at `max_count=5` per key per process (`src/mlframe/utils/log_throttle.py`). The keys `"predict_model_sha256_mismatch"` and `"predict_model_read_failed"` are fixed strings. In a long-running serving process, after 5 failures across all calls and featuresets, every later corrupt or unreadable model file is skipped with no log. The function returns a smaller `models` dict that callers average, so predictions change with no sign. A failure that excludes a model should log every time, or the key should include the file name.

### X9-05 (Medium)

In the in-place refit path, `model_obj.set_params(**_new_loss_params)` succeeds and then `fit` raises; the fresh-instance rebuild also fails. The function returns `None` while `model_obj` advertises the fallback loss, and the backends this was not checked on may also be left partly fitted. Saved metadata and later refits then read a loss the trained booster never used. The original params are not restored.

### X9-06 (Low)

These blocks feed the honest test-split report. A failing block is stored as `{"status": "skipped", ...}` and logged at debug, so a run that loses its headline CIs prints nothing at warning level. The user sees missing intervals in the payload and nothing in the log that explains them.

### X9-07 (Low)

The sidecar check (`strict=False`) exists to warn about library-version drift. If the check itself raises, for example on a malformed sidecar, that warning never happens, and any later wrong behaviour from drift has no log trail.

### X9-08 (Low)

The fused-bootstrap fallback and the odds-combine fallback from cupy to CPU log at debug only. The numbers are identical, so this is a breach of the logging rule only. It is listed because the earlier ~50x MI regression came from exactly this pattern.

## Not covered

- Most of the roughly 2900 broad-except sites in `feature_selection/filters/*` (the GPU MI, CMI, FE and MRMR families), `feature_engineering/transformer/*`, `reporting/*`, `metrics/_core_numba_warmup.py` and `utils/_param_oracle.py` were not read individually.
- Whether LightGBM's own category remapping makes `lgb_shim.py:589` and `:666` (debug-only category realignment failures) harmless was not checked, so they are not reported.
- `_handle_fit_failure`'s OOM, CPU-fallback and retry branches were only skimmed; retry bounds were not confirmed.
- `assert` usage: every sampled hit is a type-narrowing internal invariant, not a check on user input; the full list was not reviewed.
- Re-raises that drop the original cause or put secrets in the message: no repo-wide search was done.
