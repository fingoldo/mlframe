# Composite targets audit, direction 3 of 6: estimator and ensemble wrapping

Scope: `src/mlframe/training/composite/estimator/` (CompositeTargetEstimator, base extraction, predict/quantile/soft-shrink/update/from_fitted_inner), `src/mlframe/training/composite/ensemble/` (CompositeCrossTargetEnsemble, OOF holdout, stackers, calibrator, post_shim), and the training-core phases that wrap composite models (`_phase_composite_wrapping.py`, `_phase_composite_post*.py`, `_phase_composite_post_xt_ensemble/`, `_ood_lag_router.py`, MoE gate wiring, y-scale charts and metrics). The audit was read-only against origin/master a950f7e47. Transform-internal math is covered in `transforms.md` (TRF-xx) and is only cross-referenced here.

Reproductions ran as small single-process snippets (OMP_NUM_THREADS=2). Where a finding says "reproduced", the numbers come from those runs. Findings that are not marked that way were derived from reading the code.

Config defaults that matter below: `cross_target_ensemble_strategy="nnls_stack"`, `oof_holdout_source="kfold"`, `oof_holdout_frac=0.2`, `skip_wrap_pass_predict=True`, `enable_wrap_pass_watchdog=True`, `moe_gate_enabled=True`, `CompositeTargetEstimator(soft_base_shrink=True)`.

---

### EST-01 [P0] One `X` is used both for the inner model's features and for the raw base, so every call path gives wrong y-scale predictions for composite models trained behind a value-transforming pre_pipeline

- **Where**:
  - `composite/estimator/_predict.py:223-274`: `_predict_unclipped` extracts the base from `X` and calls `estimator_.predict` on the same `X`.
  - `core/_predict_main_from_models.py:570` and `core/_predict_main_suite.py:422-433`: the "CTE-RAW-X" routing gives the CTE the raw frame from before the pipeline.
  - `composite/post_shim.py:176`: `PrePipelinePredictShim.predict` gives the CTE the frame after `pre_pipeline.transform`.
  - `core/_phase_composite_wrapping.py:263,521`: the per-model hook and the wrap-pass metrics call `wrapper.predict(raw split frame)` and ignore `entry.pre_pipeline`.
  - `core/_phase_composite_post_xt_ensemble/__init__.py:105-115`: `_get_train_pred`. On a cache hit it returns the wrap-pass prediction on raw X. On a miss it calls the shim, which predicts on the pp-transformed X.
  - `composite/ensemble/__init__.py:365,632,875`: in OOF refits, `wrapped.predict(X_holdout_t)` runs on the pp-transformed holdout.
- **What**: Composite targets go through the normal per-model loop, so a linear or MLP entry gets `pre_pipeline = strategy.build_pipeline(..., imputer, scaler)` (`_phase_train_one_target_body.py:396`). Its inner model is fit on z-scored features. The wrapper keeps a single `X` argument for two different needs: the raw base column (the alpha/beta fit is on the raw scale) and the transformed feature frame (what the inner expects). Each caller chooses one of the two, so each caller gets one of them wrong:
  - With the raw frame (deployed predict, wrap-pass metrics, per-model hook, cached train preds), the inner sees unscaled features.
  - With the transformed frame (the CT ensemble shim and OOF refits), the base is read after z-scoring.

  Reproduced with y = 0.9*base + 10*f + noise (base ~ N(1000, 50)), `linear_residual`, a Ridge inner fit on `StandardScaler` output, and noise sd 1:

  | Path | RMSE |
  | --- | --- |
  | Raw X to the CTE (deployed and metrics path) | 165.8 |
  | pp-transformed X to the CTE (ensemble path) | 419.0 |
  | Correct combination: inner on pp(X), base raw | 0.99 |

  The existing test `tests/inference/test_predict_cte_raw_x.py` fits its inner on RAW features, so it cannot see this case.
- **Why it matters**: The linear and MLP composite models ship silently wrong predictions, and so do tree strategies whose pipeline encodes categories. The y-scale metrics and charts that are supposed to catch this use the same broken call, so they agree with the wrong answer. The CT ensemble then weights these components on a third, differently wrong surface.
- **Suggested fix**: Give `CompositeTargetEstimator` a separate base source. Two options:
  - `predict(X, *, base_frame=None)` / `predict_quantile(..., base_frame=None)`: extract base and groups from `base_frame` when it is given, and send `X` to the inner.
  - Store the entry's fitted `pre_pipeline` on the wrapper (`inner_pre_pipeline_`) and apply it only on the inner branch.

  Then route every caller through it: the predict entry points pass `(input_for_model, df_pre_pipeline)`, the shim passes `(transformed, raw)`, the wrap pass applies `entry.pre_pipeline` for the inner, and OOF refits pass both slices. Remove the CTE-RAW-X special case once the wrapper owns the split.
- **Test to add**: A suite-level test with a linear-model composite target (`linear_residual`, scaler pipeline). Assert that `predict_from_models`, `predict_mlframe_models_suite`, the wrap-pass y-scale val RMSE and the CT-ensemble component prediction all land within 1.5x of the inner-on-pp(X) plus raw-base oracle. Also add a unit test for `predict(X, base_frame=...)`.
- **Disposition**: COMPLETED - the wrapper owns inner_pre_pipeline_ and derives the inner frame from the suite-stage frame, with an inner_X override for callers that already applied it; PipelineCache now carries the fitted pipeline (dbe77db39; a8d1f9b06 pins both stages against an oracle at every predict entry point, test_composite_suite_persistence.py)

### EST-02 [P1] The default-ON MoE gate routes every row of a group it did not see at fit time to `lag_predict`, so on group-disjoint val/test splits it replaces the deployed ensemble with lag everywhere

- **Where**:
  - `core/_phase_composite_post_moe.py:243-257`: the gate is fit on the val split and wraps `_entries[0].model`.
  - `composite/_moe_gate.py:292`: the global fallback for unseen groups is `_lag_idx`.
- **What**: The gate learns a per-group expert choice from the val groups. Unseen groups fall back to `global_choice_`, and that is the lag failsafe whenever a lag expert exists. The xt-ensemble code calls the val split "group-disjoint (same honest regime as test)", and in that regime every test group is unseen, so every test row gets lag. Reproduced with 30 val groups where composite RMSE is 1 and lag RMSE is 8: the gate picks composite for 30 of 30 groups, `global_choice_='lag'`, and on 30 disjoint test groups the gated RMSE is 8.00 against 0.98 for the composite. This happens even right after `compute_val_veto` has just shown that the trained model beats lag by more than 10%.
- **Why it matters**: A silent, large accuracy regression in the shipped predictor, on by default whenever `group_column` and `ctx.group_ids` are set. That is exactly the grouped-panel setup that the ensemble's group-aware OOF logic targets.
- **Suggested fix**: Make the unseen-group fallback the expert with the best pooled selection-split RMSE (the not-worse-than-lag guarantee still holds per group for seen groups). Also check test-time group coverage: if the fraction of predict rows whose group was seen at fit is below a threshold, use the pooled-best expert instead of lag. At minimum, skip the MoE wrap when the val groups and train groups are disjoint (the split is then group-disjoint by construction).
- **Test to add**: Fit the gate on groups 0-29 where composite wins, predict on groups 100-129, and assert that the gated RMSE is no worse than the pooled-best expert. Reframe `test_biz_val_moe_gate.py::test_global_fallback_for_unseen_group_is_lag` to the pooled-best contract.
- **Disposition**: COMPLETED - the unseen/low-data fallback is the pooled-best expert over the matched selection rows with lag among the candidates, since such a group contributes no rows to the pooled sums the vs-lag guarantee is proved on (test_biz_val_moe_gate.py::test_global_fallback_for_unseen_group_is_the_pooled_best_expert, which fails at origin/master with 'lag' == 'composite'; the companion test keeps lag when lag is pooled-best)

### EST-03 [P1] When a component fails at predict time, `CompositeCrossTargetEnsemble.predict` drops it but keeps the other components' raw weights, which biases predictions toward 0 by the dropped weight mass

- **Where**: `composite/ensemble/_cross_target.py:551-567` (the exception is swallowed) and `:610-637` (the non-convex branch combines the surviving columns without renormalising).
- **What**: `nnls_stack` (the default strategy) and `linear_stack` are marked `is_convex=False`. When a component's `predict` raises (a missing base column, a schema drift, EST-01's `TypeError` on ndarray pp output), predict computes `sum(w_surv * p_surv) (+ intercept)`. NNLS weights sum to about 1 and the stack has no intercept, so losing a component with weight w scales every prediction by about (1 - w). Reproduced: with two components weighted 0.5/0.5 and one failing, the mean prediction is 449.4 against a mean y of 898.7. The failure is logged through `log_throttle` with the text "Excluding ... (re-normalising)", which is false for this branch. `test_composite_ensemble_linear_stack_dropout.py` pins the un-renormalised output as correct.
- **Why it matters**: One broken component silently halves the served forecast. The throttled WARNING and the wrong "re-normalising" wording make it easy to miss.
- **Suggested fix**: For the non-convex strategies:
  - NNLS: rescale the surviving weights so their sum matches the full weight sum. Better, precompute at build time the NNLS solution on each leave-one-out column subset from the OOF matrix, which keeps predict deterministic and needs no stash.
  - `linear_stack`: substitute the dropped component's OOF column mean, as the meta-stacker path already does (`_meta_col_means`).

  Make the log text branch-specific. Consider raising when the dropped weight mass is above a threshold (for example 20%), instead of serving a biased blend.
- **Test to add**: For an NNLS ensemble with one of two equal-weight components raising, assert that the prediction mean stays within 5% of the surviving component's mean. Reframe the three dropout tests to that contract.
- **Disposition**: COMPLETED - a dropped component's column is rebuilt from its own OOF mean so every surviving weight stays the one it was solved with (no refit, still deterministic); models pickled before the means were stored derive them from the stashed OOF design, and the component-failure log no longer claims to re-normalise (test_composite_ensemble_linear_stack_dropout.py: at origin/master all 3 rows are off by up to 25.5, 26%)

### EST-04 [P1] At predict time the recurrent inverses get `1.0` in place of an out-of-domain (NaN/inf) base, which corrupts the EWMA/rolling state of every later row in the batch

- **Where**: `composite/estimator/_predict.py:116-122` (`base_safe = np.where(mask, base_arr, 1.0)` and then the inverse over the whole sequence). Compare `_estimator.py:570-602`, where fit carry-forward fills non-finite recurrent inputs.
- **What**: For `ewma_residual`, `rolling_quantile_ratio`, `frac_diff`, `volatility_normalized_residual` and their `_grouped` variants, the inverse runs a recurrence over the base sequence. Fit fills non-finite bases with the previous value, and `_ewma_compute` itself also carries forward on non-finite values. Predict instead writes the literal 1.0 into the sequence before calling the inverse, so one NaN base pulls the EWMA toward 1 and the error decays over about k rows. Reproduced: `ewma_residual`, base about 1000 (random walk), y = base + N(0,1), NaN base at row 5 of a 40-row batch. The prediction at row 6 moves by 34.9 (989.7 to 954.9), and row 20 is still off by 3.3. The NaN row itself correctly gets the median fallback.
- **Why it matters**: A single missing base value silently spoils tens of neighbouring predictions, far beyond the row that was flagged. The distortion does not show up in the domain-violation counters.
- **Suggested fix**: In `_inverse_with_fallback`, for transforms with `getattr(transform, "recurrent", False)`, build `base_safe` with `_carry_forward_fill` (per column for 2-D bases, per group for grouped transforms), matching fit. Keep the 1.0 placeholder only for pointwise inverses.
- **Test to add**: For each recurrent transform, run predict twice (clean batch, and the same batch with one NaN base). Assert that every row except the NaN row matches to within 1e-9 of carry-forward semantics, and that the NaN row gets the fallback.
- **Disposition**: COMPLETED - a recurrent inverse carry-forward-fills an out-of-domain base instead of substituting 1.0, matching what fit does for its own dropped rows (test_recurrent_predict_out_of_domain_base.py: at origin/master one blanked row moved 73 of 239 other rows by up to 75.9)

### EST-05 [P2] The CT-ensemble "honest OOF gate" can never fire for the default `nnls_stack` (and in practice for `linear_stack`), because the stack weights are fit on the same OOF matrix the gate scores

- **Where**: `core/_phase_composite_post_xt_ensemble/__init__.py:841-852` (stack fit on `_oof_pred_matrix`) and `:895-997` (the gate compares the ensemble RMSE on the same matrix with the best single OOF RMSE). Also `:1015-1035` (the output calibrator is fit on the same surface after the gate).
- **What**: NNLS minimises in-sample squared error over w >= 0. Each unit vector e_k is feasible, so the ensemble's RMSE on that matrix is always at most the best single component's. `_ens_rmse > _best_single_rmse` is therefore false by construction. Ridge with alpha <= 10 on y-scale predictions behaves the same way. The only real protections left are the AR(1) lag failsafe and the dummy floor. The opt-in output calibrator is fit on the same in-sample blend, and the calibrated predictor that ships is never scored.
- **Why it matters**: The gate looks like a leakage-free check in logs and docs, but it cannot reject an over-fit stack, for example many correlated components with noise-fitted weights.
- **Suggested fix**: Score the stack with a nested split: within the OOF rows, fit the weights on K-1 folds and evaluate on the held-out fold (cross-fitted stacking), then compare with the best single component's OOF RMSE on the same rows. Evaluate the calibrator with the same nesting, or fit it inside the cross-fit.
- **Test to add**: Build a pool of one good component plus 20 pure-noise components with a small OOF n. Assert that the gate falls back to the best single component (currently it keeps the stack).
- **Disposition**: COMPLETED. For `nnls_stack` / `linear_stack` the fallback gate now compares a cross-fitted stack RMSE (`gate_stack_rmse` / `cross_fitted_stack_rmse`: 5 folds over the OOF rows, the same stack constructor fitted on four and scored on the fifth) with the best single component. On a pool of 1 good + 20 noise components at n=60 the in-sample stack beats its best single on every seed (the old blind spot). The cross-fitted stack loses on seeds 1 and 2, so the gate can now fire. The output calibrator is not cross-fitted yet. Tests: tests/training/composite/ensemble/test_combiner_invariants.py.

### EST-06 [P2] `cap_inference_components` trims non-convex stacks without refitting or renormalising, after the gate has already accepted the full stack

- **Where**: `composite/ensemble/_cross_target.py:738-822`; called at `core/_phase_composite_post_xt_ensemble/__init__.py:1049`, after the OOF gate.
- **What**: For `nnls_stack` and `linear_stack`, the top-N components keep their raw weights, so the served blend is `sum_{kept} w_k p_k` and loses the dropped weight mass. The bias is the same as in EST-03, but deterministic on every row. The trimmed predictor is never evaluated against the gate, the dummy floor or the lag failsafe.
- **Why it matters**: Enabling `max_inference_components` for latency silently biases every prediction.
- **Suggested fix**: Refit NNLS/Ridge on the OOF matrix restricted to the kept columns (the xt builder has it in scope), or renormalise NNLS weights to the original sum. Then re-run the gate on the capped predictor.
- **Test to add**: For an NNLS ensemble with weights [0.5, 0.3, 0.2] capped to 2, assert that the capped prediction mean is within 2% of the uncapped mean on the OOF surface.
- **Disposition**: COMPLETED. After `cap_inference_components`, a non-convex stack is refitted by `refit_capped_stack` with the same solver on the OOF columns it kept, so its weights describe the predictor that ships. Convex strategies renormalise at predict and are left as they are. On the test fixture, the raw-weight cap put the blend more than 2% off the target mean; the refit puts it within 2%. The capped predictor is not re-gated separately. Tests: tests/training/composite/ensemble/test_combiner_invariants.py.

### EST-07 [P2] On the time-sorted OOF holdout path, the polars branch misaligns X and y rows, in both the refit-train slice and the holdout slice

- **Where**: `composite/ensemble/__init__.py:749-753` (`train_idx` and `holdout_idx` in time order) and `:782-787` (polars `filter(mask)` keeps natural row order). Pandas uses `iloc[idx]` and stays aligned.
- **What**: When `time_ordering` is given but is not monotone, the rows are argsorted by time. `y_stack` and `y_holdout` are then indexed in time order, while the polars frames come back in ascending-index order. Reproduced with an identity component (`predict = X["yy"]`), n = 200 and reversed timestamps: pandas gives max |pred - y| = 0.0, polars gives 39.0. Components are also refit on X rows paired with the wrong y.
- **Why it matters**: OOF weights, the dummy floor and the gate run on scrambled pairs, with no error. The path is reachable with `oof_holdout_source` values other than `kfold` (`train_tail`, `external_val` without val) when `ctx.timestamps` is not sorted and the frame is polars.
- **Suggested fix**: Build the polars slices with `train_X[train_idx]` / `train_X[holdout_idx]` (polars supports integer-row gather), or sort `train_idx` and `holdout_idx` and index y the same way.
- **Test to add**: The identity-component repro above, parametrised over pandas and polars, with monotone, reversed and shuffled `time_ordering`.
- **Disposition**: COMPLETED. The polars branches of the OOF holdout, both the single-split path and the k-fold path, now gather rows by position (`train_X[idx]`), as the pandas `.iloc` branch does. A boolean mask kept row order while `y_train_full[holdout_idx]` follows index order (time order on the sorted-holdout path), so every holdout row met another row's target. The same fix went into `feature_stacking.composite_oof_predictions`, the xt-ensemble `_slice_frame_rows`, `slicing._row_select` and `row_level_average_importance._subset_rows`; the gather keeps Enum / Categorical dtypes. Regression tests in tests/training/composite/ensemble/test_frame_carrier_parity.py: an identity component's OOF prediction equals the holdout y exactly for pandas and polars under monotone, reversed and shuffled time, with k=1 and k=3; the three row slicers return the pandas rows in the index's order. Before the fix 4 and 10 of these failed respectively.

### EST-08 [P2] The wrap-pass watchdog is off by default, cannot detect the failures it names, raises a false alarm on every `quantile_residual` run, and swallows its own errors at DEBUG

- **Where**: `core/_phase_composite_wrapping.py:438-495` (the `skip_predict` early `continue`), `:634-758` (the watchdog) and `:759-770` (per-split failures logged at DEBUG); config `skip_wrap_pass_predict=True`.
- **What**:
  - (a) With the default `skip_wrap_pass_predict=True` the metric block, and the watchdog nested inside it, never run. The log line and docstring still say the watchdog "covers correctness".
  - (b) The universal check compares `wrapper.predict(X)` with `inverse(inner.predict(X), base(X), spec params)`. Both sides use the same inner, the same frame and the same params, so it can only find clip or soft-shrink differences. It cannot find the causes it lists: a wrong feature frame (EST-01), a lost TTR state, or a different base at predict time.
  - (c) `_ADDITIVE_TRANSFORMS` includes `quantile_residual`, whose inverse scales by the per-bin IQR (`y = T*IQR + median`). The "y-MAE == T-MAE" check therefore fires whenever the IQR is not 1, which is almost always.
  - (d) It also includes `linear_residual_grouped`, whose forward needs `groups=` and is called without them. The same applies to recurrent or grouped inverses in the universal check. These raise and are swallowed at DEBUG.
  - (e) The set leaves out additive transforms such as `second_diff`, `additive_residual`, `theilsen_residual` and `linear_residual_multi_robust`.
  - (f) Any exception inside a split's metric block, including `wrapper.predict` itself, drops that split's y-scale metrics with only a DEBUG log.
- **Why it matters**: The watchdog meant to catch wrapper math errors misses the real ones, and it trains operators to ignore its warnings (c). Composites whose predict raises simply disappear from the y-scale verdict.
- **Suggested fix**:
  - Run a cheap watchdog even when `skip_predict=True`, for example on a capped val sample.
  - Replace the tautological check with an independent oracle: re-extract the base from the raw split frame, recompute T with `transform.forward(y_split, base)`, and compare the wrapper's y-MAE with a y-MAE rebuilt as `inverse(T_true + (T_hat - T_true))`. Also compare against the entry's pp-aware inner prediction (catches EST-01).
  - Derive the additive set from `_soft_shrink.ADDITIVE_BASE_TRANSFORMS`, which lists the transforms that are truly linear in base, and drop `quantile_residual`.
  - Pass `groups` for grouped specs.
  - Raise per-split failures to WARNING.
- **Test to add**:
  - A default-config suite run with a corrupted base column at predict time, asserting a WARNING.
  - A `quantile_residual` run with a correct wrapper, asserting no watchdog WARNING.
- **Disposition**: COMPLETED. The watchdog moved to `core/_composite_wrap_watchdog.py`; each part is handled as follows. (a) With the default skip of the metric block it checks a 2,000-row val sample per composite. (b) The self-comparing universal check is gone. In its place, an independent base-read check compares the base the wrapper reads at predict with the spec's base columns read from the split frame. The additive check takes its true T from the split's real y and base, reads T-hat through the wrapper's own inner input (its pre-pipeline applied, the group column dropped), and skips rows the y-clip pinned or that lie outside the fitted base range, so the clip and the soft shrink cannot trip it. (c)/(e) The additive set is read off `Transform.additive_in_t` (00ce5dbc6). (d) Grouped transforms get the wrapper's `group_column` groups. (f) A check that cannot run, and a split whose predict raises, log at WARNING. Tests: tests/training/composite/estimator/test_wrap_watchdog_oracle.py (the skip-path and split-failure cases fail on the pre-fix wrapping module) and the reframed tests/training/test_regression_watchdog_yscale_object_target.py.

### EST-09 [P2] The default-ON `soft_base_shrink` guard is inert on every wrapper built by `from_fitted_inner`, which covers all suite-trained composites and every OOF refit

- **Where**: `composite/estimator/_from_fitted.py:114-121` (the `fitted_params_` built there has no `base_fit_range`); `composite/estimator/_soft_shrink.py:127-140` (`is_enabled` needs that key); only `fit()` calls `capture_base_fit_range` (`_estimator.py:731`).
- **What**: The suite never calls `CompositeTargetEstimator.fit`. It wraps already-fitted inners (`_phase_composite_wrapping.py:235,399`), and the OOF refits wrap as well (`ensemble/__init__.py:357,624,867`). Discovery never stamps `base_fit_range` into spec params (grep finds no producer). So the soft shrink and deep-OOD lag fallback, described as default ON with in-range predictions "byte-identical", never runs in production. It runs only for users who call `.fit` directly.
- **Why it matters**: The OOD extrapolation protection advertised for unseen-group tails is absent from the path that ships. This is the "corrective mechanism default ON" rule failing silently.
- **Suggested fix**: Stamp `base_fit_range` (lo, hi, iqr per base column) next to the T-train envelope in `prune_equivalent_composite_specs` / discovery. That code already has the train base arrays, so the stamp is cheap. Alternatively, accept `base_train` in `from_fitted_inner` and call `capture_base_fit_range`. Also set `target_name_` so the smart fallback can resolve the causal lag.
- **Test to add**: Wrap a `linear_residual` inner through the suite's wrapping phase, predict on a base 10 IQRs beyond the train max, and assert `soft_shrink_info_["n_shrunk"] > 0`.
- **Disposition**: COMPLETED - the suite's own wrapping path already passed `base_train` (and `target_name`) to `from_fitted_inner` by the time this was worked, so the deployed wrapper captured its base range; the three ensemble OOF refit sites did not, so the weighting surface was computed with wrappers that lacked the shrink the served model applies. They now pass their training base (`base_full[valid]` / `base_stack[valid]`). test_oof_refit_wrappers_carry_base_range.py records every `from_fitted_inner` call in a composite suite run: 5 of 6 wrappers lacked a base before, 0 after. The pinned test the former no-range test was split into `test_from_fitted_inner_without_a_base_range_is_a_documented_noop` and `test_from_fitted_inner_with_a_base_range_shrinks_deep_ood_rows`, a with-base case asserting `n_shrunk > 0` on a deep-OOD row

### EST-10 [P2] `predict_quantile` returns zero-width intervals on fallback rows and crossed quantiles for sign-flipping multiplicative inverses

- **Where**: `composite/estimator/_predict.py:410-445` (the ordering guards cover only `ratio`, `logratio` and `reciprocal_residual`) and `:504-520` (each quantile column goes through `_inverse_with_fallback`).
- **What**:
  - (a) Rows that fail the base domain check, deep-OOD rows and rows whose inverse is not finite get the same `y_train_median` (or lag) in every quantile column. The interval collapses to a point exactly on the rows where uncertainty is highest.
  - (b) `centered_ratio` (`y = T*(base+c)`), `rolling_quantile_ratio` and `rolling_quantile_ratio_grouped` (`y = T*rolling_median(base)`) multiply T by a base-derived factor that can be negative (TRF-02 shows `base + c` crossing 0 just below the train min). That reverses the quantile order (q10 > q90) without warning. No monotonicity check or sort runs after the inverse.
- **Why it matters**: The conformal and CQR bands built on `predict_quantile` inherit invalid (crossed) or overconfident (zero-width) intervals.
- **Suggested fix**:
  - For fallback rows, return NaN, or the train-y empirical quantiles at each alpha (`np.quantile(y_train, alpha)` stored at fit).
  - Replace the per-transform guard list with a generic check: after inverting, detect rows where the column order is not monotone in alpha and either sort them (a valid rearrangement, per Chernozhukov et al.) or raise for transforms not declared monotone-increasing.
- **Test to add**: `centered_ratio` with a predict base below `-c`: assert that the quantile columns are non-decreasing. Domain-violating rows: assert that the q10 and q90 values differ, or are NaN.
- **Disposition**: COMPLETED. (a) Fit stores `y_train_quantile_grid` (train-y quantiles at alpha = 0..1 in 101 steps). In `predict_quantile`, rows that fail the base domain or are deep OOD take the train-y quantile at each alpha, so they keep a real interval instead of the median in every column (`fallback_predict='nan'` keeps NaN). (b) Every multi-alpha prediction is monotone-rearranged per row in alpha order (Chernozhukov et al.), which covers any inverse whose base factor turns negative. Tests: tests/training/composite/estimator/test_predict_quantile_contract.py (the fallback case fails pre-fix). Ordering is checked for every transform; on this fixture a base below -c routes to the fallback rather than inverting crossed, so the rearrangement is a guard. The quantile-parity test now expects the per-alpha fallback.

### EST-11 [P2] The dummy-floor gate and the `oof_weighted` baseline compare the dummy's VAL-split RMSE with components' train K-fold OOF RMSE

- **Where**: `core/_phase_composite_post_xt_ensemble/__init__.py:694-704` (`_dummy_floor_rmse` from `data[strongest][primary_metric]`, where `primary_metric` is `val_RMSE`) and `:860-869` (the same value becomes `baseline_oof_rmse`).
- **What**: The floor and the "OOF" baseline come from a different split than the RMSEs they are compared against. On a drifting or volatility-shifted val period the gate either lets weak components through (val harder than train) or drops good ones (val easier). `from_train_metrics` records `baseline_source="oof"` for this value, which is wrong. Its own docstring calls cross-scale baselines "apples-to-oranges".
- **Why it matters**: Component selection and weighting depend on split drift rather than skill.
- **Suggested fix**: Evaluate the strongest dummy on the same OOF rows (dummies are cheap: fit per fold, or reuse the in-pool `lag_predict` OOF column, which is already there when injected). Use that as both the floor and the baseline. Keep the val-split dummy only for reporting.
- **Test to add**: Build a val period with twice the train noise and assert that a component beating the dummy on OOF is not dropped by the floor.
- **Disposition**: COMPLETED. `same_split_dummy_rmse` measures the dummy floor, and the `oof_weighted` baseline, on the same OOF rows as the components: the in-pool `lag_predict` OOF column, else the strongest constant strategy on the OOF targets. The val-split value is used only when neither applies, with a WARNING. Tests: tests/training/composite/discovery/test_scorer_invariance.py (fails pre-fix).

### EST-12 [P2] `sample_weight` is threaded into the OOF refits but dropped by the stack solvers, the OOF RMSEs, the gate and the output calibrator on the general CT path

- **Where**: `core/_phase_composite_post_xt_ensemble/__init__.py:841-852` (`from_linear_stack` / `from_nnls_stack` are called without `sample_weight`), `:644-649` (unweighted `_oof_rmses`), `:897-909` (unweighted gate) and `:1021` (calibrator without weights). The MTR branch at `:175-181` does pass them.
- **What**: On a weighted suite, each component's OOF predictions come from weighted refits, but the weights are then solved, ranked and gated as if every row counted equally. `compute_oof_holdout_predictions` also does not return the holdout-aligned weights, so the caller cannot apply them.
- **Why it matters**: The ensemble optimises a different objective from the one the suite (and its MTR sibling) declares. Rows that are heavily weighted in the business sense get no extra influence on the blend.
- **Suggested fix**: Return the holdout-aligned weight vector from `compute_oof_holdout_predictions` (an `oof_sample_weight` field), and pass it into the stackers (both already accept `sample_weight`), the RMSE computation, the gate and `fit_output_calibrator`.
- **Test to add**: Build two components, each good on a different half of the rows, with weights concentrated on one half. Assert that the NNLS weights move toward the component that is good on the heavy half.
- **Disposition**: COMPLETED. `compute_oof_holdout_predictions(..., return_rows=True)` returns a fourth element: each OOF row's position among the (possibly subsampled) train rows. It is `None` on the external-holdout path, whose rows are the val frame. The public signature is kept via `__signature__`. The xt-ensemble phase maps the suite weights onto the OOF rows (`_crossfit.oof_row_weights`) and weights the per-component OOF RMSEs (`column_rmses`), the NNLS / linear stack solve, the cross-fitted fallback gate (`gate_stack_rmse` / `cross_fitted_stack_rmse` take `sample_weight`) and the output calibrator. Regression tests in test_combiner_invariants.py: the returned rows align `y[rows]` with the holdout y; with weights on the half where a component is the expert, its NNLS weight rises and its weighted RMSE falls. Neither API existed before. To keep `ensemble/__init__.py` under 1000 lines, the external-holdout OOF path moved to `ensemble/_oof_external.py` (re-exported).

### EST-13 [P2] The CT_ENSEMBLE val/test metrics and charts describe the pre-MoE predictor, not the model that ships

- **Where**: `core/_phase_composite_post_xt_ensemble/__init__.py:1125-1160` (metrics stamped into `cross_target_ensemble_metrics` and charts rendered) runs before `core/_phase_composite_post.py:279` (`run_composite_moe_and_value_report`) / `_phase_composite_post_moe.py:251` replaces `entries[0].model` with `_MoEGatedDeployableModel`.
- **What**: The suite-end verdict (`_phase_composite_post_summary.py`) and the targets-performance table read metrics for a predictor that is later wrapped by the MoE gate. Given EST-02, the shipped model can be much worse than the reported one.
- **Why it matters**: The verdict can report a win over the dummy for a model that is no longer the one deployed.
- **Suggested fix**: Re-score the ensemble slot after the MoE wrap (val/test predict on the final `entries[0].model`) and overwrite `cross_target_ensemble_metrics`. Alternatively, run the MoE step inside the builder before scoring.
- **Test to add**: With MoE enabled, assert that `cross_target_ensemble_metrics[...]["test_RMSE"]` equals the RMSE of `models[..]["_CT_ENSEMBLE__t"][0].model.predict(test)`.
- **Disposition**: COMPLETED. After the MoE gate wraps `entries[0].model`, `_restamp_shipped_metrics` re-scores that shipped wrapper on val and test and overwrites `cross_target_ensemble_metrics[...]['val_RMSE'/'val_MAE'/'test_RMSE'/'test_MAE']`. The `model_name` gets a `+MoE` suffix. A split whose predict fails has its numbers removed instead of left describing the pre-MoE stack. `run_composite_moe_and_value_report` takes `test_df` / `test_idx`, and `run_composite_post_processing` passes them. Regression test `test_the_ensemble_metrics_describe_the_model_that_ships`: the recorded test RMSE equals the RMSE of `models[...]['_CT_ENSEMBLE__t'][0].model.predict(test)` and replaces the stale stack number. It fails before the fix.

### EST-14 [P2] A streaming `update()` refit leaves the soft-shrink base range at the dead regime, and its T-clip refresh leaves out the widening to the observed range that `fit()` applies

- **Where**: `composite/estimator/_update.py:160-197`.
- **What**: After a drift refit, `alpha`, `beta`, the y-clip, the median and the T-clip are refreshed from the buffer, but `fitted_params_["base_fit_range"]` is not. For `linear_residual` (which is in `ADDITIVE_BASE_TRANSFORMS`), bases from the new regime that lie outside the old train range are soft-shrunk back toward the old boundary, and rows more than 3 IQRs out are sent to the lag/median fallback. That undoes the drift correction, which is the same failure the comment at `:175-178` describes for the y-clip. Separately, the refreshed T-clip is `median +/- 10*MAD` without `min(..., T_observed_min)` / `max(..., T_observed_max)`. The comment claims it "matches fit()", but on heavy-tailed buffers it clips values fit() would keep.
- **Why it matters**: The streaming correction is partly cancelled by the default-ON guard.
- **Suggested fix**: Call `_soft_shrink.capture_base_fit_range(self, transform, buffer_base)` when a refit fires, and reuse `t_train_envelope` (from `discovery/_t_equivalence.py`) so that `fit`, `from_fitted_inner`, discovery and `update` share one envelope formula.
- **Test to add**: Fit on base ~ U(0, 10), stream a regime at base ~ U(50, 60) with a new alpha until a refit fires, and assert that predictions on the new regime are neither shrunk nor sent to the fallback.
- **Disposition**: OPEN

### EST-15 [P3] In the default shuffled K-fold OOF, recurrent composite components run their EWMA/rolling state over gapped (train) and scattered (holdout) row sequences

- **Where**: `composite/ensemble/__init__.py:531` (`KFold(shuffle=True)` when there is no time or group signal; the xt builder forces `_time_ordering=None` for kfold at `core/_phase_composite_post_xt_ensemble/__init__.py:475`) and `:579-632` (`transform.forward` / `wrapped.predict` on fold subsets).
- **What**: Each holdout fold is about 20% of rows spread across the series, so the inverse EWMA over them has no real time adjacency. The OOF RMSE of `ewma_residual`, `frac_diff` and similar components is therefore meaningless, which biases their weights, usually downward. This is a consequence of TRF-05 (batch-composition dependence) in the ensemble path.
- **Why it matters**: Recurrent composites are mis-weighted or dropped by the dummy floor for reasons unrelated to their skill.
- **Suggested fix**: For components whose transform is `recurrent`, use contiguous-block K-fold (`KFold(shuffle=False)` or `TimeSeriesSplit` on the time order), or exclude them from shuffled OOF and fall back to the train-RMSE proxy with a WARNING.
- **Test to add**: Create an AR(1) target where `ewma_residual` is the true DGP. Assert that its OOF RMSE under the chosen splitter is within 20% of its contiguous-holdout RMSE.
- **Disposition**: COMPLETED. With no time or group signal, the ensemble's outer K-fold OOF split comes from `_plain_oof_splitter`. It gives contiguous blocks (unshuffled KFold via the factory) whenever any component's transform is `recurrent`, so the EWMA / rolling / frac-diff state runs over adjacent rows. On unordered rows contiguous blocks are an ordinary partition. Tests: tests/training/composite/discovery/test_splitter_contract.py (fails pre-fix).

### EST-16 [P3] The per-fold transform refit drops `groups` and `sample_weight`, and falls back to the full-train params at DEBUG level

- **Where**: `composite/ensemble/__init__.py:578-582` and `:835-839`.
- **What**: `transform.fit(y_stack[valid], base_stack[valid])` passes no `groups`. Grouped transforms then raise and silently reuse the global `spec["fitted_params"]`, which were fit on rows that include this fold's holdout (the optimism the refit exists to remove). Weight-aware transforms (`linear_residual`) are refit without weights even when the deployed spec was weighted. The external-holdout path (`:328-332`) never refits at all.
- **Why it matters**: The OOF surface is slightly optimistic for grouped transforms and uses a different estimator for weighted suites.
- **Suggested fix**: Use the estimator's signature-gated kwargs (`_callable_accepts_param`) to pass `groups` and `sample_weight` fold slices, and log the fallback at WARNING with the transform name.
- **Test to add**: Spy on `linear_residual_grouped.fit` in a K-fold OOF call and assert that it receives `groups` of fold length.
- **Disposition**: COMPLETED. Both refit sites now go through `_refit_fold_params`, which calls `call_transform` with the fold's `groups` (the suite's row-aligned `group_ids`, sliced to the fold's train rows) and `sample_weight`; the gateway passes each one only where the fit declares it. The forward gets the fold's groups the same way. A refit that still fails logs at WARNING, naming the transform. The OOF wrapper takes the deployed component's `group_column`, so a grouped component predicts its holdout. Before this, a grouped component raised in the forward on every fold and dropped out of the ensemble entirely. The external-holdout path fits on the full train and predicts a separate frame, so the full-train params are the correct ones there: no refit is needed. Tests: tests/training/composite/ensemble/test_oof_fold_refit_groups_and_weights.py (both fail pre-fix).

### EST-17 [P3] OOF refits reuse the entry's pre_pipeline, fitted on the full train (including supervised MRMR/RFECV selection that saw each fold's holdout y)

- **Where**: `composite/ensemble/__init__.py:64-95` (`_transform_pair_via`: the "already fitted, the suite-normal case" branch).
- **What**: The docstring deliberately mirrors deployment. But the supervised feature selection inside `pre_pipeline` was fit with y on every train row, so the selected feature set carries information from each fold's holdout. The OOF RMSE is therefore optimistic for FS-pipelined components compared with plain ones, which tilts the NNLS weights toward them.
- **Why it matters**: A mild, systematic bias in component weighting and gating.
- **Suggested fix**: Use the existing unfitted-clone branch when the pipeline contains a supervised selector (clone and fit on the fold-train slice), at least for small n, and keep the cheap reuse for purely unsupervised steps (imputer, scaler).
- **Test to add**: Use a pipeline with MRMR on a target where one noise feature correlates with y only in the holdout rows. Assert that the OOF RMSE of the FS component does not beat the fold-refit baseline by more than noise.
- **Disposition**: OPEN

### EST-18 [P3] The five `moe_*` constructor parameters of `CompositeTargetEstimator` are never read

- **Where**: `composite/estimator/_estimator.py:171-175,211-218`.
- **What**: `moe_gate_enabled`, `moe_shrink_rtol`, `moe_tie_rtol`, `moe_min_group_rows` and `moe_failsafe` are stored and advertised (the comment says "Default ON ... never worse than the lag failsafe"), but nothing in the estimator uses them. The MoE gate is configured only from `CompositeTargetDiscoveryConfig` in `_phase_composite_post_moe.py`. They still show up in `get_params`, in grid searches and in the repr.
- **Why it matters**: Users tuning or disabling these on the estimator get no effect and no error.
- **Suggested fix**: Either remove them (and point the docs to the discovery config), or implement an estimator-level gate. Do not keep dead public parameters.
- **Test to add**: A meta-test asserting that every `__init__` parameter of `CompositeTargetEstimator` is read by at least one method.
- **Disposition**: OPEN

### EST-19 [P3] The `lag_predict` component that ships in CT_ENSEMBLE is never fit, so NaN lag rows at predict time are imputed with the median of the predict batch itself

- **Where**: `core/_phase_composite_post_xt_ensemble/__init__.py:216` (constructed unfitted and deployed through the ensemble or the AR(1) failsafe); `core/_phase_composite_post_lag_predict.py:83-95` (with `_impute_value is None`, it uses `np.nanmedian` of the batch).
- **What**: Only the OOF clones and the MoE copy (`_phase_composite_post_moe.py:189`) call `.fit`. The deployed instance fills group-start or missing lags from the batch it is predicting on, so results depend on batch composition and a single-row batch with a NaN lag returns 0.0.
- **Why it matters**: Non-reproducible predictions, and the fill value uses information from the test batch.
- **Suggested fix**: Call `_lag_model.fit(filtered_train_df)` when injecting it (as the MoE path does), and make `predict` raise, or use a stored constant, when unfitted.
- **Test to add**: Predict the deployed lag component on two batches that differ only in their other rows, and assert that the fill for a shared NaN row is identical.
- **Disposition**: COMPLETED. The injected `lag_predict` is fitted on the train frame when it is added to the cross-target ensemble; if its lag column cannot be read there it is not injected, with a WARNING. An unfitted model no longer takes the median of the batch it is predicting to fill a missing lag: it raises `NotFittedError`. The MoE path already fitted its copy. Tests: tests/training/composite/estimator/test_cte_row_purity.py (a shared NaN row gets the train median in any batch; unfitted raises).

### EST-20 [P3] `predict` / `predict_quantile` change shared state without synchronisation

- **Where**: `composite/estimator/_predict.py:169-178` (`runtime_stats_[...] += ...`) and `composite/estimator/_soft_shrink.py:323-337` (`self.soft_shrink_info_ = ...`).
- **What**: Concurrent predict calls on one fitted wrapper (a threaded server, `joblib` threading backend) lose counter increments. Worse, `soft_shrink_info_` (per-row masks) can describe another thread's batch, so a caller that reads the flags after its own predict gets the wrong rows.
- **Why it matters**: The monitoring counters and per-row OOD flags are not reliable under concurrency.
- **Suggested fix**: Guard the counter update with a lock that is excluded from `__getstate__`, and return the shrink info from an explicit `predict(..., return_info=True)` instead of storing it on the instance, or store it in thread-local storage.
- **Test to add**: 8 threads x 100 predicts; assert `runtime_stats_["predict_calls"] == 800`.
- **Disposition**: COMPLETED. The `runtime_stats_` read-modify-writes run under a module-level lock (nothing lock-shaped on the picklable estimator), and the callback reports a snapshot taken inside it. `soft_shrink_info_` is now a property backed by a per-thread `WeakKeyDictionary`, so each caller reads the flags of its own batch. Test: 8 threads x 150 predicts with the switch interval cut to 1 us. Before the fix, threads read other threads' shrink flags on every run; after it, all 1200 calls are counted and every thread sees its own batch.

### EST-21 [P3] `from_fitted_inner` cannot express grouped transforms or recurrence continuation

- **Where**: `composite/estimator/_estimator.py:298-350` and `composite/estimator/_from_fitted.py:31-49`: there is no `group_column` or `recurrence_continuation` argument.
- **What**: A grouped spec (`linear_residual_grouped`, `*_grouped`) wrapped by the suite raises "requires groups but group_column is not configured" on every predict. The per-model hook swallows it at WARNING, and the wrap pass keeps the entry in T-scale. A spec that relies on `recurrence_continuation` cannot opt in. These transforms are not in the default discovery list, but they are registry transforms that a user can enable.
- **Why it matters**: Enabling a grouped transform in discovery gives composites that never predict on the y scale, with only a WARNING.
- **Suggested fix**: Add `group_column` and `recurrence_continuation` to `from_fitted_inner` and have the wrapping sites pass them from the spec / discovery config. Alternatively, reject grouped transforms at discovery-config validation until that is done.
- **Test to add**: Wrap a `linear_residual_grouped` inner via the suite wrapping phase and assert that `predict` returns y-scale values.
- **Disposition**: OPEN

### EST-22 [P3] Routers and vetoes chosen on the val split are then reported with val-split metrics as if those were held-out

- **Where**: `core/_phase_composite_post_xt_ensemble/__init__.py:916-972` (`compute_val_veto`, `build_ood_lag_router`, `build_volatility_lag_router` all select on `filtered_val_df`) and `:1128-1140` (VAL metrics stamped for the same predictor); `_ood_lag_router.py:99-110`.
- **What**: The val split chooses among the trained model, lag and the routed variants, then the chosen predictor's val RMSE is reported. That number is biased optimistic. The MoE value report and gate (`_phase_composite_post_moe.py`) are also built on val.
- **Why it matters**: The suite-end verdict compares a val number that was used for selection against dummies, which overstates the lift.
- **Suggested fix**: Tag the val metrics of val-selected predictors as selection-biased in metadata and in the verdict, and base the verdict on test (reporting only) or on a nested split.
- **Test to add**: Assert that metadata marks `val_RMSE` as `selection_biased=True` whenever a router or veto was chosen on val.
- **Disposition**: COMPLETED. `cross_target_ensemble_metrics[tt][t]['val_selection_biased'] = True` is now set whenever the deployed predictor was chosen on val. That covers the AR1-failsafe val veto plus the OOD and volatility lag routers, now carved into `_phase_composite_post_xt_ensemble/_lag_routing.attach_val_selected_lag_routers`; the builder shrank from 971 to 941 lines and the module from 1095 to 1066 LOC, with both ratchets lowered. It also covers the MoE gate (`_restamp_shipped_metrics`). The suite-end cross-target verdict row appends `[val-selected: lift optimistic]` when its shown number is such a val metric. The composite-vs-raw verdict itself reads test (DSC-18). Regression test `test_a_val_selected_ensemble_is_flagged_and_tagged_in_the_verdict` fails before the fix. Also in this change: test_oof_refit_wrappers_carry_base_range.py relied on INT-11's phantom spec (an untrained, floor-dropped spec was still OOF-refit); it now disables the ship floor so a trained composite is refit.
