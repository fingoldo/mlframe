# Training loop: early stopping, loss recommendation, prediction path

Source: production training log of `jobsdetails_shuffled`, 2026-09-20 12:56-14:05.

- **Disposition**: 7 findings, 7 RESOLVED / 0 PARTIAL / 0 TODO / 0 REJECTED / 0 NOT A DEFECT. Rows mirrored in [_TRACKER.md](_TRACKER.md).

---

## TRN-01 (P0) -- a robust ES surface stops protecting the reported metric

`_phase_train_one_target.py:141-155` aligns `eval_metric` to whatever objective the loss recommender chose:

```python
if _value.startswith("Huber"):
    # Huber:delta=X IS a valid CB eval_metric (CatBoost 1.0+);
    # match the loss exactly so ES tracks the same surface the optimiser descends.
    return ("eval_metric", _value)
```

The comment above it records the failure this was built to fix: an RMSE objective early-stopping against an MAE
eval_metric plateaued at iter=147 and under-converged. That fix is correct for the direction it addresses.

It has an opposite failure mode, and this run hit it. `target_hours_to_hire` got Huber from the recommender
(`excess_kurt=10.25 > 1.5 (extreme tails, > 10.0) -- Huber bounded-influence loss keeps the gradient ...`), so ES
tracked Huber on val. Huber's whole purpose is to stop caring about large residuals -- which is exactly the population
that RMSE and R2 are made of. The result:

```
es_best_iter: 999                       (of iterations=1000 -- ES never fired)
VAL  RMSE=368.65  R2=-4.61              per_group_mean dummy: RMSE=153.43  R2=+0.029
TEST RMSE=402.30  R2=-4.60              per_group_mean dummy: RMSE=168.58  R2=+0.016
collapse sensor: pred_std=351 (225.2% of target_std=156), max|pred-y|=4.15e+03 (26.7x target_std)
```

For 1000 consecutive iterations the val Huber loss improved while the val R2 fell to -4.61. That is not a contradiction
-- it is what a bounded-influence metric does when the model buys bulk accuracy by paying in the tail. But it means ES
provided no protection at all on the metric the run reports, and the only model in the suite that catastrophically
failed is also the only one where ES never fired.

Compare the same run's `total_charge` (`es_best_iter: 29`) -- ES works when the surface is not robust.

**Fix**: keep the aligned surface (it fixes TRN's original bug) and add the missing half: after the fit, if ES
saturated the budget AND the model loses to a constant predictor on the ES rows, the robust surface demonstrably failed
to protect the reported metric -- refit on the RMSE family, which is exactly the remedy the low-side net already
applies.

---

## TRN-02 (P0) -- the refit net covers one side of the failure only

`_training_loop_refit.py:60` (`_maybe_refit_on_degenerate_best_iter`) triggers on:

```python
_threshold = min(_MIN_BEST_ITER_HEALTHY, _adaptive_floor)   # 3, or 5% of max_iter
if best_iter >= _threshold:
    return None
```

i.e. only `best_iter` too LOW, under a robust loss, refit with RMSE. The module docstring describes the class correctly
("a CatBoost / LGB / XGB fit converged at `best_iter < threshold` under a robust loss"), and the symmetric case --
`best_iter` pinned at the cap under a robust loss, with the honest metric collapsed -- has the same cause (the robust
surface is not tracking the reported objective) and the same remedy.

It is also already structured for reuse: the backend detection, `_RMSE_FALLBACK` table, `_NON_DEFAULT_LOSS_TOKENS`
gate, and the `set_params` -> fresh-instance refit ladder are all generic.

**Fix**: a sibling policy in the same module, sharing that machinery, gated on (saturated budget) AND (robust loss) AND
(R2 <= 0 on the ES rows).

---

## TRN-03 (P1) -- every CatBoost model is pessimised at construction

`_trainer_configure.py:533`:

```python
try:
    _cb_model._mlframe_polars_fastpath_broken = True
except Exception as e:
    ...
```

Unconditional, for every CatBoost instance the suite builds. The comment justifies it with "CB 1.2.x's
`_set_features_order_data_polars_categorical_column` has dispatch gaps on our nullable-Categorical / Enum schema".

That claim is testable, and the codebase already tests it. `_polars_native_support.accepts_polars("catboost")` fits and
predicts a CatBoost model in a child process on a frame that is precisely `{float, nullable float, pl.Enum}` -- the
exact schema the comment names. On this run it answered YES, and the suite printed so itself:

```
[predict] ... The installed CatBoost DOES accept a polars frame in a probe, so this is a per-model condition
```

while converting anyway, 52 times:

```
[predict fallback] polars->pandas(predict) 51_898x103 in 0.1s
[predict fallback] prepare_df_for_catboost(predict) in 0.0s
```

The mirror pre-sets at `train_eval.py:501` (reloaded models) and `_phase_train_one_target_schema.py:140` (clone
propagation) carry the same assumption forward.

The cost on this run is ~10-15s, which is small; the defect is that a measured question is answered by a constant, and
the measurement already exists three imports away.

**Fix**: drive the pre-set from `accepts_polars("catboost")`. Broken builds keep the optimisation; working builds stop
paying for it.

---

## TRN-04 (P1) -- the log line asserts a history that did not happen

`_cb_pool.py:463`:

```
"  [predict] this model is flagged as having missed the CatBoost polars fastpath earlier, so its "
"frames are converted to pandas from here on. ..."
```

"earlier" is true when the flag was set by the `except TypeError` handler at `_cb_pool.py:503`. It is false when the
flag came from the blanket pre-set in TRN-03 -- and on this run it did: the `CatBoost %s Polars fastpath rejected the
data` warning that accompanies a genuine miss appears **nowhere in the log**.

So the message describes a failure that never occurred, on a build the same message reports as working. Reading the log
top to bottom, there is no way to reach the truth.

**Fix**: distinguish the two provenances on the flag and say which one applies.

---

## TRN-05 (P1) -- loss knobs written for absent backends, none for the present one

The run is `All models ['cb']` (13:13:39). At 13:13:36 the mini-HPT emitted:

```
knob_overrides={'mlp_kwargs': {'model_params': {'loss_fn': 'huber'}}, 'lgb_kwargs': {'objective': 'huber'}, 'xgb_kwargs': {'objective': ...
```

and at 13:31:57 the per-target application reported:

```
[auto-loss] target='target_total_charge' skipped backends: lgb:no_entry_in_models_params, xgb:no_entry_in_models_params
```

No `cb_kwargs` was produced by the analyzer at all. The one component whose recommendation could have reached the model
that was actually trained produced nothing for it, and the three it did produce were discarded. The same
`skipped backends: lgb:..., xgb:...` line repeats for all 13 targets.

**Fix**: emit knob overrides for the backends present in the run, and report when an analyzer's entire output was
discarded because it targeted absent backends -- silently dropping 3 of 3 recommendations should not be a debug-level
event.

---

## TRN-06 (P2) -- two components recommend opposite losses on the same kurtosis

`residual_audit` (in `report_regression_model_perf`) printed 25 times:

```
hypothesis: Contaminated / outliers
suggested:  Huber (robust to outliers; tune delta around 1-2 sigma_resid)
why:        extreme tails / outlier contamination: excess kurt=+757.65 (> 10.0)
```

`_apply_loss_recommendation_in_place` on the same target, from the same statistic:

```
[auto-loss] target='target_total_charge' excess_kurt=2415.82 (n_finite=467084) -- excess_kurt=2415.82 > 20.0 --
Huber gradient collapses on extreme-kurt residual (delta*sign(r) approx 0 when most rows ...)
```

The loss recommender has a documented kurtosis ceiling above which Huber is the wrong answer; the residual audit's
advice table has no such ceiling and recommends Huber into it. A reader following the residual audit's suggestion would
re-create the failure the loss recommender exists to prevent.

The same report also gives three different distributional hypotheses for one target across adjacent splits
(`LogNormal` on the val dummy, `Gamma / Exponential` on the test dummy, `Contaminated / outliers` on the model), with
no indication that they describe the same `y`.

**Fix**: apply the loss recommender's kurtosis ceiling inside the residual-audit advice table so the two agree, and
name the competing hypothesis when adjacent splits disagree.

---

## TRN-07 (P2) -- a known, already-handled CatBoost error is swallowed at a second call site

`_phase_composite_wrapping.py:368` logs, six times:

```
[CompositeTargetEstimator] per-model y-scale emit failed for composite='...' (non-fatal): No matching signature found
```

`No matching signature found` is CatBoost's polars-dispatch `TypeError`, and `_cb_pool.py:495` already recognises that
exact string and recovers from it by converting to pandas and retrying:

```python
if not (_is_cb and _pl_df is not type(None) and isinstance(X, _pl_df) and "No matching signature found" in str(e)):
    raise
```

The composite y-scale emitter does not route through that recovery, so it gives up. The affected composites
(`total_charge-yqclip`, `-logY`, `-cbrtY`, `feedback_to_client_score-yqclip`, `-yjY`, `feedback_score-yjY`) lost their
per-model y-scale emit entirely -- which is the artefact used to compare a composite against raw on the honest scale,
i.e. precisely the evidence needed to judge the DSC findings.

**Fix**: route the emitter's predict through the same guarded path, so the recovery that exists is actually reached.
