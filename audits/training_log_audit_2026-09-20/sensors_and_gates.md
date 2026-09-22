# Sensors and gates

Source: production training log of `jobsdetails_shuffled` (576_646x127, 14 targets, CatBoost on GPU), 2026-09-20 12:56-14:05.

- **Disposition**: 12 findings, 12 RESOLVED / 0 PARTIAL / 0 TODO / 0 REJECTED / 0 NOT A DEFECT. Rows mirrored in [_TRACKER.md](_TRACKER.md).

The common shape across this group: a sensor computes the right numbers, then compares them against a threshold whose
units, scale or reference population do not match, and emits a verdict that reads as an all-clear.

---

## SEN-01 (P0) -- absolute spread tested against a relative threshold

`_target_temporal_audit_from_agg.py:114`:

```python
if mean_rates and max(mean_rates) - min(mean_rates) > drift_warn_threshold:
```

`drift_warn_threshold` defaults to `0.10` and its docstring reads "0.10 = a 10pp swing across segments" -- it is a
probability difference, correct for the `target_type="binary_classification"` default where `target_rate` lives in
`[0, 1]`. For a regression target `target_rate` is the segment MEAN in target units, so the comparison is
dimensionally wrong and the predicate reduces to "does this target have a scale larger than 0.1".

Both regression targets in the log tripped it, one of them while being stable:

| target | segment rates | absolute spread | relative spread | verdict |
|---|---|---|---|---|
| `target_total_charge` | 130.451 / 84.484 / 40.984 | 89.467 | 69% | WARN (correct) |
| `target_hours_to_hire` | 309.114 / 324.579 | 15.465 | **5.0%** | WARN (**false positive**, 5% < 10%) |

The false positive matters more than it looks: the same code path emits the `ACTIONABLE: Consider restricting training
to the most-recent stable segment` recommendation, so a stable target gets told to throw away 6 weeks of training rows.

**Fix**: keep the absolute (percentage-point) comparison for probability-valued rates and switch to a relative spread
for unbounded ones, reporting both numbers so the verdict is auditable.

---

## SEN-02 (P0) -- regression drift measured in train-sigma, which a heavy tail makes unreachable

`drift_report.py:283`:

```python
z = (delta / train_std) if (train_std and not np.isnan(train_std) and train_std > 0) else float("nan")
if not np.isnan(z) and abs(z) > regression_mean_z_threshold:
```

`target_total_charge` has `excess_kurt=2415.8` and `skew=32.10` (the suite's own `target_distribution_analyzer` says so
at 13:13:36). Its train `std=560.9` is set by the tail, not by the bulk -- `median=0`, `p99=1640`. So:

```
delta = 35.07 - 85.65 = -50.58
z     = -50.58 / 560.9 = -0.090 sigma      ->  far below any sane threshold
```

while the actual distribution moved like this:

| split | n | mean | std | median | p99 |
|---|---|---|---|---|---|
| train | 467_084 | 85.65 | 560.9 | 0 | 1640 |
| val | 51_898 | 57.57 | 349.2 | 0 | 1164 |
| test | 57_664 | 35.07 | 206.7 | 0 | 601 |

Mean -59%, std -63%, p99 -63%, and the report says `(no drift warnings - splits within threshold)`. The same module's
BINARY branch correctly flagged a 6.6pp move on `target_total_hired_above_1` in the same run, so the asymmetry is in the
regression statistic, not in the thresholds.

A mean-shift-in-sigma test is also the wrong significance scale: with n=57_664, a 0.09-sigma shift is ~22 standard
ERRORS of the mean. The effect size is small; the evidence that it is real is overwhelming. The report conflates them.

**Fix**: add relative-mean, dispersion (std ratio) and upper-quantile (p99 ratio) checks alongside the existing
sigma test, all reported with their numbers; keep the sigma test as the effect-size line.

---

## SEN-03 (P1) -- nothing detects a still-accruing target

`target_total_charge` is a cumulative amount; `target_hours_to_hire` is a duration. The test window runs to 2026-09-13
and the run happened on 2026-09-20, so the newest test rows had at most 7 days to accrue against a train-period mean of
`318.64` hours (13 days). Every number in SEN-02's table is the signature of right-censoring, and the temporal audit's
own segmentation shows it as a monotone decline (130.5 -> 84.5 -> 41.0).

The consequences are spread across the log and never connected:

- test R2 is negative for `total_charge` (-0.080) while val R2 is positive (+0.011);
- the temporal audit recommends training on the MOST RECENT segment, which is the most censored one;
- `honest_rmse_gate` and `yscale_gate` both score on train-period rows, so no composite is ever tested against the
  censored regime it will be reported on.

**Fix**: a censoring sensor that compares, per time bin, the observation window still available to each row against the
target's own realisation scale, and warns when the newest bins cannot have matured. It has the inputs already
(`ts_field` is auto-detected at 13:25:19; the target scale is in the distribution analyzer's output).

---

## SEN-04 (P1) -- the audit's stable window excludes the evaluated window

`_target_temporal_audit_from_agg.py:126` drops bins below `min_bin_fraction * median(n_obs)` BEFORE change-point
detection. On this run `threshold_n = 11_291` and 5 bins were dropped. The kept window is `2026-03-02..2026-08-10`
(24 bins, `n_obs=567_060`); the test split is `2026-07-29..2026-09-13`, ~8.8k rows/week, i.e. below the threshold for
its whole length.

So the verdict "target rate is/is not stable" describes a window that ends five weeks before the test period ends, and
the bins it discards are thin *because* of the censoring in SEN-03 -- the filter removes exactly the evidence the
sensor exists to find. The WARN text calls them "typically the partial first / last bins", which is true and also the
reason they matter.

**Fix**: report the dropped bins' time range and their overlap with the evaluation window, and say plainly when the
audit does not cover the split it is being read against.

---

## SEN-05 (P1) -- "unmeasurable" conflated with "measured and collapsed"

`_achievable_ceiling.py:360-392`:

```python
best_composite_rmse = float("inf")
for bcol in base_cands:
    ...
    if np.isfinite(comp) and comp < best_composite_rmse:
        best_composite_rmse = comp
...
if not np.isfinite(best_composite_rmse):
    return _verdict(headroom=nan, decision="proceed",
                    reason="optimistic composite ceiling unmeasurable (all candidate bases collapsed / absent)")
```

`best_composite_rmse` stays `inf` under two different conditions: `base_cands` was empty (nothing measured) or every
candidate produced a non-finite `comp` (everything measured and everything failed). The reason string names both and
distinguishes neither, and there is no count of how many candidates were tried. The log shows the ambiguous form twice:

```
target='target_feedback_score'            best_composite=inf headroom=nan base=None decision=proceed
target='target_feedback_to_client_score'  best_composite=inf headroom=nan base=None decision=proceed
```

Each then spent 114.71s and 125.02s in discovery, shipping specs that scored y-scale R2 +0.0005 and +0.0028.

Proceeding on a genuine absence of measurement is defensible. Proceeding on N measured, N collapsed is proceeding
against evidence.

**Fix**: count evaluated / collapsed candidates, carry both into the verdict and the log line, and let the existing
`strong_floor` gate act on the "measured and all collapsed" case.

---

## SEN-06 (P1) -- a tolerance gate reported as an endorsement

`_honest_rmse_gate.py:140,205`:

```python
tol = float(getattr(cfg, "honest_rmse_gate_tolerance", 1.05))
threshold = raw_rmse * tol
...
if not np.isfinite(rmse_y) or rmse_y > threshold:   # rejected
```

A spec is kept when its holdout RMSE is anywhere below `1.05 x raw`, which includes being up to 5% WORSE than raw. The
summary line then says:

```
[CompositeTargetDiscovery.honest_rmse_gate] all 10 spec(s) passed the honest-holdout y-scale RMSE gate
```

"Passed" here means "not more than 5% worse". Two of those ten specs (`total_charge-logY`, `total_charge-cbrtY`) went
on to score y-scale R2 of -0.024 and -0.026 -- worse than predicting the mean -- after 2.6 and 5.6 minutes of GPU.

The tolerance itself is defensible (it protects a spec whose value shows up at full model size), but the reporting is
not: the line gives no way to tell "10 beat raw" from "10 were within 5% of raw, 6 of them worse".

**Fix**: split the count into better-than-raw and within-tolerance-but-worse, and log both.

---

## SEN-07 (P2) -- sensors fire, the model ships, nothing aggregates

`target_hours_to_hire` triggered four separate warnings:

```
clip_predictions_to_train_envelope  VAL:  804 rows below -495.9, 53 above 4152
run_collapse_sensor:group-ood-shift VAL:  pred_std=351 (225.2% of target_std=156), R2=-4.61
clip_predictions_to_train_envelope  TEST: 1075 rows below -495.9, 79 above 4152
run_collapse_sensor:group-ood-shift TEST: pred_std=386 (227.4% of target_std=170), R2=-4.6
```

and was then saved (`1.46 Mb`) and reported like any other model. Each sensor logs in isolation; none of them is
consulted when deciding whether the artefact is fit to persist, and the suite-end verdict that would have aggregated
them never ran (the process was killed at 14:05).

**Fix**: record every sensor trip on the model's metadata so persistence and the suite verdict can see them, and stamp
the saved artefact with the trips rather than leaving the evidence only in the log stream.

---

## SEN-08 (P2) -- unsmoothed group means emit p=0 / p=1

`_dummy_baseline_compute.py:207`:

```python
stats_df = train_pair.group_by(cat_col).agg(pl.col("__y__").mean().alias("__mean__"), pl.len().alias("__size__"))
...
val_pred = val_joined.get_column("__mean__").fill_null(global_mean).to_numpy()
```

The per-group prediction is the raw group mean. For a binary target a group with one training row is predicted at
exactly 0.0 or 1.0, and `pg_diag["repeat_entity_rate"]` (fraction of val rows whose train group has >= 5 members) was
`1.00` on this run -- which measures the BULK, not the tail, so singleton groups still exist underneath it.

Any unbounded proper scoring rule then explodes on a single wrong-and-certain row. `exploss` is
`y*sqrt((1-a)/a) + (1-y)*sqrt(a/(1-a))` clipped at `eps=1e-12`, so one such row contributes up to `1e6`. The log shows
the same constant-prior dummy scoring:

```
VAL  (DUMMY) ... exploss=0.88
TEST (DUMMY) ... exploss=18.29
```

a 20x move for a baseline whose only input is the training prior. `LOG_LOSS` moved 0.57 -> 0.64 over the same rows,
which is the honest magnitude of the shift; `exploss` is reporting its clip constant.

The `__size__` column needed for the correction is already computed in the same aggregation.

**Fix**: shrink the group mean toward the global mean by the group size (Beta posterior mean for binary,
James-Stein-style for regression). This removes the 0/1 endpoints AND makes the baseline stronger, which is the point
of a baseline.

---

## SEN-09 (P2) -- calibration reported, never fitted

`record_split_membership` wrote `{'train': 467084, 'val': 51898, 'test': 57664, 'calib': 0}`. No calibration rows were
allocated, yet calibration is measured and degrades exactly as the prior-shift warning predicted:

```
WARN: TEST P(y=1)=0.360 vs train 0.293 (Δ=+6.6pp); ... model will be miscalibrated on test
VAL  CALIBRATIONs: 1: MAEW=1.48%, STD=0.91%
TEST CALIBRATIONs: 1: MAEW=3.74%, STD=2.57%
```

The suite detected the cause, measured the effect, and did nothing in between.

**Fix**: when a prior shift is detected and no calibration split exists, say so at the point of the shift warning and
name the knob that would allocate one, rather than only reporting the downstream damage.


---

## SEN-10 (P2) -- threshold optimisation is on by default and silently does nothing without a calib slice

Found while writing up SEN-09, which is the same gap seen from the other end.

`_model_configs_behavior.py:135` sets `auto_optimize_threshold: bool = True`. The function it gates,
`_optimize_decision_threshold_on_calib_slice`, described itself as:

```
Gated by ``TrainingBehaviorConfig.auto_optimize_threshold``; default OFF (bit-identical no-op).
```

which is the opposite of the actual default -- and the function immediately below it in the same file carries a
docstring that was already corrected for this exact error ("this docstring and the suite kwarg used to describe it
as disabled unless requested, which was the opposite of what every run actually did").

Being on by default is not the same as running. The optimizer fits only on a disjoint calib slice:

```python
_cp = getattr(_e, "calib_probs", None)
_ct = getattr(_e, "calib_target", None)
if _cp is None or _ct is None:
    continue
...
if out:
    ctx.metadata["decision_threshold"] = out
```

With `calib_size` unset no entry carries `calib_probs`, every iteration `continue`s, `out` stays empty, and the
function returns having logged nothing at all. The production run was exactly there:

```
record_split_membership ... {'train': 467084, 'val': 51898, 'test': 57664, 'calib': 0}
TEST ... target_total_hired_above_1 ... ROC AUCs: 1=0.81   PR AUCs: 1=0.74
           1     0.78    0.48    0.59    20742      <- recall 0.48 at the default 0.5 cut
WARN: TEST P(y=1)=0.360 vs train 0.293 (Δ=+6.6pp); ... model will be miscalibrated on test
```

The one model in the suite with real separation kept the default cut, under a prior shift the same run had already
flagged, while a step meant to move that cut was enabled, reached, and unable to act -- with nothing connecting any
of it in the log.

**Fix**: report the enabled-but-unable state once, naming the knob that would resolve it, and correct the docstring.
The warning counts binary entries and fires only when NONE of them could run, so a partially-calibrated suite and a
regression-only suite both stay quiet.


---

## SEN-11 / SEN-12 -- the class behind SEN-10, swept

SEN-10 was the first instance found, not the only one. The shape is: a step that is ON by default, has a
precondition, reports only when it produced something, and so is completely silent in the "enabled, reached,
unable to act" state.

An AST sweep of `src/mlframe` for that shape (a config-gated function that `continue`s past entries lacking a
precondition and logs only inside an `if <result>:` block with no `else`) returned 11 candidates. The detector is
deliberately broad, so each was judged by one criterion: **would the step have delivered value had its
precondition held?** If yes, its silence lost the user something and is a defect. If the thing it guards against
cannot exist without the precondition, silence is correct.

| Candidate | Verdict | Reason |
|---|---|---|
| `_phase_finalize_calibration._optimize_decision_threshold_on_calib_slice` | **RESOLVED** (SEN-10) | ON by default, needs calib; the threshold was genuinely lost |
| `_phase_finalize_calibration._conformal_on_calib_slice` | **RESOLVED** (SEN-11) | `enabled=True`, `classification_mode="sets_lac"`; classification sets have no fallback. Regression intervals fall back to OOF and are reported only when both are absent |
| `_phase_finalize_calibration._isotonic_overfit_risk_check` | **RESOLVED** (SEN-11) | ON by default (`check_isotonic_overfit_risk=True`) though its docstring said OFF |
| `_phase_finalize_calibration._apply_confidence_shrinkage_to_regression` | **RESOLVED** (SEN-12) | ON by default, needs OOF, which `oof_n_splits=0` never produces |
| `composite/discovery/_honest_rmse_gate.apply_honest_rmse_gate` | already fixed | SEN-06 added an unconditional summary line after the `if rejected:` block |
| `_phase_finalize_calibration._recalibrate_regression_on_calib_slice` | **NOT A DEFECT** | opt-in, `point="off"` by default; its silence is what the caller asked for |
| `composite/discovery/_yscale_holdout_gate.apply_structural_fragility_gate` | **NOT A DEFECT** | it guards against level offsets on UNSEEN GROUPS; with no group ids that risk does not exist, so nothing is lost |
| `feature_selection/filters/_cat_interactions_step.run_cat_interaction_step` | **NOT A DEFECT** | an `if verbose:` progress log -- explicit quiet mode |
| `feature_selection/filters/_cat_mm_correction._maybe_rerank_with_mm` | **NOT A DEFECT** | an `if verbose:` progress log |
| `training/core/_phase_recurrent.train_recurrent_models` | **NOT A DEFECT** | an `if verbose:` progress log |
| `feature_selection/wrappers/rfecv` `__init__` and `fit` | **NOT A DEFECT** | config-conditional info lines, not a precondition-gated step |

Four real defects out of eleven candidates. The detector over-fires on `if verbose:` blocks, which are a
different and legitimate pattern.

**Design choices.** The calib case reports as ONE aggregated WARNING rather than one per step: the log's own
problem was repetition (WST-05) as much as silence, and one knob restores every step, so one line should say so.
The OOF case reports at INFO in a separate line, because it has a different remedy (`oof_n_splits`) and because
`_model_configs_behavior` records a deliberate decision that OOF-dependent steps no-op "silently (not a WARN)" as a
caller cost choice. That decision is kept -- `oof_n_splits` is not flipped, since a K-fold refit of every model is a
real cost -- and only the invisibility is removed. The weakness in the original reasoning is that at defaults the
caller made no choice: the defaults switch these steps ON and simultaneously make them impossible.
