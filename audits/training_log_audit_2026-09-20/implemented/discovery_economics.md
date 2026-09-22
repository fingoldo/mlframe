# Composite discovery: cost against delivered value

Source: production training log of `jobsdetails_shuffled`, 2026-09-20 12:56-14:05.

- **Disposition**: 6 findings, 6 RESOLVED / 0 PARTIAL / 0 TODO / 0 REJECTED / 0 NOT A DEFECT. Rows mirrored in [_TRACKER.md](../_TRACKER.md).

## What the run spent

Suite entry 13:12:52, first real model fit 13:26:29, process killed 14:05:16 -- about 52 minutes of suite time
observed.

| Stage | Wall |
|---|---|
| discovery, 4 targets (`121.13s + 114.71s + 125.02s + ~120s`) | ~8.0 min |
| composite model fits (8 completed, 1 killed mid-fit) | ~19.4 min |
| raw-target model fits (5) | 13.3 min |

So roughly 27 of 52 minutes went to composite machinery. What it delivered, on the honest y scale:

| composite | fit wall | y-scale val R2 |
|---|---|---|
| `total_charge-yqclip` | 2.3 min | +0.026 |
| `total_charge-cbrtY` | 5.6 min | **-0.026** |
| `total_charge-logY` | 2.6 min | **-0.024** |
| `feedback_to_client_score-chain_linear_residual_sp-title_upper_ratio` | 1.6 min | +0.0033 |
| `feedback_to_client_score-yjY` | 1.3 min | +0.00051 |
| `feedback_to_client_score-yqclip` | 1.5 min | +0.0028 |
| `feedback_score-yjY` | 1.3 min | +0.0032 |
| `feedback_score-chain_linear_residual_sp-row_summary_q50` | 3.2 min | +0.0084 |
| `feedback_score-yqclip` | killed | -- |

Two of the nine are worse than predicting the mean. The rest are within a rounding error of it. For reference, the raw
targets they were derived from scored +0.011, +0.0051 and +0.0016 -- the whole family is in the noise, which is a
property of the data, not of discovery. The findings below are about discovery not being able to TELL.

---

## DSC-01 (P1) -- the ship/no-ship floor has no relation to the noise it filters

`_composite_target_discovery_config_base.py:523`:

```python
min_honest_gain_to_train: Optional[float] = 0.001
```

applied at `_phase_composite_discovery.py:917` against a RELATIVE gain (fraction of baseline RMSE saved OOS). The
comment records that this floor already exists because of an earlier bad run ("a production run trained 12 specs with
gain <= 0"). It is an absolute constant, chosen once, with nothing tying it to how precisely the gain is measured.

The nine specs shipped on this run had gains of `+0.011, +0.009, +0.005, +0.005, +0.004, +0.004, +0.003, +0.003,
+0.002`. These are measured with a 60-tree / 15-leaf LightGBM on a capped 20k-row holdout and then used to decide
whether a 1000-iteration CatBoost fit on 467k rows is worth 2-6 minutes of GPU. The gap between measurement and
decision is several orders of magnitude of model capacity.

**Fix**: make the floor noise-aware. The gate already has everything needed for a paired test (see DSC-02): require
the gain to exceed both the constant floor and `z x` its own standard error.

---

## DSC-02 (P1) -- the raw baseline's predictions are discarded, so no paired test is possible

`_honest_rmse_gate.py:132`:

```python
raw_rmse = rmse(y_eval, _fit_predict(y_fit))
```

`_fit_predict` returns a full prediction vector on the shared `eval_idx` rows; only the scalar RMSE survives the
statement. Every spec is later scored on those same rows with the same model class, so the per-row squared errors of
spec and raw are PAIRED -- the single cheapest and most informative statistic available, and it is thrown away on the
line that computes it.

With both vectors retained, `se(RMSE_raw - RMSE_spec)` follows from the per-row squared-error differences:

```
d_i    = (y_i - yhat_raw_i)^2 - (y_i - yhat_spec_i)^2
se_d   = std(d) / sqrt(n)
se(dRMSE) ~ se_d / (RMSE_raw + RMSE_spec)
```

O(n), no extra fits, and it turns DSC-01's constant into a measurement.

**Fix**: keep the raw prediction vector, stamp `honest_holdout_rmse_gain_se` on each spec.

---

## DSC-03 (P1) -- specs selected on differences smaller than the run's own non-determinism

At 13:25:19 the suite emits, four lines apart:

```
[CompositeTargetDiscovery] added composite target 'target_feedback_score-yqclip' ... (honest gain +0.002).
[CompositeTargetDiscovery] 9 composite target(s) added to target_by_type.
WARNING ... composite mode + GPU training detected (catboost) AND 9 composite spec(s) shipped. GPU non-determinism is
amplified by the K=9 extra fits; ensemble weights may drift across runs even with random_seed...
```

The warning is correct and the selection immediately above it is made on differences of 0.2%-1.1%. The suite knows the
noise floor exists and does not use that knowledge when choosing what to ship.

**Fix**: DSC-01 + DSC-02 give the selection a noise scale; this finding is closed by them, and the warning should
reference the measured SE once it exists.

---

## DSC-04 (P2) -- zero-inflated targets routed to multiplicative y-compressors

`target_total_charge` is reported by the suite's own drift report as `median=0, p01=0, mean=85.65` -- a point mass at
zero covering more than half the rows. The same run trains `target_total_hired_above_1` with `P(y=1)=0.2934`, i.e. 71%
of jobs never hire, which is the same fact stated as a classification target.

Discovery selected `logY` and `cbrtY` for it. Both compress a right tail and both have curved inverses; on a
distribution that is mostly an exact zero the inverse cannot reproduce the spread, and the suite measured exactly that
after paying for the fits:

```
[regression-collapse-sensor:std-collapse] ... target_total_charge-logY  pred_std=6.44 (1.8% of target_std=349)
[regression-collapse-sensor:std-collapse] ... target_total_charge-cbrtY pred_std=4.49 (1.3% of target_std=349)
```

2.6 + 5.6 = 8.2 minutes of GPU to discover a property visible in the target's own summary statistics before discovery
started. The transform zoo has the information: `fit-line:553` already skips right-tail compressors for LEFT-skewed
targets, so the precedent for a pre-screen on distributional shape exists.

**Fix**: measure the zero (or any single-value) mass once per target and skip curved y-compressors above a threshold,
by the same mechanism the left-skew skip already uses.

**Follow-up capability**: skipping the doomed transforms stops the waste but leaves the target modelled by a single
regressor. The fuller answer is a hurdle decomposition, and the same run already trained its first half
(`target_total_hired_above_1`, test ROC AUC 0.81) as a separate target. `HurdleRegressor`
(`training/composite/hurdle.py`) now provides it: on a synthetic reproducing this target it keeps a 12.1% prediction
spread where the single `log1p` model collapses to 1.8% -- the production figure -- and has the best R2 of the three
variants at every tail weight measured (`test_biz_val_hurdle_regressor.py`). It is a new estimator, not a defect fix,
so it carries no finding ID.

---

## DSC-05 (P2) -- discovery runs before the diagnostic that would gate it

Ordering in the log:

```
13:14:29  [CompositeTargetDiscovery] auto-enabled ...
13:25:19  9 composite target(s) added to target_by_type
13:25:32  [BaselineDiagnostics] target='target_total_hired_above_1' ... composite_recommendation=unlikely_to_help
13:58:42  [BaselineDiagnostics] target='...-row_summary_q50' ... composite_recommendation=unlikely_to_help
```

`BaselineDiagnostics` computes a `composite_recommendation` whose entire purpose is to say whether composite discovery
is worth running, and it is computed after discovery has finished and its specs have been committed. Discovery does
consume a `hint` from a cached BaselineDiagnostics precompute (`run_composite_target_discovery-line:500`), so the
component is reachable earlier -- the verdict field simply is not consulted.

Every `composite_recommendation` in this run was `unlikely_to_help`, with `reason: top ablation delta%=0.25 < 2.00
(no dominant features)`.

**Fix**: read the recommendation from the same precompute the hint comes from, and let it gate (or at minimum
downgrade) discovery instead of arriving after the fact.

---

## DSC-06 (P2) -- a leak warning for a leak that cannot occur

`_opt_in_steps.py:150`:

```python
synth, records = discover_interaction_bases(
    candidates, y_screen, top_k=top_k, max_pairs=max_pairs,
    nbins=int(self.config.mi_nbins),
)
```

no `train_mask=`, so `generate_interaction_bases` warns four times:

```
WARNING - generate_interaction_bases: no train_mask given, so the div eps floor is the median over all 100000 rows
supplied. Pass train_mask=<train rows> to keep test-row scale out of the synthetic columns.
```

But in this call path every supplied row is already a train row: `candidates` comes from `self._auto_base_pool`, which
the docstring states is "already restricted to `train_idx`", and the row selection is
`screen_idx = train_idx[sample_idx]`. There is no test row in the 100_000, so the eps floor is train-derived and the
warning is a false positive.

It is still a defect: the invariant is implicit, an unrelated refactor could break it silently, and a WARNING that
cries leak four times per run trains the reader to ignore the one that matters. `TRF-24` in
`audits/composite_audit_2026-09-19/transforms.md` added this warning precisely so a real missing mask would surface --
it is working, and this call site is making it lie.

**Fix**: pass an explicit all-train mask, making the invariant checkable rather than assumed.
