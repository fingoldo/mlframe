# Recomputation, contradictory decisions, and log lines that mislead

Source: production training log of `jobsdetails_shuffled`, 2026-09-20 12:56-14:05.

- **Disposition**: 8 findings, 5 RESOLVED / 2 PARTIAL / 0 TODO / 1 REJECTED / 0 NOT A DEFECT. Rows mirrored in [_TRACKER.md](_TRACKER.md).

---

## WST-01 (P1) -- target-independent drift work repeated once per target

`_phase_train_one_target_model_setup.py:113` calls, inside the per-target setup:

```python
render_target_drift_diagnostics(
    train_frame=train_df, test_frame=test_df,
    timestamps=_ts_test, task=task, ...
)
```

Its two expensive builders take only feature frames:

- `psi_heatmap(test_frame, ts, feature_names=...)` -- quantile PSI per column;
- `adversarial_validation(train_frame, test_frame, ...)` -- a 3-fold LightGBM train-vs-test classifier.

**Correction after verification**: only the FIRST of those actually repeats. `_ADVERSARIAL_CACHE` /
`_adversarial_cache_key` already memoise the adversarial panel on master, and the log confirms the cache works -- it
holds exactly four `_drift_adversarial` dataset builds, all between 13:25:59 and 13:26:12 for the first target, not
four per target. The first-pass reading of this finding claimed 14 adversarial fits and was wrong. What does repeat is
`compute_psi_matrix`, 14 times, on frames the suite itself proves are identical:

```
PipelineCache HIT key=imp0_scale0_enc0_tier(False, False)_kindpl_feats17f80e691c284c0b_dtc48f294e61a2 (hits=13 misses=1 size=1)
```

one miss, thirteen hits -- one feature matrix for the whole run.

**Fix**: give PSI the same content-keyed cache the adversarial panel already has (`_psi_cache_key` + `_PSI_CACHE`),
re-rendering only the chart per target.

---

## WST-02 (P2) -- one component keeps columns as signal, another drops them as constant

13:13:39, `_maybe_auto_drop_after_feature_analyzer`:

```
[mini-HPT] keeping 9 NaN-heavy column(s) that would otherwise be dropped: every model in this run (cb) consumes NaN
natively, so the missingness is signal rather than something to impute away. Kept: deliverables_upper_ratio,
deliverables_digit_ratio, deliverables_punct_ratio, local_desc_len, local_desc_words, local_...
```

13:25:21, `_maybe_run_unsupervised_pre_screen`:

```
[pre-screen] dropped 9 column(s) suite-wide (variance=0.0, null_fraction>0.99): ['deliverables_digit_ratio',
'deliverables_punct_ratio', 'deliverables_upper_ratio', 'local_desc_digit_ratio', 'local_desc_len',
'local_desc_lines', 'local_desc_punct_ratio', 'local_desc_upper_ratio', 'local_desc_words']
```

The same nine columns, kept as signal and then dropped as constant twelve minutes apart. The keep decision is defensible
in isolation (CatBoost does consume NaN natively) but it is made without the variance check the later pre-screen
applies -- a column that is 99%+ null AND has zero variance in its observed cells carries no missingness signal either,
because the missingness pattern is the only thing in it and that is already what "99% null" says.

In between, composite discovery screened them four times, once per target:

```
[CompositeTargetDiscovery] auto-base: dropping 12 feature(s) with <10% finite cells in screening sample: [...]
```

**Fix**: give the earlier keep-decision the same variance criterion the pre-screen uses, so a zero-variance
near-all-null column is dropped at the first opportunity rather than the last.

---

## WST-03 (P2, REJECTED) -- redundancy found once, applied never, re-derived per target

13:13:38, `analyze_feature_distribution` reported 7 pairs at `|corr| >= 0.95` and a `drop_candidates` list. 13:13:39,
`_maybe_auto_drop_after_feature_analyzer` applied exactly one drop:

```
[mini-HPT] auto-drop applied: 1 of 107 column(s) removed. Breakdown by rule: low_variance: 1
[mini-HPT]   low_variance -> hide_budget
```

The seven redundant pairs stayed. `_auto_base` then rediscovered and re-deduplicated them independently for each of
the four discovery targets:

```
13:14:45  auto-base dedup dropped 8 candidate(s) at |corr|>=0.950: ... hourly_budget_max~=hourly_budget_mid(|corr|=0.985) ...
13:17:23  auto-base dedup dropped 7 candidate(s) at |corr|>=0.950: hourly_budget_mid~=hourly_budget_max(|corr|=0.975) ...
13:19:52  auto-base dedup dropped 3 candidate(s) at |corr|>=0.950: hourly_budget_max~=hourly_budget_mid(|corr|=0.979) ...
13:22:33  auto-base dedup dropped 6 candidate(s) at |corr|>=0.950: row_summary_std~=row_summary_mean(|corr|=0.993) ...
```

Note the same pair reports `|corr|` of 0.985 / 0.975 / 0.979 across targets -- each recomputed on that target's own
screening sample. The suite-level analyzer had the full-data answer at 13:13:38 and nothing consumed it.

**Disposition: REJECTED.** The two computations are not the same computation. The suite-level analyzer answers WHICH
pairs are correlated; `_auto_base` additionally has to choose which member of a pair survives for THIS target, and that
choice comes from the target's own MI ranking -- it is target-dependent by construction and cannot be hoisted. The
only genuinely duplicated work is the per-target correlation recompute, about 0.2s per target, four times on this run.
Feeding the suite-level list in would change which feature survives per target: a selection change with no measured
benefit, which is not a trade worth making for 0.8s. The differing `|corr|` values (0.975 / 0.979 / 0.985) are four
subsample estimates of one population quantity, not four contradictory answers.

---

## WST-04 (P2) -- two train/val boundaries printed for one split

```
13:13:34,125  Temporal layout (val_placement='forward', default): train_max=2026-07-10 17:16:58.812000,
              val=[2026-07-10 17:17:55.564000..2026-07-29 23:39:58.704000], ...
13:13:34,216  467084 train rows 2026-03-01/2026-07-21, 51898 val rows 2026-07-21/2026-07-29 +25949Rnd, ...
```

Ninety-one milliseconds apart, for the same split: `train_max` is 07-10 in one line and 07-21 in the next. The
reconcilable explanation is that the first line describes the purely temporal layout before the random augmentation and
the second describes the realised split -- but neither line says so, and every downstream report ("trained on 467.1K
rows 2026-03-01/2026-07-21") uses the second.

**Fix**: label the pre-augmentation layout as such, or emit it only when no augmentation follows.

---

## WST-05 (P2) -- a split choice silently disables a whole baseline family

`_dummy_baseline_regression.py:244` logs, once per regression target, 13 times:

```
[dummy-baselines] target='...' timestamps present but split is interleaved (monotonic check failed) -- TS baselines skipped
```

The cause is one suite-level decision -- the val split carries `+25949Rnd` in-period rows -- and the consequence is
suite-level: no regression target in the run was ever compared against a time-series-naive baseline, on data that is
explicitly temporal (`val_placement='forward'`, `train->prod_estimated_gap=64.95d`).

Reported per target as an incidental skip, it reads as a minor detail thirteen times instead of as one structural gap.

**Fix**: detect the condition once at suite level and report it once, naming the split knob responsible and the set of
baselines lost.

---

## WST-06 (P2) -- an ordinal metric reported for a binary target

Every classification block in the run prints:

```
CLASSIFICATION METRICS:
	quadratic_weighted_kappa=0.37
	weighted_kappa=0.37
```

on `target_total_hired_above_1`, which is binary. With two classes the quadratic weight matrix has a single off-diagonal
level, so quadratic weighting degenerates to linear weighting and both reduce to Cohen's kappa -- the two lines are the
same number twice, and neither carries the ordinal-agreement meaning the metric name implies. The dummy rows in the
same run print `-0.00` and `0.00`, which is the tell.

**Fix**: report the weighted kappas only for ordinal targets with three or more classes; for binary, print plain
Cohen's kappa once, under its own name.

---

## WST-07 (P3, PARTIAL) -- monitor ETA, and polls after the fit returned

```
13:27:29  iter=140/1000  it/s=2.33 (early 2.33) elapsed=60s  cb-ETA=4.8m
13:28:29  iter=401/1000  it/s=4.35 (early 4.35) elapsed=2.0m cb-ETA=2.6m
13:30:14  model.fit(CatBoostClassifier) done -- 225.6s   (3.8m)
```

**Correction after verification**: the printed `cb-ETA` is NOT an mlframe extrapolation. `poll_once` reads it from
CatBoost's own progress file via `read_time_left_tail`, so it is CatBoost's estimate and mlframe only relays it; the
first-sample rate (`early`) feeds the collapse check alone. The ETA half of this finding is **NOT A DEFECT**. (The
variance is real -- it/s ranges 2.33-9.35 within one fit, with `SearchUI.exe`, `Telegram.exe`, another `python.exe`
and `LogonUI.exe` resident on the same 8GB card at `mem=8032/8192MB` -- but that is GPU contention on the host, not a
reporting bug.)

What IS a defect: polls continue after the fit has returned.

```
13:33:09,545  model.fit(CatBoostRegressor) done -- 70.4s
13:33:09,569  [cb-gpu-monitor] CatBoostRegressor: iter=279/1000 it/s=1.92 elapsed=70s cb-ETA=2.1m
```

a stale poll 24ms after completion, reporting an ETA for a finished fit.

**Fix**: re-check the stop flag after the interval wait, so a `stop()` that lands between the wait returning and the
poll starting suppresses that poll.

---

## WST-08 (P3) -- an empty report section is printed unconditionally

Every classification block ends:

```
ICEs:
	1=-0.29
RICEs:

```

`RICEs:` with no rows. The section header is emitted before its contents are known to exist.

**Fix**: emit the header only when there is at least one value under it, as the sibling sections already do when
their inputs are absent.
