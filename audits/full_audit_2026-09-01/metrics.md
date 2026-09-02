# metrics

Files reviewed: 22 read in full or in the sections that matter (of 178 .py in the cluster) | LOC: ~7,800 read of 33,868 in cluster

## Summary
The moment-formula bug class the brief flagged does NOT appear anywhere in this cluster -- a repo-wide grep for
raw-power-sum + binomial-expansion skew/kurtosis (`s3/n - 3*mean*(s2/n) + 2*mean**3`) and for additive
denominator epsilons (`std**3 + 1e-12`, `var*var + 1e-12`) over `metrics/`, `calibration/` and `evaluation/`
returns nothing; the only `**1.5` in the cluster is the BCa acceleration denominator, which is correct.
Dispersion estimators consistently use `ddof=1`, `_drift`'s PSI/KL/JS/W1/KS kernels are correct, and the
bootstrap/jackknife machinery in `evaluation/` is unusually careful (documented bit-identity contracts,
closed-form jackknives, RNG-order preservation). The findings that DO exist are concentrated in two places:
(1) one outright silent-wrong-number bug -- `optimal_threshold_bootstrap_ci` sweeps bootstrap resamples in
RANDOM score order because its index matrix is never sorted, so the "95% CI" it reports (and which
`training/_honest_decision_threshold.py` prints in a production log line) is off by ~800x in interval width on
a verified repro; and (2) a cluster of statistical-honesty defects in the smaller evaluation/calibration
diagnostics -- a normal-z quantile where a Student-t is required, a paired-difference band computed from one
arm's spread only, a self-verifying leakage check that can never fail, and two in-sample-selected thresholds
reported as production estimates. Separately, `pick_best_calibrator`'s DEFAULT path computes a full
1000-resample bootstrap CI plus an O(n*2000) gather jackknife per candidate and then throws both away, which
also makes it raise `MemoryError` on any OOF above ~268k rows for work that is never used.

## Findings

### METRICS-1 [P0] silent-wrong-number
**File:** src/mlframe/metrics/classification/_threshold_optimization.py:212 (index draw), :161-176 (kernel sweep)
**Summary:** `optimal_threshold_bootstrap_ci` feeds `_bootstrap_threshold_kernel` an UNSORTED random index
matrix, but the kernel's incremental confusion-count sweep is only valid on descending-score order, so the
returned threshold CI is a percentile interval over essentially random prefixes rather than over per-resample
optimal thresholds.
**Failure scenario:** verified live on this worktree. `n=4000`, `y ~ Bernoulli(0.3)`,
`s = 0.6*y + 0.4*U(0,1)` (perfectly separable: positives in [0.6,1.0], negatives in [0,0.4]).
`optimal_threshold(y, s, metric="f1")` -> `(0.60056, F1=1.0)`. A reference bootstrap that calls
`optimal_threshold` on 50 actual resamples gives a CI of `[0.60056, 0.60105]` (min 0.60056, max 0.60150).
`optimal_threshold_bootstrap_ci(y, s, metric="f1", n_boot=50)` returns `[0.60903, 0.99071]` -- an interval
~800x too wide whose upper end is a threshold at which the classifier predicts almost nothing positive, on a
perfectly separable problem. This value is consumed by `mlframe/training/_honest_decision_threshold.py:105`
and printed as `95% CI [lo, hi]` in `format_decision_threshold_line`, i.e. the log line that is supposed to
tell an operator whether the tuned threshold is trustworthy says "the interval spans most of the score range"
(the module's own stated signal for "tuning bought nothing") on data where tuning is in fact perfect.
`grep -rn optimal_threshold_bootstrap_ci` finds NO test anywhere in the repo.
**Suggested fix:** sort each row of the index matrix ascending before the kernel:
`idx = np.sort(rng.integers(0, n, size=(n_boot, n)), axis=1)` at :212. Because `y_sorted`/`s_sorted` are in
DESCENDING score order, ascending positions reproduce descending scores, the resample multiset is unchanged,
and the RNG draw order is untouched, so reproducibility from `random_state` is preserved. Also fix the
kernel docstring at :146-148, which currently asserts the false invariant ("resampling positions keeps every
resample in sorted order for free"). Ship a regression test comparing against a loop of `optimal_threshold`
calls on explicit resamples.
**Evidence:** read the kernel and the caller; ran both the shipped function and a reference bootstrap in the
worktree and compared the intervals (numbers above). `np.random.default_rng(...).integers(0, n, size=(B, n))`
is unsorted by construction; the kernel's `while i < n and s_sorted[idx[b, i]] == cur` tie-run only groups
CONSECUTIVE equal entries, and `tn = N - fp` / `fn = P - tp` are only the true confusion counts if the prefix
`idx[b, :i]` is exactly "all rows with score >= cur".

**Disposition:** RESOLVED. `idx.sort(axis=1)` after the draw restores the kernel's descending-score walk; the false docstring invariant is corrected. Measured 8.7x too wide before, 1.01x after. `tests/metrics/test_optimal_threshold_bootstrap_ci.py`.

### METRICS-2 [P1] wasted-work-in-default-path
**File:** src/mlframe/calibration/policy.py:794, :811, :816-825
**Summary:** In the DEFAULT `selection="inner_cv"` path, `pick_best_calibrator` builds a
`(n_bootstrap, n_oof)` resample matrix and runs a full 1000-resample bootstrap ECE CI plus a BCa jackknife per
candidate, then unconditionally overwrites both results with the inner-CV numbers -- and the matrix build can
hard-fail with `MemoryError` for a value that is never read.
**Failure scenario:** `pick_best_calibrator(None, None, oof_probs, oof_y)` with `n_oof = 300_000` and all
defaults. Line 794 computes `projected_bytes = 4 * 1000 * 300000 = 1.2 GiB`, which exceeds
`DEFAULT_RESAMPLE_MATRIX_MAX_BYTES = 1 << 30`, so `_build_resample_indices` raises `MemoryError` and the whole
call dies -- even though every consumer of that matrix (`ci` at :811) is discarded at :822 and :825. Below the
ceiling it is "merely" pure waste: per candidate (4 by default) 1000 `_ece_score_idx_numba_serial` calls plus
`_jackknife_metric` at :496, which is 2000 leave-one-out iterations each doing two `np.concatenate` copies of
an ~n-length array -- ~2.4e9 element touches per candidate at n=200k, all thrown away.
**Suggested fix:** move the `idx_matrix = _build_resample_indices(...)` build and the
`_bootstrap_ece_with_indices` call inside an `if inner_folds is None:` branch (or compute the held-out
inner-CV number first and skip the bootstrap when it succeeds). The `cal_oof` fit at :801-806 must stay -- it
is the only producer of `results[name]["calibrated_probs"]`, which the reliability plot reads at :634.
**Evidence:** read :781-833. `ci` is assigned only at :811 and read only at :816-817; the `if inner_folds is
not None:` block at :818-825 reassigns both `rank_ece` and `ece_ci`, and `inner_folds` is non-None whenever
`selection == "inner_cv"` (:797-798), which is the documented default (:676, :683-685). Constants confirmed at
:41 (`DEFAULT_N_BOOTSTRAP = 1000`) and :52 (`1 << 30`).

**Disposition:** RESOLVED. The resample matrix and the bootstrap now run only on the `same_oof` path that reads them; the `inner_cv` default no longer builds a 1.2 GiB matrix for a discarded value. `tests/calibration/test_policy_skips_the_bootstrap_it_discards.py`.

### METRICS-3 [P1] optimised-primitive-not-wired
**File:** src/mlframe/calibration/policy.py:496
**Summary:** `_bootstrap_ece_with_indices` computes the BCa acceleration term with the generic
O(max_n * n) gather `_jackknife_metric`, even though the O(n) closed-form `_jackknife_ece` for exactly this
metric already ships in `evaluation/_bootstrap_jackknife.py` and is already wired into the two other ECE
bootstrap call sites.
**Failure scenario:** `pick_best_calibrator(..., selection="same_oof")` (or any future caller of
`_bootstrap_ece_with_indices`) at `n_oof = 1.6M`: `_jackknife_metric` performs 2000 LOO iterations, each
allocating two ~1.6M-element arrays and re-running `_ece_score` over them. The already-shipped
`_jackknife_ece` is documented in CLAUDE.md as ~800-1345x faster on this exact shape and verified
bit-identical to 1.1e-16. This is the third instance of the same "closed form wired into one caller and not
its twin" pattern the project has already fixed twice (honest_diagnostics 2026-07-31,
`_bootstrap_fused_binary_bundle` 2026-08-04).
**Suggested fix:** at :496, when `n_bins is not None`, try
`_jackknife_ece(y_true, y_pred, n_bins=n_bins)` first and fall back to `_jackknife_metric` only on its
documented `None` return -- mirroring `_bootstrap_fused_binary_bundle.py:305-311` exactly.
**Evidence:** read `_bootstrap_ece_with_indices` (policy.py:437-498) and `_jackknife_ece`
(`_bootstrap_jackknife.py:222-273`); read the already-corrected twin at
`_bootstrap_fused_binary_bundle.py:300-311` whose inline comment describes this same gap being closed there.
`_bootstrap_ece_with_indices` already receives `n_bins` (:443) and already special-cases on it at :466, so the
information needed to take the fast path is present.

**Disposition:** RESOLVED. `_jackknife_ece` is tried first, falling back to the generic gather on its documented `None`. Verified bit-identical to 1e-12 against the gather path.

### METRICS-4 [P1] wrong-quantile
**File:** src/mlframe/evaluation/noise_band.py:24, :78
**Summary:** `cv_score_equivalence_band(method="sem")` -- the default -- multiplies a standard error
ESTIMATED from `n_folds` observations by a normal-distribution quantile `z_{1-alpha/2}` instead of the
Student-t quantile `t_{n-1, 1-alpha/2}`, so the band it returns is materially narrower than the CI half-width
its docstring claims it is.
**Failure scenario:** the canonical usage, `cv_score_equivalence_band(fold_scores)` with 5 folds and
`alpha=0.05`. The function returns `1.960 * sem`; the 95% CI half-width for a mean whose SE was estimated
from 5 values is `t_{4,0.975} * sem = 2.776 * sem`. The shipped band is 29% too small (the true band is
1.42x wider), so its actual coverage is ~86%, not the documented 95%. Every consumer inherits it:
`is_within_noise_band` (:98) and `triage_cv_delta`/`CVDeltaHistory.pooled_band`
(`cv_delta_triage.py:68, :131`), whose entire job is to stop selection loops from accepting noise -- an
under-wide band accepts noise, which is precisely the failure the module was written to prevent. The project
has already diagnosed and fixed this exact defect elsewhere: `calibration/policy.py:576-597`
(`_heldout_ece_ci`) switched from z to `scipy.stats.t` with the comment "the prior normal-z quantile
understated the interval width at the typical k=5".
**Suggested fix:** make the quantile depend on the sample count -- replace `_two_sided_z(alpha)` with a
cached `_two_sided_t(alpha, df)` using `scipy.stats.t.ppf(1 - alpha/2, n - 1)` (keep the `lru_cache`, key on
`(alpha, df)`; the profiling rationale in the docstring at :20-23 is unaffected since `df` is as stable as
`alpha` in a selection loop). `CVDeltaHistory.pooled_band` should use its own `pooled_dof`, which it already
tracks, as the df -- so the pooled path correctly converges toward z as history accumulates.
**Evidence:** read `noise_band.py` in full and `cv_delta_triage.py` in full. `_two_sided_z` at :18-24 is a
pure function of `alpha` with no `df` parameter; it is the only quantile used on the `"sem"` branch (:78).
The `"std"` branch returns the raw std and is unaffected. `grep -rn "norm.ppf|stats.t"` over the cluster
confirms `noise_band` is the only place a z is applied to a small-sample estimated SE.

**Disposition:** RESOLVED, and wider than reported. `_two_sided_z` became `_two_sided_t(alpha, df)`; `CVDeltaHistory.pooled_band` uses its own pooled dof. Fixing it exposed a second defect in the same band: it brackets a DIFFERENCE of two fold-score means but was derived from one mean, up to sqrt(2) too narrow. `triage_cv_delta` now uses `two_sample_score_band` and `is_within_noise_band` scales by sqrt(2). Measured false-positive rate on a null delta fell from 0.12-0.26 to 0.04-0.07 against a nominal 0.05. `tests/evaluation/test_noise_band_uses_the_t_quantile.py`.

### METRICS-5 [P1] evaluation-honesty
**File:** src/mlframe/evaluation/expanding_window_leakage.py:181-200
**Summary:** `auto_remediate=True`'s `remediation_verified` flag is a tautology -- the verification callback
ignores its `fit_df` argument, so the recursive re-check's "leaky" and "honest" feature arrays hold identical
values and the reported inflation is exactly 0.0 by construction, meaning the field is `True` for every
input, including one where remediation genuinely failed.
**Failure scenario:** any call with `auto_remediate=True`. `_remediated_fit_transform(fit_df, transform_df)`
(:181-185) returns `remediated_sorted[transform_df.index]` regardless of `fit_df`. Inside the recursive call,
the leaky branch computes `fit_transform_fn(df_sorted, df_sorted)` -> `remediated_sorted` (whole array) and
slices it with `fold_idx_all` (:145); the honest branch computes
`fit_transform_fn(df_sorted.iloc[train_idx], df_sorted.iloc[fold_idx_all])` -> `remediated_sorted[fold_idx_all]`
(:139). These are element-for-element equal, so `cross_val_score` is handed identical X in both branches,
`leaky_score - honest_score == 0.0` for every fold, `inflation == 0.0`, `leak_detected` is `False`, and
`remediation_verified` is `True` unconditionally. The docstring at :100-101 claims it proves the suggested
recomputation boundary actually removes the inflation rather than masking it -- it proves nothing and cannot
ever return `False`.
**Suggested fix:** the verification must re-derive the feature from `fit_df` rather than look it up. Either
(a) make the check meaningful by having the verification pass call the ORIGINAL `fit_transform_fn` for its
honest branch and the stitched `remediated_feature` only for its leaky branch, so a residual leak in the
stitching shows up as a non-zero gap, or (b) delete `remediation_verified` and stop reporting a guarantee the
code does not compute. Add a test with a deliberately-broken remediation that asserts
`remediation_verified is False`.
**Evidence:** read `detect_expanding_window_feature_leakage` in full (:97-202). The closure's own comment at
:183 states the mechanism outright (Ignore fit_df). The recursive call at :187-196 passes that closure as
`fit_transform_fn`, and lines :139/:145 are the only two places `fit_transform_fn` output enters the score.

**Disposition:** RESOLVED. The verification re-scores the caller-visible remediated array against an honest per-fold refit instead of recursing with a callback that ignored its `fit_df`; the measured gap is reported as `remediation_inflation` and the flag can now be False. `tests/evaluation/test_remediation_verified_can_be_false.py`.

### METRICS-6 [P2] evaluation-honesty
**File:** src/mlframe/calibration/smoothed_override_backtest.py:143-159
**Summary:** `backtest_override` selects `safe_threshold` by scanning bucket improvements on the supplied
rows and then reports `mae_blend_safe` computed on those SAME rows, and the docstring presents that number as
what a caller who thresholds on `safe_threshold` would actually get in production.
**Failure scenario:** an override source that is pure noise. With `n_buckets=5` and finite data, each bucket's
`improvement` has a roughly 50% chance of being positive by luck; the top-down scan at :144-148 stops at the
first non-positive bucket, so on average it selects a top region whose in-sample improvement is positive by
selection, not by signal. `mae_blend_safe` at :159 is then the blended MAE restricted to exactly that
selected region -- an optimistically biased number, and the field a caller is told to read as the production
estimate. The bias grows as `n_buckets` grows and as bucket population shrinks; on a rare-event target with
5 buckets over a few thousand rows it is easily larger than the real effect.
**Suggested fix:** either split the input (select `safe_threshold` on one half, report `mae_blend_safe` on the
other) or compute `mae_blend_safe` under a small K-fold: choose the threshold on K-1 folds, score the blend
on the held-out fold, average. At minimum, rename the field and amend the docstring so it no longer claims to
be a production estimate, and report the number of buckets the scan consumed so a reader can judge the
selection cost.
**Evidence:** read the module in full. `safe_threshold` is derived at :144-148 from `buckets`, which come from
`y_true_arr`/`blended_all` over the whole input (:126-137); `safe_mask` at :150 and the MAE at :159 use the
same `y_true_arr`. There is no held-out split anywhere in the function.

**Disposition:** RESOLVED. `mae_blend_safe_heldout` picks the threshold on K-1 folds and scores the blend on the held-out fold, averaged; the docstring and the rendered summary both label `mae_blend_safe` as in-sample and point at the new field. NaN when the input is too small to split, rather than inventing a number. `tests/calibration/test_threshold_and_override_report_honest_scores.py`.

### METRICS-7 [P2] evaluation-honesty
**File:** src/mlframe/calibration/threshold_optimizer.py:147-157
**Summary:** `optimize_decision_threshold` returns `best_score` = the maximum of `metric_fn` over 200
candidate thresholds evaluated on the same `(y_true, y_proba)` the threshold was selected from, with no
caveat in the return-value documentation that this is a selected maximum and therefore upward biased.
**Failure scenario:** `optimize_decision_threshold(y_val, p_val, f1_score)` on 500 rows with a 5% positive
rate (25 positives). The 200-point sweep is a 200-way maximisation over roughly 25 informative events; the
resulting `best_score` routinely exceeds the F1 the same threshold achieves on fresh data by a wide margin,
and a caller reading `result["best_score"]` as the operating point's F1 will over-promise. The `cv_report`
extension (:175-189) reports how much the THRESHOLD moves but never how much the SCORE shrinks out of sample,
so nothing in the returned dict surfaces the bias. The per-group path at :167-168 is worse: each segment's
`best_score` is a 200-way maximisation over as few as `min_group_size=20` rows.
**Suggested fix:** add a held-out or K-fold `best_score` to the `cv_report` (fit the threshold on K-1 folds,
score on the held-out fold, average) and document `best_score` explicitly as an in-sample selected maximum.
The module docstring already tells callers to pass a validation fold; the return-value doc at :135-141 should
carry the same warning that `_threshold_optimization.py`'s own HOLDOUT CONTRACT paragraph carries.
**Evidence:** read the module in full. `scores` (:147-150) and `best_idx` (:152) are computed on the
function's own `y_true`/`y_proba`; `best_score` at :155 is `scores[best_idx]`. No resampling or held-out
evaluation of the score exists anywhere in the file.

**Disposition:** RESOLVED. `cv_report` gains `heldout_score_mean` -- the threshold chosen on each fold's train side and SCORED on the held-out side -- and the return-value documentation now states plainly that `best_score` is an in-sample selected maximum, with the arithmetic (200 tries against ~25 informative events at 500 rows / 5% positives) and a pointer to the honest companion. Same test file.

### METRICS-8 [P2] contract-drift
**File:** src/mlframe/calibration/threshold_optimizer.py:60-66
**Summary:** `_threshold_stability_report` refits the threshold on each KFold's TEST index (`n/n_splits`
rows) rather than on the complement, so the reported fold-to-fold coefficient of variation reflects the
instability of a threshold fitted on a fifth of the data, not of the threshold the caller actually gets.
**Failure scenario:** `optimize_decision_threshold(y, p, f1, cv=5)` on n=5000. Each fold's threshold is
fitted on 1000 rows, whereas the returned `best_threshold` was fitted on all 5000. The spread of the fold
thresholds is inflated by roughly `sqrt((n - n/k) / (n/k)) = 2.0` at k=5 relative to a leave-one-fold-out
refit, so `cv` is about 2x too large and `is_stable` (compared against the default
`stability_cv_threshold=0.15` at :99) reports unstable for thresholds that are in fact stable at the full
sample size. The docstring at :49 says "Fit the threshold independently on each of n_splits folds", and the
parameter doc at :125-127 says the threshold is refit independently on cv random folds -- neither states that
the fit set is one fifth of the data.
**Suggested fix:** use the train index (`for i, (train_idx, _) in enumerate(kfold.split(...))`) so each
fold's threshold is fitted on `n - n/k` rows, which is the leave-one-fold-out analogue of the full-data fit,
and recalibrate the default `stability_cv_threshold` against the new smaller spread. If the current
one-fifth-fit semantics is intentional, say so explicitly in both docstrings and explain why the resulting CV
is comparable to a full-data threshold.
**Evidence:** read `_threshold_stability_report` in full. `sklearn.model_selection.KFold.split` yields a
`(train_index, test_index)` pair; line :60 discards the first element with `_` and binds the second to
`fold_idx`, which is then the ONLY index used for the fit at :61-66.

**Disposition:** RESOLVED. `KFold.split` yields `(train, test)` and the first element was being discarded; each fold's threshold is now fitted on the train index (`n - n/k` rows), the leave-one-fold-out analogue of the full-data fit the caller receives. The test counts the MODAL row count across metric calls, which distinguishes the sweep from the single held-out scoring call. Same test file.

### METRICS-9 [P2] contract-drift / multiplicity
**File:** src/mlframe/evaluation/compare_cv_schemes.py:112-117; src/mlframe/evaluation/cv_delta_triage.py:131
**Summary:** `compare_cv_schemes(significance_alpha=...)` runs one significance test per non-winning scheme
against a winner that was itself selected as the minimum over all schemes on the same folds, with no
multiplicity correction -- and the Bonferroni knob that exists for exactly this
(`cv_score_equivalence_band(n_comparisons=...)`) is unreachable, because `triage_cv_delta` neither accepts
nor forwards `n_comparisons`.
**Failure scenario:** `compare_cv_schemes(..., schemes={5 candidate schemes}, significance_alpha=0.05)`.
`best_scheme` at :91 is an argmin over 5 gap values, then :104-120 tests it against each of the other 4 at
alpha=0.05 each. Under a true null (all schemes equally good), the family-wise probability that the
post-hoc-selected winner clears all 4 tests is well above the nominal 5%, so `best_scheme_significant=True`
is returned for a difference that is pure fold noise -- exactly the failure mode `n_comparisons` was added to
`noise_band.py` to prevent, on the module in the cluster that most obviously needs it.
`grep -rn n_comparisons src/` shows it is read only inside `noise_band.py`; no production caller anywhere
passes anything but the default 1.
**Suggested fix:** add `n_comparisons: int = 1` to `triage_cv_delta` and forward it to
`cv_score_equivalence_band` at :131 (and to `CVDeltaHistory.pooled_band`); have `compare_cv_schemes` pass
`n_comparisons=len(other_names)`. Document in `compare_cv_schemes`'s return-value section that
`best_scheme_significant` is a post-hoc comparison against a selected winner.
**Evidence:** read both modules in full plus `noise_band.py`. `triage_cv_delta`'s signature (:71-79) has no
`n_comparisons`; its only call to `cv_score_equivalence_band` (:131) passes `alpha` and `method` only.
`compare_cv_schemes` calls `triage_cv_delta` in a loop over `other_names` (:104-118) with a fixed
`alpha=significance_alpha`.

**Disposition:** RESOLVED. `triage_cv_delta` accepts `n_comparisons` and divides alpha by it before building either band, and `compare_cv_schemes` passes `len(other_names)` -- the number of runner-ups its post-hoc-selected winner is tested against. A non-positive family size is refused. Same test file.

### METRICS-10 [P2] statistical-contract
**File:** src/mlframe/evaluation/cv_delta_triage.py:119-131
**Summary:** `triage_cv_delta` compares a difference of two fold-score MEANS against a band derived from the
BASELINE arm's spread alone; `candidate_fold_scores` is read only for its mean, so the candidate's own
fold-to-fold variance never enters the decision even though the docstring states the two arrays are paired by
fold and the correct paired statistic is available.
**Failure scenario:** baseline fold scores `[0.800, 0.801, 0.799, 0.800, 0.800]` (std 7e-4) versus candidate
`[0.760, 0.840, 0.770, 0.830, 0.805]` (std 0.036, same mean 0.801). `delta = 0.001`; the band is built from
the baseline's tiny SEM (`z * 7e-4/sqrt(5) = 6e-4`), so `abs(delta) > band` and the call returns
`actionable=True` -- declaring a difference of 0.001 likely LB/OOS-actionable for a candidate whose own fold
spread is 36x larger than the claimed effect. The correct paired test uses
`std(candidate - baseline)/sqrt(n)`, which here is about 0.016 and yields `actionable=False`.
**Suggested fix:** compute the band from the per-fold differences,
`cv_score_equivalence_band(candidate_fold_scores - baseline_fold_scores, ...)`, which is exactly the paired
statistic the paired-by-fold contract licenses and which is correct whether the two arms are correlated or
not. Keep the current one-arm band only behind an explicit flag for the unpaired case, and name which is
being used in the returned `reason` string. The same applies to `CVDeltaHistory.pooled_band` (:60-68), which
pools baseline-arm variances rather than difference variances.
**Evidence:** read `triage_cv_delta` in full. `candidate_fold_scores` appears only at :113-115 (coercion and
shape check) and :119 (`np.mean`); the band at :126/:131 is a function of `baseline_fold_scores` only.

**Disposition:** RESOLVED, but by a different statistic than the one suggested, and the difference is load-bearing.

The defect is real and is closed: the band is now built by `two_sample_score_band`, whose SE is `sqrt((var_a + var_b)/n)`, so the candidate arm's own spread enters the decision. On the finding's own fixture the band comes out at 0.03654 against a delta of 0.001, so `actionable=False` -- the reported failure no longer reproduces. `pooled_band` was corrected in the same way, using `std * sqrt(2/n_folds)` rather than `std/sqrt(n_folds)`.

The suggested PAIRED band, `cv_score_equivalence_band(candidate - baseline)`, was measured and REJECTED as the default. On a candidate that beats the baseline by a perfectly consistent +0.010 on all five folds -- the shape of a genuine, well-behaved improvement -- `std(candidate - baseline)` is exactly 0, so the paired band is exactly 0.0 and any delta is declared actionable with infinite confidence off five folds. For a gate whose stated failure mode is ACCEPTING NOISE, that is the wrong direction to be wrong in. On the finding's own fixture the paired band is 0.0434, wider than the two-sample band, not the 0.016 the finding computes, so the paired form is not uniformly tighter either. The two-sample band is more conservative in both regimes and cannot degenerate. `tests/metrics/test_band_stability_and_ktc_diagnosability.py` pins both fixtures, including the zero-band case, so a future switch to the paired form has to confront it.

### METRICS-11 [P3] dtype-dependent-metric
**File:** src/mlframe/metrics/_log_loss_and_separation.py:161-162
**Summary:** `fast_log_loss`'s default clipping epsilon is `np.finfo(y_pred.dtype).eps`, so the same
probabilities produce a materially different log-loss depending only on whether the caller handed over a
float32 or a float64 array -- which makes the metric non-comparable across models that emit different
prediction dtypes.
**Failure scenario:** a LightGBM model returning float32 probabilities and a linear model returning float64
are scored with `fast_log_loss` in the same report. A confidently-wrong row (`p=0` for a positive) is
penalised `-log(1.19e-7) = 15.9` for the float32 model and `-log(2.22e-16) = 36.0` for the float64 model.
On a target with even a handful of such rows, the float32 model wins the comparison purely on dtype. The
sibling `fast_log_loss_binary` (:120) defaults to a fixed `1e-15` and does not have this property, so which
of the two entry points a call site happens to use silently changes the number.
**Suggested fix:** the per-dtype rationale in the docstring (:141-145) is sound for a SINGLE array, but the
cross-model comparability consequence is not documented. Either add an explicit warning to the docstring that
`fast_log_loss` results are not comparable across dtypes and that a cross-model report must pass an explicit
`eps`, or have callers that build comparison tables upcast to float64 once at the boundary and pass
`eps=1e-15` explicitly.
**Evidence:** read `_log_loss_and_separation.py` in full; `eps` is threaded straight into the kernels'
`max(eps, min(1 - eps, p))` clip at :63 and :102-105, and `fast_log_loss` is the only entry point that
derives it from the input dtype.

**Disposition:** RESOLVED as documentation, which is what the finding asks for. The per-array rationale in the docstring is correct and is kept; what was missing is that it makes two models' scores incommensurable. The docstring now says so explicitly, gives the concrete 15.9-vs-36.0 penalty for a confidently-wrong row, states that a cross-model table must pass one explicit `eps` or upcast to float64 at the boundary, and names the `fast_log_loss_binary` divergence so the entry-point choice is visible.

### METRICS-12 [P3] diagnosability
**File:** src/mlframe/calibration/_ktc_dispatch.py:49-51, :54-56, :128-129
**Summary:** three broad `except Exception` handlers silently downgrade `odds_ratio_combine`'s backend
selection to a hardcoded size threshold and log only at DEBUG, so a persistent kernel-tuning-cache failure is
invisible in production.
**Failure scenario:** a transient or persistent fault while importing `mlframe.feature_selection.filters`
(the module that probes CUDA at import time -- the exact mechanism CLAUDE.md records as having silently
downgraded the whole process's MI backend on 2026-08-02) makes `_get_cache()` return `None` for the remaining
life of the process. Every subsequent `choose_odds_combine_backend` call returns the caller's `fallback` size
threshold instead of the measured per-host winner, and the only trace is a `logger.debug` line that
production logging does not emit. The measured spread between backends on this host (module docstring, :7-9)
is up to 9x, so the cost is real.
**Suggested fix:** narrow :49 to `ImportError` (the one genuine package-absent case) and log any other
exception at WARNING, matching the fix already applied to `_select_mi_backend`; log the :128 handler at
WARNING through `log_throttle` so a repeated cache hiccup is surfaced once rather than never.
**Evidence:** read `_ktc_dispatch.py` in full. All three handlers catch bare `Exception` and call
`logger.debug`; there is no path that raises or warns.

**Disposition:** RESOLVED exactly as suggested. The import guard is narrowed to `ImportError` (the one genuine package-absent case, which stays at debug); any other exception from that import -- the module probes CUDA at import time, so a transient device fault lands here -- now warns through `log_throttle`, as do the singleton and lookup handlers. Throttled rather than plain warnings because the lookup sits on a dispatch path. `tests/metrics/test_band_stability_and_ktc_diagnosability.py`.

### METRICS-13 [P3] test-quality
**File:** tests/metrics/test_warmup_skip_parallel_env_gate.py:92, :105
**Summary:** the only two tests that assert `MLFRAME_NUMBA_WARMUP_SKIP_PARALLEL` actually changes which
kernels get warmed are both `@pytest.mark.skip`, so the feature's core contract has no coverage at all.
**Failure scenario:** a refactor of `_prewarm_numba_cache_body` that stops honouring the env var (or honours
it for `_seq` kernels too) passes CI unchanged. The one surviving test
(`test_skipped_par_kernel_still_works_correctly_via_lazy_compile`, :108) only checks that a `_par` kernel
still computes correctly after lazy compilation, which is true whether the flag works or not.
**Suggested fix:** the recorded blocker (monkeypatching an njit dispatcher is not observed by the warmup
body) is real, but it is sidesteppable: run the warmup in a fresh subprocess with the env var set and
without, and compare `len(kernel.nopython_signatures)` for the `_par` kernels in each child -- a fresh
process has no prior compilation state, which is exactly the fresh-compile-state reset mechanism the skip
reason says is missing.
**Evidence:** read the file. `_SKIP_REASON` (:76-89) documents the harness limitation; both
`@pytest.mark.skip` decorators reference it. The file contains no other assertion about which kernels are
warmed.

**Disposition:** RESOLVED via the suggested subprocess approach, and the recorded blocker turned out to be entirely sidesteppable. A fresh interpreter has no prior compilation state, so `nopython_signatures` becomes decisive: measured, with the flag off all six `_par` kernels have >= 1 signature; with it on all six have exactly 0 while `_fast_mae_seq` still has 1. Both `@pytest.mark.skip` decorators are gone, replaced by three `@pytest.mark.slow` tests (the two directions plus one that asserts both runs together, so neither can pass by the kernels simply never warming on the host). Marked slow because each subprocess pays a full numba warmup; the pair takes about ten minutes here.

### METRICS-14 [P3] test-quality
**File:** tests/metrics/classification/test_classification_extras.py:346
**Summary:** a wall-clock assertion (`assert med >= 1.02`) makes a property of the host machine into a test
of the code -- it pins a 1.02x median speedup of the fused KS gate over a reference implementation.
**Failure scenario:** the test already had its floor lowered once (1.05 -> 1.02, per the comment at :348-353)
because CI hardware measured 1.0498x. A slower or more contended runner, a different numba version's codegen,
or a host where the two kernels genuinely tie will fail this test with no code defect. The contention probe
at :338 only skips when the ref-block spread exceeds 2.0x, which does not catch a uniformly-slow host.
**Suggested fix:** move the timing assertion behind a slow/perf marker excluded from the default run and keep
a behavioural test (fused gate returns bit-identical KS to the reference on tie-free and tied inputs) in the
default suite. A perf regression belongs in `_benchmarks/`, where a number that drifts with hardware is
expected.
**Evidence:** read :300-355. The assertion compares `min(_timed_block(...))` wall times of two Python
callables; nothing about the code's OUTPUT is asserted in this test.

**Disposition:** RESOLVED as suggested. `test_ks_fused_gate_perf_sentinel` keeps its measurement but carries `@pytest.mark.slow`, so it is deselected from the CI run that was flaking it, and a new behavioural test in the default suite asserts the fused gate returns bit-identical KS to the reference on tie-free, heavily-tied and all-tied scores -- which is the contract that actually protects the code.

### METRICS-15 [P3] bootstrap-degeneracy
**File:** src/mlframe/calibration/prediction_band_correction.py:158
**Summary:** `assess_prediction_band_stability` substitutes the meaningful value `1.0` (no correction) into
the bootstrap distribution whenever a resample's in-band `mean(y_pred)` is exactly zero, contaminating
`bootstrap_mean`, `bootstrap_std`, and both CI endpoints with a value that is not a draw from the estimator's
sampling distribution.
**Failure scenario:** a band `(lo, hi]` with `lo = -1e-9, hi = 1e-9` over near-zero predictions, or any band
where positive and negative predictions cancel. A meaningful fraction of the 500 resamples hit
`sample_pred_mean == 0.0` and are recorded as exactly `1.0`; `bootstrap_std` then measures the spread between
the real factor and a pile of 1.0s, and `is_stable` at :166 is decided from that mixture. The point estimate
(`find_prediction_band_shift`, :44-45) uses the same `1.0` convention, so the inconsistency is invisible from
the output.
**Suggested fix:** drop degenerate resamples from `boot_factors` (track and report the drop count) rather
than substituting a value, and return `is_stable=False` when too many were dropped -- mirroring the
skip-and-count discipline `bootstrap_metrics` uses at `evaluation/bootstrap.py:531-533` and :559-566.
**Evidence:** read the module in full. `boot_factors` is pre-allocated at `n_bootstrap` (:154) and every slot
is filled at :158, with no valid-count tracking; :160-165 reduce the full array.

**Disposition:** RESOLVED with the suggested skip-and-count discipline. Degenerate resamples are dropped rather than recorded as 1.0, the count is warned, `is_stable` is False when more than a tenth of the resamples degenerated, and fewer than two survivors returns a NaN-uncertainty report instead of a confident one. `tests/metrics/test_band_stability_and_ktc_diagnosability.py` pins that its own fixture genuinely produces degenerate resamples, so the assertions cannot pass for an unrelated reason.

## Coverage

Read in full:
- src/mlframe/metrics/classification/_threshold_optimization.py
- src/mlframe/metrics/_log_loss_and_separation.py
- src/mlframe/metrics/_core_precision_mape.py
- src/mlframe/calibration/threshold_optimizer.py
- src/mlframe/calibration/smoothed_override_backtest.py
- src/mlframe/calibration/prediction_band_correction.py
- src/mlframe/calibration/_ktc_dispatch.py
- src/mlframe/evaluation/_bootstrap_jackknife.py
- src/mlframe/evaluation/noise_band.py
- src/mlframe/evaluation/cv_delta_triage.py
- src/mlframe/evaluation/compare_cv_schemes.py
- src/mlframe/evaluation/expanding_window_leakage.py

Read in the sections that matter:
- src/mlframe/calibration/policy.py (:55-300 ECE kernels and label normalisation, :365-499 resample/CI
  helpers, :500-893 `pick_best_calibrator`, inner-CV folds, held-out CI, reliability plot)
- src/mlframe/calibration/quality.py (:215-320 `bin_predictions` / `estimate_calibration_quality_binned`;
  full njit-decorator inventory)
- src/mlframe/evaluation/bootstrap.py (:318-596 `bootstrap_metrics`, :598-660 `auc_variance`, :726 `auc_ci`)
- src/mlframe/evaluation/_bootstrap_fused_binary_bundle.py (:160-340 resample generation, fused bundle,
  jackknife wiring)
- src/mlframe/evaluation/reports.py (:118-330 `evaluate_estimators` fit/eval split -- the CatBoost
  early-stopping carve-out at :213-250 correctly reports on the non-early-stopping half of the test set)
- src/mlframe/evaluation/blend_source_selection.py (:1-120)
- src/mlframe/evaluation/group_leakage_guard.py (:1-60)
- src/mlframe/metrics/_drift.py (:100-400 PSI / KL / JS / Wasserstein-1 / KS kernels)
- src/mlframe/metrics/iteration_metrics.py (:1-60, :120-200 binary and multiclass aggregators)
- src/mlframe/metrics/_core_auc_brier.py (:380-470 bootstrap AUC resamplers and the tie-free gate)
- tests/metrics/test_warmup_skip_parallel_env_gate.py; tests/metrics/classification/test_classification_extras.py (:300-355)

Verified clean by targeted repo-wide grep over `metrics/`, `calibration/` and `evaluation/`:
- no raw-power-sum / binomial-expansion skew or kurtosis anywhere in the cluster (`s3/n`, `3*mean`,
  `2*mean**3`, `sum_x3`, `sum_x4`, `m3 =`, `m4 =` -- zero hits)
- no additive epsilon in a moment denominator (`std**3 + 1e-12`, `var*var + 1e-12`, `**1.5 + eps`); the only
  `**1.5` in the cluster is the correct BCa acceleration denominator at `_bootstrap_jackknife.py:76`
- no JSON serialization feeding a hash / cache key (`json.dumps`, `orjson.dumps`, `hashlib`, `md5`, `sha256`
  -- zero hits); no `__getstate__` and no instance-attached runtime cache in the cluster
- no `inspect.getsource()` assertions in `tests/metrics`, `tests/calibration`, `tests/evaluation`
- no `xfail` anywhere in the three test packages; every `pytest.skip` other than METRICS-13/14 is a genuine
  optional-dependency or no-CUDA guard
- `default_via_or` sweep (`x or default`): only two hits, both benign -- `_ktc_dispatch.py:125`
  (`(result or {})`, an empty dict yields the same outcome) and `bootstrap.py:452` (`os.cpu_count() or 1`,
  correct since `cpu_count()` returns `None`, never 0)
- dispersion estimators consistently pass `ddof=1` with a guarded `n > ddof` denominator check
  (`_fairness_metrics.py:396` and :480, `bootstrap.py:658`, `constant_group_leak_scan.py:119`)
- checked line by line and correct: `bootstrap_metrics`'s point-estimate / resample / CI flow and its
  failure accounting; `_ci_from_samples`'s BCa degeneracy fallbacks; `_jackknife_mean_metric`'s single-class
  exclusion; `_jackknife_auc`'s Mann-Whitney placement-value algebra; `_jackknife_ece`'s per-bin closed form;
  `bootstrap_auc_brier_ll_ece_batch`'s tie-free vs grouped kernel selection and RNG-order-preserving index
  generation; `make_bootstrap_auc_resampler` / `bootstrap_auc_distribution_parallel`'s base-rank counting
  (order-independent, so the METRICS-1 defect does not affect them); `expanding_window_leakage`'s
  sorted/original index round-trip (`remediated_sorted[inverse_order]` is correct)
