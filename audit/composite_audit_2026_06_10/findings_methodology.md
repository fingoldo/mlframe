# Composite Targets audit - ML/DS methodology vs world-class practice

Agent dimension: statistical rigor of discovery/ensemble - MI estimator bias, single- vs multi-seed
selection, multiplicity over bases x transforms, val-vs-honest-holdout discipline, threshold-default
justification, small-n overfitting, uncertainty quantification, reproducibility.

Scope read in full (current code, 2026-06-10):
src/mlframe/training/composite/discovery/{__init__,_fit,_eval,_filter,_screening_tiny,_tiny_rerank,_auto_base,_stacked,forward_stepwise,bayesian,screening,auto_detect}.py,
src/mlframe/training/composite/ensemble/{__init__,_cross_target,stacking,feature_stacking}.py,
plus training/_composite_target_discovery_config.py, training/_cv_aggregation.py,
core/_phase_composite_post_xt_ensemble/__init__.py (call-site verification),
composite/streaming.py, and tests/composite_discovery_audit_notes.md.

Deduplication: sibling findings files (findings_discovery_core.md D1-D25, findings_discovery_adv.md
A1-A34, findings_ensemble.md N1-N28) were read; items already reported there (Wilcoxon power floor
A2, multiseed no-op A3, fold-scheme mismatch A4, domain-subset asymmetry A5, dead early-stop A12,
per-bin raw protocol A14, sqrt(2) drift SE D5, stratified-truncation D6, config-mutation D7,
sample-weight carve misalignment N1, OOF fitted_params reuse N6, etc.) are NOT re-reported here.
One sibling "verified-clean" claim is disputed (M10).

Verified-clean from my dimension (checked, no finding):
- Train-only discipline of fit: every fitted quantity (transform params, MI edges, clip bounds,
  drift halves, rerank sample) derives from train_idx only; val_idx/test_idx are stashed and
  never read during discovery. Matches the user's val=ES / test=honest terminology.
- Default mi_estimator="bin" (quantile binning) for screening: quantile bins are invariant under
  strictly monotone marginal transforms, so the R10b rationale (knn/KSG bias asymmetric between
  fat-tailed y and sub-Gaussian T inflates mi_gain) is sound; plug-in bin-MI bias (~(nbins-1)^2/2n)
  cancels to first order in the mi_t - mi_y difference because both terms use the same nbins and
  the same row subset (_eval.py:181-196 recomputes mi_y on the shrunken domain - correct pairing).
- oof_holdout_source="kfold" default for ensemble weighting avoids re-using the ES val surface
  for stacking weights - correct Sill-et-al hygiene, better than most production stacks.
- bayesian_alpha_fit conjugate NIG math (already verified clean by sibling; agreed).
- Gate uses bootstrap LCB (entry["mi_gain_lcb"], _fit.py:469) not the point estimate when
  bootstrap is on - matches audit-notes claim, still true in current code.

---

## M1 - P2 - bug - discovery/_fit.py:221-263 (with screening.py:497-501)
**Heavy-tail mi_n_strata auto-boost is a complete no-op under the default mi_sample_strategy="random"**

fit detects heavy-tail y (|skew|>2 or kurt>5) and boosts mi_n_strata 10 -> 30 via
model_copy(update={"mi_n_strata": boost}), logging "boosted mi_n_strata 10 -> 30". But grep over
src shows mi_n_strata is consumed ONLY as the n_strata argument of _sample_indices
(_fit.py:293, _tiny_rerank.py:109, _auto_base.py:152), and _sample_indices ignores
n_strata entirely when strategy == "random" (screening.py:500-501 returns the uniform draw
before any stratification logic). Config default is mi_sample_strategy="random"
(_composite_target_discovery_config.py:327). So on exactly the heavy-tail targets the mechanism
was built for, the boost changes nothing - the log line claims a corrective action that has zero
effect. This violates the project's "enable corrective mechanisms by default" rule: the actual
corrective lever for heavy tails is stratified_quantile sampling, which stays off.

**Fix:** on heavy-tail detection, flip BOTH knobs in the same model_copy:
mi_sample_strategy="stratified_quantile" AND mi_n_strata=boost (log both). Alternatively gate
the boost+log on mi_sample_strategy != "random" so the log never claims a no-op. Add a biz_value
test: heavy-tail synthetic where stratified screening recovers a tail-driven spec that random
sampling misses.

## M2 - P2 - leak - discovery/_screening_tiny.py:557-612 (esp. 604-611) + discovery/_eval.py:209-274
**Tiny-CV rerank scores transforms whose parameters were fitted on the full train sample - each CV val fold helped fit the transform it is evaluated on**

_tiny_cv_rmse_y_scale receives fitted_params (fit once on ALL of y_train[valid] in
_eval.py:56) and computes t_clean = transform.forward(...) over the whole screening sample
(line 610) BEFORE the K-fold split. Every fold's validation rows therefore participated in fitting
the transform whose quality the fold measures. For 2-parameter OLS transforms this is negligible,
but the DEFAULT transform list (_composite_target_discovery_config.py:82-109) includes flexible
fitters - monotonic_residual (PCHIP knots), smoothing_spline_residual, quantile_residual,
rank_residual, polynomial_residual_deg2 - where in-fold transform leakage is material at
tiny_model_sample_n=20_000. The raw-y baseline (_tiny_cv_rmse_raw_y) has NO fitted transform,
so the comparison the gate/ranking rests on is structurally tilted toward flexible composite
transforms. World-class practice (sklearn TransformedTargetRegressor evaluated inside
cross_val_score; M-competitions residual-modeling protocols) refits the target transform inside
each fold. Related but distinct from sibling N6 (ensemble OOF path) and A5 (row-subset asymmetry);
this is the discovery-rerank instance. Same root cause makes the MI-gain bootstrap CI too narrow:
_eval.py:209-274 resamples rows but holds t_screen (and fitted_params) fixed across
replicates, so the LCB excludes transform-parameter uncertainty entirely.

**Fix:** inside _one_fold, call transform.fit(y_clean[train_fold], base_clean[train_fold]) and
forward both fold halves with the fold-local params (transforms are O(n) fits; cost is ~K extra
cheap fits per spec). Keep the full-train params for the shipped spec. For the bootstrap CI either
refit per replicate (cheap for OLS-family) or document that the CI is conditional on the fitted
transform. Regression test: a smoothing_spline_residual spec on pure-noise y must NOT beat the
raw baseline after the fix (it currently can, via in-fold spline leakage).

## M3 - P2 - bug - discovery/__init__.py:184-234 (with _screening_tiny.py:432-433,490-491)
**fit_with_stability_check is seed-jitter, not stability selection - and its seed stride collides with the multiseed stride, making "independent runs" share most of their randomness**

Two defects against the method it cites ("matches the standard stability-selection literature
(Meinshausen-Buhlmann)", line 198):

1. **No data perturbation.** M-B stability selection requires subsampling rows (~n/2 without
   replacement) per run. This implementation only varies config.random_state
   (line 210). The seed reaches _sample_indices (which subsamples ONLY when
   train_idx.size > mi_sample_n = 100k default) and the tiny-CV fold splitters. On any dataset
   with <=100k train rows the screening sample is the IDENTICAL np.arange(n) in every run, so
   "stability" reduces to KFold-split jitter; combined with sibling A3 (TimeSeriesSplit /
   GroupKFold / deterministic families ignore the seed) the n runs can be bit-identical and every
   spec trivially scores n/n - the filter silently passes everything exactly in the small-n regime
   where lucky-split wins are most likely.
2. **Seed-stride collision.** Stability runs use base_seed + i*7919 (line 210); the multiseed
   rerank inside each run derives its per-seed states as base_random_state + s_idx*7919
   (_screening_tiny.py:433/491). Run i's seed set {(i+s)*7919, s=0..2} overlaps run i+1's set in
   2 of 3 elements - consecutive "independent" stability runs share most CV splits, so survival
   counts are positively correlated across runs and overstate stability.

**Fix:** per run, draw a row subsample of train_idx (e.g. 80% without replacement, seeded) and
fit on it - that is the actual M-B perturbation; derive run seeds via the existing-but-unused
derive_seeds(base_seed, [f"stability_{i}"]) (see M12) to kill the stride collision. Add a unit
test: on n<100k data with a deterministic splitter, two stability runs must NOT produce identical
specs_ lists (or the method must warn it has no perturbation source).

## M4 - P2 - extension - discovery/_eval.py:196-274 + discovery/_fit.py:461-482 + discovery/_tiny_rerank.py:744-757
**No multiplicity control anywhere across the ~26-transform x K-base candidate family; per-candidate LCB at fixed 2.5% + top-M by point estimate is winner's-curse selection**

The default config evaluates ~26 transforms x auto_base_top_k=3 bases (~60-80 candidates after
unary dedup). The only per-candidate inferential gate is the optional bootstrap LCB at a fixed
2.5% level (_eval.py:274) - with 60+ simultaneous candidates that is an expected ~1.5 false
positives per fit even when NO composite has real gain, and nothing (Bonferroni/BH/maxT) accounts
for the family size. Downstream, _tiny_model_rerank trims to top_m_after_tiny=10 by the
argmin/lexsort of noisy CV-RMSE point estimates (line 753) - min-of-K-noisy-estimates is biased
low (winner's curse), and the per-seed arrays needed for selection-aware inference are already
collected but unused for ranking. To be fair: under the CURRENT defaults the screening gates are
intentionally disabled (eps_mi_gain=-10, require_beats_raw_baseline=False) and the honest decision
is deferred to the k-fold-OOF ensemble gate - a defensible architecture. The multiplicity gap
bites users who re-enable the gates (the config explicitly invites this for tree-only zoos,
_composite_target_discovery_config.py:430-434).

**Fix (specific):** (a) when mi_gain_bootstrap_n>0, convert per-candidate bootstrap distributions
to p-values and apply Benjamini-Hochberg across the candidate family before the eps gate - the
shared bootstrap seed (12345) already pairs replicates across candidates, so a maxT permutation
null over the family is nearly free; (b) in the rerank, rank by the paired per-seed difference
LCB vs the raw baseline (arrays already exist in _wilcoxon_per_seed_composite) or apply a 1-SE
rule below the best spec instead of the raw argmin. Both are standard selective-inference hygiene
(Kaggle stacking lore: "never pick the max of N noisy CVs").

## M5 - LOW - bug - _cv_aggregation.py:38,59-61,88 + discovery/_screening_tiny.py:374-382,770-778 + discovery/_tiny_rerank.py:641-655
**Repeated-CV inference is anti-conservative (Nadeau-Bengio) and the existing correlation_inflation remedy is never wired into any composite path**

All multi-seed machinery (median-of-seeds rerank, Wilcoxon gate, cv_selector_mode="t_lcb")
treats per-seed/per-fold RMSEs as independent samples. They are repeated CV on the SAME 20k rows:
fold scores share overlapping training sets, and seed repeats reshuffle the same data, so their
variance underestimates the true generalization variance (Nadeau & Bengio 2003; Bouckaert & Frank
2004). The infra ALREADY contains the standard remedy - aggregate_fold_scores(...,
correlation_inflation=...) whose docstring names Nadeau-Bengio explicitly - but every composite
call site (_screening_tiny.py:374-382, 770-778, forward_stepwise.py:136-143) leaves it at
the naive 1.0 and CompositeTargetDiscoveryConfig's cv_selector_* block
(_composite_target_discovery_config.py:126-130) does not expose it. The Wilcoxon gate inherits
the same anti-conservatism on top of its power-floor problem (sibling A2/A3).

**Fix:** add cv_selector_correlation_inflation to the config (default the NB factor
sqrt(1 + n_val/n_train) per split geometry, ~1.15 for 3-fold), thread it through the four tiny-CV
call sites and forward-stepwise; note the caveat in the Wilcoxon gate docstring (or replace it with
the corrected-resampled t-test, which is the literature-standard for exactly this comparison).

## M6 - P2 - leak - discovery/_tiny_rerank.py:204-207 + discovery/_fit.py:149-157 (cf. ensemble/__init__.py:525,772-783)
**Discovery has no time-signal input at all; rerank time-awareness is inferred from base monotonicity, which never fires for the flagship lag-of-y bases - temporal screening runs on shuffled KFold**

The suite knows the time ordering (it threads ctx.timestamps as time_ordering into the
ensemble OOF helper, which correctly switches to a trailing-slice holdout). But
CompositeTargetDiscovery.fit has no time_ordering parameter, so Phase B infers temporality per
spec via _is_monotone_nondecreasing(base_screen_local) (_tiny_rerank.py:207). A lag-1 of y -
the documented canonical base - is monotone only if y itself is cumulative; on ordinary temporal
data (returns, sensor series, demand) it is NOT monotone, so the tiny-CV uses
KFold(shuffle=True) for both composite and raw. Shuffled K-fold on autocorrelated rows leaks
future->past within the screening CV: absolute RMSEs are optimistic, and the composite-vs-raw and
spec-vs-spec rankings are made under a leakage regime that production (forward-walk val/test)
never sees - the exact mismatch class that time_aware was added to prevent (and that
forward-stepwise fixed by defaulting time_aware=True, C-P2-11 note in forward_stepwise.py:38-43).
Sibling A16 covers descending/constant edge cases of the detector; this finding is the missing
architecture input: the detector cannot work even in principle for non-monotone bases.

**Fix:** add time_ordering: np.ndarray | None = None to fit() (threaded from the suite phase
exactly like the ensemble path), stash it, and in _tiny_model_rerank use
time_aware = time_ordering is not None and _is_monotone_nondecreasing(time_ordering[train_idx_screen])
for ALL specs AND the raw baseline (also resolving the per-spec/global mismatch of sibling A4 in
the common case). Keep the per-base monotone heuristic only as fallback when no signal is given.

## M7 - LOW - extension - composite/__init__.py:55 + estimator/ architecture
**Classification targets are out of scope ("regression only here") although the margin-offset (init_score) analogue fits the existing architecture exactly**

The composite mechanism exists to stop a dominant feature from drowning the learner - the same
pathology occurs in classification (e.g. churn dominated by days_since_last_login). The
established analogue is residual modeling on the margin scale: fit margin = alpha*g(base)+beta
on train (logistic regression of y on the base), then train the booster on the SAME y but with
init_score/base_margin = fitted offset (LightGBM/XGBoost/CatBoost all support it natively),
inverse = sigmoid(offset + raw_score). This decomposes as a Transform-like
(fit/forward-as-offset/inverse/domain_check) and reuses the wrapper's y-clip-equivalent
(probability clipping), the tiny-rerank (logloss instead of RMSE), and the cross-target ensemble
(stack on probabilities). The suite already computes init_score-style dummy baselines, so the
plumbing precedent exists.

**Fix:** ship a margin_offset_logistic spec family + CompositeTargetEstimatorClassifier that
passes the offset via init_score/base_margin; screening metric = OOF logloss vs a raw-y tiny
baseline. biz_value test: AR-dominated binary target where raw booster underperforms the
offset-composite on a group-aware split.

## M8 - LOW - extension - ensemble/__init__.py:516-944 + _cross_target.py:485-559 + core/_phase_composite_post_xt_ensemble/__init__.py:646-720
**Honest OOF residuals are computed and then thrown away - split-conformal prediction intervals would be nearly free**

The default pipeline already produces exactly the artifact conformal prediction needs: an honest
k-fold OOF prediction matrix and aligned y (oof_holdout_source="kfold", 5 folds) for every
ensemble component AND the combined stack. After NNLS weights and the OOF gate, the residuals are
discarded. Storing one number per coverage level - the (1-alpha)(1+1/n) empirical quantile of
|y_oof - yhat_oof| of the FINAL ensemble - yields distribution-free marginal-coverage intervals
(split-conformal; the standard offering in nixtla/statsforecast and darts via
PredictionIntervals, and sklearn-adjacent via MAPIE). Zero extra fits; a few floats in
ensemble.notes.

**Fix:** at ensemble build, compute the stack's OOF residual quantiles for e.g. alpha in
{0.05, 0.1, 0.2}, store as notes["conformal_q"], expose
predict_interval(X, alpha) on CompositeCrossTargetEnsemble. Document that coverage is marginal
and exchangeability-based (for strongly temporal data, note the adaptive/weighted-conformal caveat).

## M9 - LOW - extension - _composite_target_discovery_config.py:80 + discovery/auto_detect.py:43-148 (unwired) + transforms registry
**Stationarity-aware transforms (ewma_residual / rolling_quantile_ratio / frac_diff) are excluded from default discovery while the auto-detect machinery built to enable them ships unwired**

The config excludes the three time-series transforms because they "require chronological row order
which most datasets lack at the discovery stage". But detect_time_column_candidates /
sort_df_by_time_column were written precisely to detect and establish that order
(auto_detect.py:29-39) - and grep shows ZERO production call sites: they are exported and tested
but never invoked by fit or any suite phase. So the package carries both the transforms and
their enabling machinery, permanently disconnected. M-competition evidence (M4/M5) consistently
ranks differencing/deflation-style residual targets among the top moves on trending/persistent
series - exactly frac_diff/ewma_residual's domain.

**Fix:** in fit, when the suite passes time_ordering (see M6) or
detect_time_column_candidates returns a confident candidate, evaluate the three time-series
transforms on the chronologically-ordered screening sample (sort once, screen, map indices back);
keep them out of the candidate list otherwise. This converts two shipped-but-dead features into
the standard forecasting-practice path.

## M10 - LOW - bug - discovery/_screening_tiny.py:741-754 + _cv_aggregation.py:79-94
**Early-stop partial-sum bound is NOT conservative for median_minus_mad / low-quantile selector modes (disputes the adv-agent's "verified-clean" note)**

Sibling findings_discovery_adv.md's verified-clean list states the early-stop bound
(sum_so_far > early_stop_threshold * cv_folds) "is conservative for every aggregate_fold_scores
mode with direction=min". Counterexample: folds [0.1, 0.1, 100] with thr=10, cv_folds=5 - after
fold 3, sum=100.2 > 50 fires the break; the aggregate over the truncated folds under
median_minus_mad is median=0.1 + 1.0*MAD=0 = 0.1, far BELOW the threshold, so the spec would
pass the gate with a truncated, biased-down score (the bound only guarantees the MEAN exceeds thr:
partial mean = sum/k_run >= sum/cv_folds). Same failure for mode="quantile" with
quantile_level <= 0.5. Currently latent - nothing passes early_stop_threshold (sibling A12) and
the rerank never threads cv_selector_mode (sibling A27) - but A12+A27's recommended fixes would
activate this bug in combination.

**Fix:** restrict the early-stop branch to cv_selector_mode == "mean" (or compute a
mode-specific sound bound); add a unit test with the counterexample folds pinning that
median-mode never early-stops.

## M11 - LOW - bug - discovery/bayesian.py:279-300
**bayesian_alpha_fit_bootstrap(subsample_n=m) reports m-out-of-n bootstrap percentiles as the posterior without the sqrt(m/n) width rescale**

When subsample_n < n, each replicate fits on m rows, so the spread of alphas scales like
sigma/sqrt(m) while the uncertainty of the full-data estimate is sigma/sqrt(n) - the returned
alpha_std / CI overstate the posterior width by ~sqrt(n/m) (e.g. 3.2x at n=1M, m=100k). The
m-out-of-n bootstrap literature requires re-centering the interval and shrinking the half-width by
sqrt(m/n). Conservative direction (too-wide CI), but any consumer thresholding on
alpha_ci width or comparing against the conjugate variant (which IS n-scaled) gets inconsistent
answers between the two "drop-in swappable" implementations.

**Fix:** rescale: alpha_ci = alpha_mean +/- (q - alpha_mean) * sqrt(m/n) (same for std), or WARN
in the docstring + return the raw spread under a distinct key. Unit test: subsampled CI must match
full-n CI within tolerance on Gaussian data.

## M12 - LOW - docs - ensemble/__init__.py:73-93 (zero call sites in src)
**derive_seeds - the documented anti-correlation seed mechanism - has no production caller; discovery threads one raw seed with ad-hoc +7919 strides instead**

The docstring explains exactly the right design ("Threading the same random_state through every
[randomness source] creates correlation ... Sub-seeds break the correlation while keeping
reproducibility") and the function is exported, re-exported, CHANGELOG'd and unit-tested - but
grep over src/ shows zero invocations. In practice discovery passes config.random_state
directly into MI sampling, tiny-CV splits, the bootstrap (+7919), the permutation null
(+7919), and stability runs (i*7919) - the ad-hoc identical stride is what produces the M3
seed-set collision. Documentation promises a property the codebase does not deliver.

**Fix:** actually use derive_seeds(config.random_state, ["mi_sampling", "tiny_cv", "mi_bootstrap",
"null_perms", "stability_i", "oof"]) at those six sites (one-line each), or demote the docstring
to "available for external callers" and remove the rationale paragraph.

## M13 - LOW - extension - discovery/forward_stepwise.py:153-186
**Greedy step accepts the argmin of unpaired CV means; per-fold paired differences are computed and discarded**

Each round picks best_name = argmin(cv_mean) over all remaining candidates and gates on the
relative point-estimate gain (2%). All candidates in a round are scored on the SAME folds
(kf is constructed identically per _cv_rmse_with_folds call with a fixed seed/TSS), so the
fold-level scores are paired - yet the comparison ignores pairing, leaving the selection exposed
to winner's curse over the candidate pool (min of K noisy means is biased low; the 2% gate
partially, but not size-adaptively, compensates - it is the same fixed threshold for a pool of 3
or of 25 candidates). The fold lists are already returned (cv_persist_fold_scores).

**Fix:** accept a candidate only if it beats rmse_current on a majority of folds AND the paired
mean gain clears the gate (or use the paired t/sign test on per-fold diffs); this reuses
already-computed arrays at zero extra fits and is the standard remedy for stepwise selection on
CV (Hastie et al., ESL ch. 7 "1-SE" practice).
