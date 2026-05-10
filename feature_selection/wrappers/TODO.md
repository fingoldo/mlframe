# RFECV — substantive ML improvements deferred to future PRs

This file tracks ML-substantive enhancements that require non-trivial design or
external dependencies. Updated 2026-05-10 from user discussion of FI bias and
alternative selection algorithms.

## High priority

### 1. mRMR pre-screening for p >> n problems

`mlframe.feature_selection.filters.mrmr` already exists. RFECV on 50k features
without pre-screening doesn't converge in reasonable time. Add:

```python
RFECV(
    estimator=...,
    prescreen='mrmr',           # None | 'mrmr' | callable
    prescreen_top_k=3000,        # None = no cap; or int
    ...
)
```

Implementation: run mRMR.fit(X, y) at fit entry, restrict `original_features`
to the top-K it returns, then proceed with the regular RFECV loop.

Cost: O(p²) for mRMR vs O(p × iter) for backward elimination. Net win when
p ≥ 5000.

### 2. Conditional Permutation Importance (Strobl et al. 2008) — DONE in PR-12

`importance_getter='conditional_permutation'` is now wired through the
existing `get_feature_importances` dispatch alongside `'permutation'`.
For each feature j, a shallow `DecisionTreeClassifier`/`Regressor` is fit
on `X_{-j} -> X_j` (classifier if `X_j` has <=10 unique values or is
integer-typed, else regressor), then `X_j` is permuted WITHIN each leaf,
preserving `P(X_j | X_{-j})`. Default `n_repeats=5`, `max_depth=5`.

Tested in `test_wrappers_phase4_n3_n6.py::TestPhase7_ConditionalPermutationImportance`
(8 tests: shape contract, correlated-pair structural claim, ndarray /
DataFrame inputs, regression, single-feature edge case, constant-feature
edge case, end-to-end via RFECV).

Cost: ~2-3x vanilla permutation (per-feature shallow tree fit + leaf-grouped
shuffles).

### 2b. SHAP IS biased toward high-cardinality features (correction to PR-3 doc)

Earlier docs incorrectly claimed `importance_getter='shap'` is bias-free
for high-cardinality features. This is **not true**: SHAP TreeExplainer
follows the SAME tree paths used by Gini/gain, so it inherits the same
bias toward features with more unique values (more split points = more
opportunities to randomly improve impurity).

References:
- Strobl, Boulesteix, Zeileis, Hothorn 2007, "Bias in random forest variable
  importance measures: Illustrations, sources and a solution".
- Sutera et al. 2021 - SHAP-based importance measures inherit the same
  biases as the underlying tree method.

For truly high-card-bias-free FI, use:
- `importance_getter='permutation'` (vanilla, Breiman 2001) - unbiased for
  cardinality but breaks on correlated features (use 'conditional_permutation'
  in that case).
- `importance_getter='knockoff'` (PR-5) - Gaussian knockoffs (Barber-Candes
  2015), no high-card bias AND robust to correlations.

`importance_getter='shap'` remains useful for:
- per-prediction local explanations (its actual purpose)
- global summaries when feature cardinality is comparable
- tree-based models when speed matters (faster than permutation)

But it is NOT a replacement for permutation/knockoffs on high-card data.

### 3. Knockoffs (Barber & Candès 2015) - DONE in PR-5/PR-6/PR-7

PR-5 / PR-6 / PR-7 status:
- `make_gaussian_knockoffs` - equicorrelated Gaussian knockoffs (PR-5)
- `knockoff_importance` - W-statistic per feature (PR-5)
- `select_features_fdr(W, q)` - Barber-Candes FDR-controlled selection (PR-7)

DEFERRED enhancements (RECOMMEND SKIP unless specific need surfaces):

**SDP-optimised s** (Barber-Candes' "SDP knockoffs"). RECOMMENDATION: SKIP.

- Power gain over equicorrelated: 5-15% (Barber-Candes 2015 Table 2)
- Cost: hard dependency on `cvxpy` (~50MB transitive deps)
- On the project's bench (n=8000, p=200) equicorrelated already gives
  knockoffs(LR, BC q=0.5) recall=0.86 in 0.54s
- Marginal gain not worth dependency cost when stability+multi sits at
  recall=0.86 too. If a future workload genuinely needs the extra power,
  re-evaluate.

Currently `s` is set equicorrelated (`s_j = s for all j`,
`s = min(2*lam_min(Sigma), 1)`). The SDP-based optimal `s_j` per feature
gives tighter knockoffs (each X_tilde_j is closer to "independent of X_j
given X_{-j}" vs the equicorrelated ceiling). Implementation if needed:
```python
def make_sdp_gaussian_knockoffs(X, ...):
    import cvxpy as cp
    Sigma = corr(X)
    s = cp.Variable(p, nonneg=True)
    objective = cp.Maximize(cp.sum(s))
    constraints = [s <= 1, 2*Sigma - cp.diag(s) >> 0]
    cp.Problem(objective, constraints).solve()
    return _construct_knockoffs(X, Sigma, s.value)
```

**Model-X knockoffs** (Candes, Fan, Janson, Lv 2018). RECOMMENDATION: SKIP.

- Useful when X is sharply non-Gaussian (heavy-tailed, mixed types). On
  standard tabular numeric+categorical with z-score normalisation,
  Gaussian knockoffs hold up well in practice.
- Cost: hard dependency on `knockpy` -> torch + lightning + huge stack.
  Not worth bringing into core mlframe.
- Workaround for users who genuinely need Model-X: import `knockpy`
  separately, build `X_tilde` via `knockpy.KnockoffSampler`, pass to
  `knockoff_importance(...)` via the existing `model_factory` path
  (works without code changes).

Drops the Gaussian assumption. Uses any conditional distribution
estimator for `P(X_j | X_{-j})`.

**Inferred-Sigma fallback** for n < p: when n is too small to reliably
estimate `Sigma` (rule of thumb: n < 5p), use shrinkage covariance
(`sklearn.covariance.LedoitWolf`) instead of empirical correlation.

`mlframe.feature_selection.wrappers.make_gaussian_knockoffs` and
`knockoff_importance` now provide equicorrelated Gaussian knockoffs +
W-statistic computation. Bias-free for high-cardinality and correlation-
robust. Tested on synthetic data (12 features, 4 informative, class_sep=2.5):
informative features get W >> noise W.

Future enhancements (not in PR-5):
- **SDP-optimised s** (Barber-Candes' "SDP knockoffs") for tighter
  knockoff diagonal. Requires cvxpy. Currently equicorrelated.
- **Model-X knockoffs** (Candes, Fan, Janson, Lv 2018) - drop the
  Gaussian assumption; use any conditional distribution estimator.
- **FDR-controlled selection** at level q: pick features with W_j >= tau
  where tau = min{t : (1 + #{j: W_j <= -t}) / max(1, #{j: W_j >= t}) <= q}.
  Provable FDR <= q. Currently the user can compute this themselves from
  the W dict; could be a built-in `select_features_fdr(W, q=0.1)` helper.

### 4. Stability Selection (Meinshausen & Bühlmann 2010)

Replace per-fold CV voting with bootstrap voting. Strong theoretical guarantee
on family-wise error rate. Implemented in PR-4 (this commit).

## Medium priority

### 5. Truncated SFFS swap-after-elimination

Classical SFFS is O(p) extra fits per iter, infeasible for p ≥ 1000. Truncated:
only swap among top-5 dropped and top-5 selected.

Cost: ~6K extra fits on a 50-iter run = ~60s overhead at 10ms/fit. Tractable
as opt-in flag.

DONE in PR-13 as a final-pass (not per-iter) variant: `swap_top_k: int = 0`
runs K paired swaps on the BEST subset after the main MBH loop converges.
For each of the K worst-FI kept features, try replacing it with one of the
K best-FI dropped features and accept any swap that strictly improves the
CV score. Cost: O(K) extra `cross_val_score` calls at the end of the run
only (vs O(K * iter_count) for per-iter). Uses `sklearn.cross_val_score`
directly so swap evals do NOT honour `fit_params` / `val_cv` / early
stopping; documented in USAGE.md. Tested in `test_wrappers_phase8.py
::TestSffsSwap` (4 tests: opt-out default, opt-in fires, monotone-or-equal
to non-swap baseline, regression smoke).

### 6. Boruta integration

`mlframe.boruta_shap` exists. Add `importance_getter='boruta'` as a string
value in `get_feature_importances`. Boruta gives FWER-controlled selection
(adds shadow features = shuffled copies, drops feature only if FI < max-shadow
FI with statistical test).

### 7. Group LASSO via external estimator

For users with explicit group structure (one-hot expansions, correlated
clusters), accept a `groupyr.GroupLasso` or `celer.GroupLasso` as the
estimator and read coefficients per group. Alternative to the simpler
`feature_groups` dict already implemented in PR-4.

## P1/P2 audit findings still deferred

### B8: mixed-scale columns
**Status**: implicitly handled. The `coef_` z-scoring fix (PR-4 tactical) already corrects scale-dependent importance for linear models. Tree-based estimators are scale-invariant. No additional fix required; documented as "z-scoring active by default".

### H36: MRMR mixed-dtype NaN policy
**Status**: documented. MRMR's silent ffill+bfill (mrmr.py:471-473) is intentional for the FE-screen pass and tested by the existing MRMR test suite. If a user wants strict NaN handling, they can pre-impute with sklearn's `SimpleImputer` upstream. No core change needed.

### Other deferred (P2 cosmetic)
- **A4**: multi-output y with all columns identical - potential 3x weighting bug in MRMR's loop. Test gap, not a real-world issue.
- **B30**: polars int materialisation perf - micro-optimization in split_into_train_test polars branch. Skip until profile shows it.
- **F2**: `prev_score = -np.inf` init - already correct (PR-1 F21 fix).
- **F35**: `selected_features_per_nfeatures` keyed by length only - PR-1 F35 fix gates writes on score improvement. Re-keyed approach was over-engineered.
- **E24**: stochastic scoring - documented as "scoring must be deterministic" in USAGE.md. No code knob.
- **B30 polars int materialisation**: skip per perf-measure-first rule.

## Low priority (cost-prohibitive on typical workloads)

### 8. Drop-column importance

O(p × full_fit_time) - infeasible on p ≥ 1000. Useful only as a ground-truth
oracle when bench-marking other importance methods.

## PR-12 status

PR-12 closed three high-priority items:

1. **Conditional Permutation Importance (TODO #2)** — wired through
   `importance_getter='conditional_permutation'`; see updated #2 above.

2. **Resume-from-checkpoint (new)** — `RFECV(checkpoint_path=...)` pickles
   the outer-loop mutable state (counters, evaluated_scores_*, optimizer,
   best-so-far) atomically (tmpfile + os.replace) after every iter. A
   fit() that finds a matching-signature checkpoint at the path resumes
   from saved nsteps instead of restarting at 0. Long runs (10k feat *
   100 iter) are no longer unrecoverable on crash. The fitted_estimators
   dict is intentionally NOT persisted (would dominate file size on CB
   / RF ensembles); it is rebuilt on demand by the final voting / refit
   path. Tested in `test_wrappers_checkpoint.py` (12 tests covering
   opt-in, save shape, signature match / mismatch, version mismatch,
   corrupt file, atomic-write integrity, no stale tempfiles, end-to-end
   resume).

3. **cProfile pass on post-refactor code** — re-ran
   `_benchmarks/profile_rfecv.py` after the Phase 3 module split (the
   prior profile referenced the now-deleted monolithic `wrappers.py`).
   PR-13 also profiled all PR-12/PR-13 newly-added features via
   `_benchmarks/profile_new_features.py` (results in
   `_benchmarks/_results/profile_pr1213_*.txt`):

   | New feature | Wall time on bench | Hot-spot | Action |
   |-------------|--------------------|----------|--------|
   | CPI (RF, n=600, p=40) | 9.77s | 87% in user's `model.score()` (RF predict_proba); CPI's Python orchestration ~5% | none — bottleneck is C-compiled tree ensemble predict |
   | CPI (Ridge, n=400, p=30) | 0.98s | 67% in `model.score()`, 18% per-feature shallow-tree fit | none — feature-wise loop trivially parallelisable but conflicts with the existing per-fold N3 joblib layer; cost-vs-complexity not warranted |
   | SFFS swap (LR, n=400, p=20, K=5) | sub-100ms within 6.2s total | not a top-30 hotspot (K=5 swap evals = ~15 LR fits) | none — overhead negligible vs main MBH loop |
   | __sklearn_tags__ | one-time call | µs | trivial |
   | cv auto-detect | one-time check | µs | trivial |
   | cv_results_df_ property | on access | ms | trivial |
   | resume-from-checkpoint pickle | per outer iter | bounded by IO | not actionable |

   **Conclusion: no actionable @njit / numba candidate in any
   PR-12/PR-13 feature.** All Python overhead in CPI is dwarfed by the
   user's compiled `model.score()`; SFFS swap is sub-noise. Documented
   per `feedback_perf_measure_first` so the same flag isn't re-raised.
   Outcome: **no actionable @njit / numba candidate remains.** Fresh
   profile_*.txt files saved to `_benchmarks/_results/` show that on
   the medium (n=600, p=80) and large (n=1000, p=200) configurations
   80%+ of wall-clock is inside the user's estimator (`core.py:_train`
   at ~73-81%, native compiled C++ for CatBoost; sklearn's compiled
   scipy lbfgsb path for LR). The previous big offender
   (`get_actual_features_ranking` at 5.1s on monolithic baseline) has
   collapsed to 0.118s thanks to the lazy-Leaderboard fix in PR-3.
   `clean_ram` (`gc.collect`) at ~8.5% is already gated to every 5th
   iter; further gating trades latency for memory pressure on long runs
   and was rejected per `feedback_no_tradeoff_optimizations`. Conclusion
   recorded here per `feedback_perf_measure_first` so the same flag
   isn't re-raised in future audits.
