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

### 2. Conditional Permutation Importance (Strobl et al. 2008)

Vanilla `importance_getter='permutation'` (added in PR-3 N6) breaks on
correlated features: shuffling X_j independently while X_k stays creates
out-of-distribution combinations the model never saw, inflating measured
importance.

Conditional fix: for each feature j, fit a shallow tree X_{-j} -> X_j (or
classification if discrete), then permute X_j WITHIN each leaf of that tree.
Preserves P(X_j | X_{-j}).

Cost: 2-3x vanilla permutation (~90 min on n=1M, p=10k vs ~30 min for vanilla).
Worth it on highly-collinear feature sets.

Add as `importance_getter='conditional_permutation'`. Implementation: use
sklearn DecisionTreeRegressor / DecisionTreeClassifier per feature.

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

Add `swap_after_each_iter=False` (default) / `swap_top_k=5` parameters.

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
