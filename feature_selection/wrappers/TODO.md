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

### 3. Knockoffs (Barber & Candès 2015)

Generate a synthetic "knockoff" X_tilde_j with the same correlation structure
as X_j vs X_{-j} but independent of y. Importance = score with X_j swapped for
X_tilde_j.

Per-fit cost: Cholesky factorisation of X.T @ X (~p^3) plus the original fit.
Tractable up to p ~ 10k.

Reference: `knockpy` Python package.

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

## Low priority (cost-prohibitive on typical workloads)

### 8. Drop-column importance

O(p × full_fit_time) - infeasible on p ≥ 1000. Useful only as a ground-truth
oracle when bench-marking other importance methods.
