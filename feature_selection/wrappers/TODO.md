# RFECV - deferred ML improvements

ML-substantive enhancements not yet implemented. Items move OUT of this file the moment they ship; completed work lives in git history, not here.

## High priority

### 1. mRMR pre-screening for p >> n problems

`mlframe.feature_selection.filters.mrmr` already exists. RFECV on 50k features without pre-screening doesn't converge in reasonable time. Add:

```python
RFECV(
    estimator=...,
    prescreen='mrmr',           # None | 'mrmr' | callable
    prescreen_top_k=3000,        # None = no cap; or int
    ...
)
```

Implementation: run `mrmr.fit(X, y)` at fit entry, restrict `original_features` to the top-K it returns, then proceed with the regular RFECV loop. Cost: `O(p^2)` for mRMR vs `O(p * iter)` for backward elimination - net win when `p >= 5000`.

### 2. Boruta integration

`mlframe.boruta_shap` exists. Add `importance_getter='boruta'` as a string value in `get_feature_importances`. Boruta gives FWER-controlled selection (adds shadow features = shuffled copies, drops a feature only if its FI is below the max-shadow FI under a statistical test).

### 3. Group LASSO via external estimator

For users with explicit group structure (one-hot expansions, correlated clusters), accept a `groupyr.GroupLasso` or `celer.GroupLasso` as the estimator and read coefficients per group. Alternative to the simpler `feature_groups` dict already available.

## Knockoff enhancements (RECOMMEND SKIP unless a workload demands it)

The default Gaussian knockoffs (`make_gaussian_knockoffs`) use equicorrelated `s_j = s` with `s = min(2 * lam_min(Sigma), 1)`. Two tighter constructions exist but are not worth the dependency cost on current workloads.

### SDP-optimised s

Per-feature optimal `s_j` solved via SDP. Power gain over equicorrelated is 5-15% (Barber-Candes 2015 Table 2). Cost: hard dependency on `cvxpy` (~50MB transitive). On the project bench (n=8000, p=200) equicorrelated already hits recall=0.86 in 0.54s; stability+multi sits at the same recall - the marginal gain doesn't justify the dep. Sketch if needed:

```python
def make_sdp_gaussian_knockoffs(X, ...):
    import cvxpy as cp
    Sigma = corr(X)
    s = cp.Variable(p, nonneg=True)
    objective = cp.Maximize(cp.sum(s))
    constraints = [s <= 1, 2 * Sigma - cp.diag(s) >> 0]
    cp.Problem(objective, constraints).solve()
    return _construct_knockoffs(X, Sigma, s.value)
```

### Model-X knockoffs (Candes, Fan, Janson, Lv 2018)

Drops the Gaussian assumption; uses any conditional distribution estimator for `P(X_j | X_{-j})`. Useful when X is sharply non-Gaussian (heavy-tailed, mixed types). Cost: hard dependency on `knockpy` -> torch + lightning + huge stack. Users who genuinely need Model-X can import `knockpy` separately, build `X_tilde` via `knockpy.KnockoffSampler`, and pass it to `knockoff_importance(...)` via the existing `model_factory` path (works without changes here).

### Inferred-Sigma fallback for n < p

When n is too small to reliably estimate `Sigma` (rule of thumb: n < 5p), substitute shrinkage covariance (`sklearn.covariance.LedoitWolf`) for the empirical correlation.

## Low priority

### 4. Drop-column importance

`O(p * full_fit_time)` - infeasible on `p >= 1000`. Useful only as a ground-truth oracle when bench-marking other importance methods.

## Deferred audit items (no action required)

- **Mixed-scale columns:** implicitly handled. The `coef_` z-scoring on linear models already corrects scale-dependent importance; tree-based estimators are scale-invariant.
- **MRMR mixed-dtype NaN policy:** MRMR's silent ffill+bfill (`mrmr.py:471-473`) is intentional for the FE-screen pass and covered by the existing MRMR test suite. Users who want strict NaN handling can pre-impute with `sklearn.SimpleImputer` upstream.
- **Multi-output y with identical columns:** potential 3x weighting bug in MRMR's loop. Test gap, not a real-world issue.
- **polars int materialisation in `split_into_train_test`:** micro-optimization. Skip until a profile flags it.
- **Stochastic scoring:** documented as "scoring must be deterministic" in USAGE.md. No code knob.
