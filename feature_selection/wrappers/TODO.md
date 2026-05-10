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

DEFERRED enhancements:

**SDP-optimised s** (Barber-Candes' "SDP knockoffs"). Currently `s` is set
equicorrelated (`s_j = s for all j`, `s = min(2*lam_min(Sigma), 1)`). The
SDP-based optimal `s_j` per feature gives tighter knockoffs (each X_tilde_j
is closer to "independent of X_j given X_{-j}" vs the equicorrelated
ceiling). Cost: requires `cvxpy` dependency. Implementation:
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

**Model-X knockoffs** (Candes, Fan, Janson, Lv 2018). Drops the Gaussian
assumption. Uses any conditional distribution estimator (autoencoder,
deep learning model) for `P(X_j | X_{-j})`. Library: `knockpy`. Useful
when X is non-Gaussian (e.g. heavy-tailed, mixed types). Implementation
would be a thin wrapper around `knockpy.KnockoffSampler`.

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

## Low priority (cost-prohibitive on typical workloads)

### 8. Drop-column importance

O(p × full_fit_time) - infeasible on p ≥ 1000. Useful only as a ground-truth
oracle when bench-marking other importance methods.
