# RFECV - deferred ML improvements

ML-substantive enhancements not yet implemented. Items move OUT of this file the moment they ship; completed work lives in git history, not here.

> **2026-05-28 status:** Waves 1-5 of the audit overhaul landed (see USAGE.md), including L4 (SHAP-OOF alias) and L7 (in-tree native univariate-HT prescreen at `_univariate_ht.py`, no external deps). The high-priority items below remain deferred because they each require a non-trivial new code path / external dependency.

## Highest priority

### A. Auto-parameter tuning via synthetic benchmarks (planned)

After Wave 1-5 the RFECV constructor exposes ~25 ML-tuning knobs (rule, fi_missing_policy, fi_decay_rate, optimizer_target, init_design_size, dichotomic_epsilon, coef_scale_source, multiclass_coef_aggregation, cpi_max_depth, etc.). Each has a *defensible* default but no single default is universally best:

- `fi_missing_policy='worst'` vs `'median'` flips a ranking depending on FI ragged-pattern density.
- `n_features_selection_rule='one_se_max'` vs `'argmax'` flips N depending on whether `cv_mean_perf(N)` curve is flat-plateau or has a clear peak.
- `init_design_size` vs `max_refits` interact non-trivially on small budget (test_basic_regression LightGBM flake).

**Plan:** run a synthetic-benchmark sweep over:
- *Data distributions:* normal / heavy-tailed / categorical-encoded / mixed numeric+cat / high-cardinality
- *Target type:* binary / multiclass / regression with low / high noise
- *Feature redundancy:* independent / clusters / linear collinear / non-linear collinear
- *Information content:* high MI / weak MI / null target

For each (data-shape, knob-combo) measure recall@informative, n_features, time, stability. Fit a small classifier on (data-fingerprint -> best-knob-combo) and ship that as `RFECV(auto_tune=True)`:

```python
RFECV(estimator=..., auto_tune=True).fit(X, y)
# Inspects X (col-MI to y, col-col MI, dtype mix, redundancy) and picks the
# winning knob-combo from the calibrated table. No more manual rule-tuning.
```

Provenance: store the picked combo in `rfecv.auto_tune_decision_` for transparency.

This eliminates the "one-default-fits-all" myth. Until then, every config knob default is a calibrated guess based on the audit + Wave 5 bench. Document in USAGE.md the per-scenario recommendation.

## High priority

<!-- 1. mRMR pre-screening shipped 2026-05-28. Use prescreen='mrmr' on RFECV. -->
<!-- 2. Boruta integration shipped 2026-05-28. Use importance_getter='boruta' (Gini-based, in-tree) or 'boruta_shap' (TreeSHAP-based, optional dep). -->

### 3. Group LASSO via external estimator (still deferred)

For users with explicit group structure (one-hot expansions, correlated clusters), accept a `groupyr.GroupLasso` or `celer.GroupLasso` as the estimator and read coefficients per group. The current code path already handles linear estimators with `coef_`, so a Group-LASSO estimator passes through unchanged IF the user wants per-feature `coef_` interpretation. The remaining work is: (a) recognise group structure from the estimator's `groups_` attribute and aggregate coef per group for RFECV elimination; (b) expose `feature_groups` parity so the all-or-nothing decision matches the estimator's group regularisation. Until then, the existing `feature_groups` dict is the simpler workaround.


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

## Deferred literature items (2026-05-28 audit Wave 5)

### L4. SHAP-OOF elimination (`eliminate_by='shap_oof'`)

After each inner CV fold, score features by mean(|SHAP|) on the HELD-OUT fold (not the in-bag FI). Drops the next-eliminated feature by that signal instead of `feature_importances_`. Catches interaction effects that Gini misses. Requires a new elimination-strategy hook in the outer loop (currently the next-subset choice is downstream of `get_feature_importances`).

<!-- L7 native univariate-HT prescreen shipped 2026-05-28; see _univariate_ht.py and USAGE.md. -->

## Cross-selector contract gaps (surfaced 2026-05-28 by shared parametrized test suite)

The new `tests/feature_selection/test_fs_selector_contract.py` parametrises 19 contract tests across MRMR, RFECV, ShapProxiedFS. Two real gaps surfaced (both xfail / skip in the suite for now):

### B. MRMR does not validate NaN in y at fit entry

`MRMR.fit(X, y_with_nan)` lets NaN flow into the MI scorer silently. RFECV and ShapProxiedFS both reject upfront with ValueError. Fix: add the same `_isnan_mask(y).any()` guard at MRMR's `_coerce_target_dtype` to match the sibling selectors.

### C. ShapProxiedFS lacks `get_feature_names_out`

sklearn convention is that any TransformerMixin selector exposes `get_feature_names_out(input_features=None)`. ShapProxiedFS has `get_support` and `selected_features_` instead. Add a `get_feature_names_out` shim that just returns `selected_features_` to match the API surface.


## Low priority

### 4. Drop-column importance

`O(p * full_fit_time)` - infeasible on `p >= 1000`. Useful only as a ground-truth oracle when bench-marking other importance methods.

## Deferred audit items (no action required)

- **Mixed-scale columns:** implicitly handled. The `coef_` z-scoring on linear models already corrects scale-dependent importance; tree-based estimators are scale-invariant.
- **MRMR mixed-dtype NaN policy:** MRMR's silent ffill+bfill (`mrmr.py:471-473`) is intentional for the FE-screen pass and covered by the existing MRMR test suite. Users who want strict NaN handling can pre-impute with `sklearn.SimpleImputer` upstream.
- **Multi-output y with identical columns:** potential 3x weighting bug in MRMR's loop. Test gap, not a real-world issue.
- **polars int materialisation in `split_into_train_test`:** micro-optimization. Skip until a profile flags it.
- **Stochastic scoring:** documented as "scoring must be deterministic" in USAGE.md. No code knob.
