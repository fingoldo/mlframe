# Composite Targets — feature guide

A composite target is a transform `T = f(y, base)` such that the model learns
`T` from features `X` (typically *excluding* the dominant feature `base`), and
a wrapper applies `f⁻¹` at predict time to recover `y` on the original scale.

The structural example: `y = target`, `base = lag_feature`. The autoregressive
lag is captured natively by the transform (e.g. `diff`: `T = y − base`), and the
model is forced to explain only the *remaining residual*. This concentrates the
learner's capacity on the part of the signal that is actually hard, removes a
feature that would otherwise dominate splits / coefficients, and lets the inverse
re-attach the easy base term exactly.

Every public symbol referenced below is importable from
`mlframe.training.composite`.

```python
from mlframe.training.composite import (
    CompositeTargetEstimator, CompositeClassificationEstimator,
    CompositeGLMEstimator, CompositeQuantileEstimator,
    CompositeMultiOutputEstimator, make_per_column_specs,
    CompositeTargetDiscovery, CompositeTargetDiscoveryConfig,
    make_composite_regressor, CompositeTargetTransformer,
    list_transforms, get_transform, conformal_quantile, report_to_markdown,
)
```

---

## 1. `CompositeTargetEstimator` basics

The wrapper holds a `base_estimator` (any sklearn-style regressor), a
`transform_name` (one of `list_transforms()`), and the `base_column` whose values
feed the forward / inverse. It clones the inner at `fit`, trains it on the
`T`-scale target, and inverts at `predict` back to the y-scale.

```python
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from mlframe.training.composite import CompositeTargetEstimator

rng = np.random.default_rng(0)
n = 2000
lag = rng.normal(size=n)
resid = 0.3 * rng.normal(size=n)
X = pd.DataFrame({"lag": lag, "f1": rng.normal(size=n), "f2": rng.normal(size=n)})
y = lag + resid                      # y is mostly its own lag plus noise

est = CompositeTargetEstimator(
    base_estimator=HistGradientBoostingRegressor(max_iter=100),
    transform_name="diff",           # T = y - base
    base_column="lag",
)
est.fit(X, y)
yhat = est.predict(X)                # y-scale prediction (inner predicts T, then inverts)
```

Key constructor knobs (see the class docstring for full semantics):

- `fallback_predict` — per-row inverse-failure strategy: `"y_train_median"`
  (default) or `"nan"`.
- `drop_invalid_rows` — drop fit-time domain-violating rows (default `True`) vs
  raise `DomainViolationError`.
- `base_columns` — multi-column base path (`linear_residual_multi`); overrides
  `base_column`.
- `group_column` — group labels for grouped transforms
  (`linear_residual_grouped`); dropped from `X` before the inner sees it.
- `auto_variance_stabilise` — for `ratio` / `logratio`, weight ∝ 1/|base| to
  flatten multiplicative heteroscedasticity (default off; changes the loss).
- `recurrence_continuation` — for the left-recurrent transforms
  (`ewma_residual` / `frac_diff` / `rolling_quantile_ratio`) seed the predict-time
  inverse from the train-TAIL state so a predict batch that *continues* the
  training series is not biased on its first ~k rows (default off — predict stays
  stateless).
- `monotone_constraints` — per-feature ±1/0 vector forwarded to the inner GBDT,
  enforced on the **T (residual) scale** (additive-residual inverses carry the
  monotonicity through to y; non-additive inverses preserve the sign).

Fitted state lives in `est.fitted_params_` (alpha/beta, y-clip + T-clip bounds,
`y_train_median`), `est.estimator_` (the fitted inner), `est.feature_names_in_`,
and `est.runtime_stats_` (live domain-violation / clip-hit counters).

### Post-hoc wrapping of an already-fitted inner

```python
inner = HistGradientBoostingRegressor().fit(X, y - X["lag"].to_numpy())  # trained on T
wrapped = CompositeTargetEstimator.from_fitted_inner(
    fitted_inner=inner, transform_name="diff", base_column="lag",
    transform_fitted_params={}, y_train=y,
)
wrapped.predict(X)
```

---

## 2. Discovery — find the best `(base, transform)` automatically

`CompositeTargetDiscovery` ranks base candidates by residualised-MI, screens the
whole transform registry, optionally reranks with a tiny model, runs multi-base
forward-stepwise auto-promotion, gates on validation, and emits a
`CompositeProvenance`.

```python
from mlframe.training.composite import (
    CompositeTargetDiscovery, CompositeTargetDiscoveryConfig,
)

cfg = CompositeTargetDiscoveryConfig(enabled=True)
disc = CompositeTargetDiscovery(cfg)

train_idx = np.arange(0, 1500)
val_idx = np.arange(1500, 2000)
disc.fit(
    df=X.assign(target=y), target_col="target",
    feature_cols=["lag", "f1", "f2"],
    train_idx=train_idx, val_idx=val_idx,
)
specs = disc.export_specs()          # list of dict specs (best (base, transform) pairs)
rows = disc.report()                 # per-candidate audit rows (kept / rejected + reason)
```

`fit` requires an explicit `train_idx` (no implicit "use full df"); `val_idx` /
`test_idx` are stored for integrity checks and never touched during fit.

### `suggest_discovery_config` — data-driven config

Inspect the frame cheaply (bounded `sample_n`-row slice, never copies the whole
frame) and get a populated config plus a per-field rationale.

```python
from mlframe.training.composite.autoconfig import suggest_discovery_config

cfg, rationale = suggest_discovery_config(
    df=X.assign(target=y), target_col="target",
    feature_cols=["lag", "f1", "f2"], sample_n=20_000, seed=0,
)
for field, why in rationale.items():
    print(field, "->", why)
disc = CompositeTargetDiscovery(cfg)
```

### Time column / M6 / time-series transforms

Pass `time_ordering` (per-row sortable key) so the MI-screening sample is sorted
by time and the tiny-model CV uses a forward-walk (TimeSeriesSplit) instead of a
shuffled K-fold — the canonical `lag(y)` base is non-monotone, so the screen must
not leak future→past on temporal data.

```python
from mlframe.training.composite import (
    detect_time_column_candidates, sort_df_by_time_column,
)

tcols = detect_time_column_candidates(X.assign(target=y))   # ranked time-col guesses
df_sorted = sort_df_by_time_column(X.assign(target=y), tcols[0]) if tcols else X.assign(target=y)
disc.fit(
    df=df_sorted, target_col="target", feature_cols=["lag", "f1", "f2"],
    train_idx=train_idx, val_idx=val_idx,
    time_ordering=df_sorted.index.to_numpy(),
)
```

The time-series transforms (`ewma_residual`, `frac_diff`,
`rolling_quantile_ratio`) are recurrent: each output row depends on its
neighbours in row order. Combine with `recurrence_continuation=True` on the
estimator when the predict batch continues the training series.

---

## 3. Transform catalogue

`list_transforms()` returns the full registry; `get_transform(name)` returns the
frozen `Transform` (forward / inverse / fit / domain_check + `requires_base` /
`requires_groups` / `recurrent` flags + the provenance formula).

```python
from mlframe.training.composite import list_transforms, get_transform

print(sorted(list_transforms()))
t = get_transform("linear_residual")
print(t.requires_base, t.formula)    # provenance formula string
```

Families (non-exhaustive):

| Group | Transforms |
|---|---|
| Additive residual | `diff`, `additive_residual`, `linear_residual`, `linear_residual_robust`, `linear_residual_multi`, `linear_residual_grouped`, `theilsen_residual`, `median_residual`, `polynomial_residual_deg2`, `pairwise_interaction_residual`, `reciprocal_residual`, `geometric_mean_residual` |
| Ratio / multiplicative | `ratio`, `centered_ratio`, `logratio` |
| Unary y-transform (base-free) | `log_y`, `cbrt_y`, `signed_power_y`, `yeo_johnson_y`, `quantile_normal_y`, `y_quantile_clip` |
| Rank / quantile residual | `rank_residual`, `quantile_residual`, `monotonic_residual`, `smoothing_spline_residual`, `asinh_residual` |
| Categorical-base residual | `target_encoding_residual` |
| Time-series / recurrent | `ewma_residual`, `frac_diff`, `rolling_quantile_ratio` |
| Chained | `chain_linres_cbrt`, `chain_linres_cbrt_qn`, `chain_linres_yj`, `chain_monres_cbrt`, `chain_monres_yj` |

Highlighted entries from the task surface:

- **`signed_power_y`** — unary `T = sign(y)·|y|^p` (fitted power), recovers
  heavy-tailed targets symmetric about 0; base-free, named `y-signedPowY`.
- **`target_encoding_residual`** — residual over an OOF target-mean encoding of a
  categorical base column; the encoding is fit train-only to avoid leakage.
- **`theilsen_residual`** — robust additive residual using a Theil–Sen slope
  (median-of-pairwise-slopes), resistant to outliers in `base`.
- **`ewma_residual`** — residual over an exponentially-weighted moving average of
  past `y` (`_EWMA_RESIDUAL_DEFAULT_K`); recurrent.
- **`frac_diff`** — fractional differencing (`_FRAC_DIFF_DEFAULT_D`,
  `_FRAC_DIFF_DEFAULT_LAGS`); recurrent, stationarises while preserving memory.

`ewma_residual` / `frac_diff` / `rolling_quantile_ratio` pair with
`recurrence_continuation=True` so the predict-time inverse continues the trained
recurrence instead of restarting from the train mean.

---

## 4. Uncertainty — conformal intervals & quantiles

### Split-conformal (constant width)

```python
est.fit(X.iloc[train_idx], y[train_idx])
est.calibrate_conformal(X.iloc[val_idx], y[val_idx], alpha=0.1)   # held-out rows only
lo, hi = est.predict_interval(X, alpha=0.1)                       # marginal coverage >= 0.9
```

`X_cal` / `y_cal` **must** be rows the inner never trained on (the suite val split
or an OOF fold) — conformal validity rests on exchangeability with the calibration
set.

### CQR — adaptive width (heteroscedastic)

Conformalized Quantile Regression widens / narrows the band per row using the
wrapper's quantile predictions:

```python
est.calibrate_conformal_cqr(X.iloc[val_idx], y[val_idx], alpha=0.1)
lo, hi = est.predict_interval_cqr(X, alpha=0.1)
```

### Mondrian — group-conditional

A separate finite-sample radius per group, for conditional coverage ≥ 1−alpha
within each group (global-radius fallback for unseen / too-small groups):

```python
groups_cal = X["f1"].iloc[val_idx].round().to_numpy()
groups_all = X["f1"].round().to_numpy()
est.calibrate_conformal_mondrian(X.iloc[val_idx], y[val_idx], groups_cal, alpha=0.1)
lo, hi = est.predict_interval_mondrian(X, groups_all, alpha=0.1)
```

### `predict_quantile`

```python
q = est.predict_quantile(X, alpha=[0.1, 0.5, 0.9])   # (n, 3) y-scale quantiles
```

### `CompositeQuantileEstimator` — native pinball heads

One inner per quantile, each fit on `T` with the pinball loss and inverted to
y-scale; non-crossing enforced per row.

```python
from sklearn.ensemble import GradientBoostingRegressor
from mlframe.training.composite import CompositeQuantileEstimator

qest = CompositeQuantileEstimator(
    base_estimator=GradientBoostingRegressor(loss="quantile"),
    transform_name="linear_residual", base_column="lag",
    quantiles=[0.1, 0.5, 0.9], enforce_non_crossing=True,
)
qest.fit(X, y)
bands = qest.predict_quantile(X)     # (n, 3), ascending, non-crossing
mid = qest.predict(X)                # the 0.5 head, y-scale
```

Blend several fitted multi-quantile members per-quantile-column with
`predict_quantile_ensemble(members, X, quantiles)`.

---

## 5. Classification — `CompositeClassificationEstimator`

Models a base-margin (log-odds) residual: a base classifier supplies a margin,
the inner learns the correction, and `predict_proba` recombines. Binary and
multiclass; includes a `calibration_report`.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from mlframe.training.composite import CompositeClassificationEstimator

yc = (y > np.median(y)).astype(int)
clf = CompositeClassificationEstimator(
    base_estimator=HistGradientBoostingClassifier(),
    base_margin_estimator=LogisticRegression(max_iter=500),
)
clf.fit(X, yc)
proba = clf.predict_proba(X)
labels = clf.predict(X)
rep = clf.calibration_report(X, yc, n_bins=10)   # reliability / ECE-style summary
```

For multiclass the base margin is `(n, K)` (softmax is shift-invariant, so
`log(proba)` is a valid margin when `decision_function` is absent).

---

## 6. GLM — `CompositeGLMEstimator`

Log-link Poisson / Gamma / Tweedie residual over a positive base mean — counts,
durations, insurance-style targets.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from mlframe.training.composite import CompositeGLMEstimator

ycount = np.random.default_rng(1).poisson(np.exp(0.5 * X["lag"].to_numpy()))
glm = CompositeGLMEstimator(
    base_estimator=HistGradientBoostingRegressor(),
    family="poisson",            # "poisson" | "gamma" | "tweedie"
    tweedie_power=1.5,
)
glm.fit(X, ycount)
mu = glm.predict(X)              # positive mean on the original scale
```

When `base_mean_estimator` is omitted a `PoissonRegressor` / `GammaRegressor`
default supplies the base mean.

---

## 7. Multi-output

One `CompositeTargetEstimator` per column of a vector target `(n, K)`, each with
its own transform + base; `predict` returns `(n, K)`.

```python
from mlframe.training.composite import (
    CompositeMultiOutputEstimator, make_per_column_specs,
)

Y = np.column_stack([y, y * 2 + rng.normal(size=n)])     # (n, 2)
specs = make_per_column_specs(
    n_outputs=2,
    shared_spec={"transform_name": "diff"},
    base_columns_map={0: "lag", 1: "lag"},
)
mo = CompositeMultiOutputEstimator(
    base_estimator=HistGradientBoostingRegressor(),
    column_specs=specs, skip_failed_columns=True,
)
mo.fit(X, Y)
preds = mo.predict(X)            # (n, 2)
```

---

## 8. sklearn `Pipeline` / `TransformedTargetRegressor` integration

`make_composite_regressor` is a thin factory returning a ready estimator usable
as a `Pipeline` final step or a grid-search `estimator`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mlframe.training.composite import make_composite_regressor

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", make_composite_regressor(
        HistGradientBoostingRegressor(), transform_name="diff", base_column="lag",
        monotone_constraints=None,
    )),
])
pipe.fit(X, y)
pipe.predict(X)
```

`CompositeTargetTransformer` exposes the forward / inverse as a sklearn
`TransformerMixin` for use inside `sklearn.compose.TransformedTargetRegressor`:

```python
from sklearn.compose import TransformedTargetRegressor
from mlframe.training.composite import CompositeTargetTransformer

ttr = TransformedTargetRegressor(
    regressor=HistGradientBoostingRegressor(),
    transformer=CompositeTargetTransformer(transform_name="cbrt_y"),  # base-free unary
)
ttr.fit(X[["f1", "f2"]], y)
ttr.predict(X[["f1", "f2"]])
```

(Base-dependent transformers need the base aligned at transform time; the unary
transforms — `cbrt_y` / `log_y` / `signed_power_y` / … — are the drop-in case for
`TransformedTargetRegressor`.)

---

## 9. Diagnostics plots

`mlframe.training.composite.diagnostics` ships matplotlib helpers (lazy pyplot
import — no hard matplotlib dependency at import time):

```python
from mlframe.training.composite import diagnostics

diagnostics.plot_target_distribution(y)
diagnostics.plot_qq(y)
diagnostics.plot_linear_fit(X["lag"].to_numpy(), y)
diagnostics.plot_predictions_vs_actual(y, est.predict(X))
diagnostics.plot_reliability_diagram(yc, clf.predict_proba(X)[:, 1])
diagnostics.plot_interval_coverage(y, lo, hi)
diagnostics.plot_interval_width_vs_x(X["lag"].to_numpy(), lo, hi)
diagnostics.plot_mi_gain_with_ci(rows)            # discovery audit rows
diagnostics.plot_alpha_stability(disc)
```

---

## 10. Monotonic constraints

Pass a per-feature `±1 / 0` vector; it is forwarded to the inner GBDT and
enforced on the **T (residual) scale**. For additive-residual transforms the
inverse adds a base-only term back, so monotonicity in T carries through to y at
fixed base. The vector length must match the **post-drop feature count** (columns
the inner trains on, i.e. wrapper columns minus any plumbing column dropped before
fit).

```python
est_mono = CompositeTargetEstimator(
    base_estimator=HistGradientBoostingRegressor(),
    transform_name="diff", base_column="lag",
    monotone_constraints=[+1, 0, 0],      # f-order: lag (+), f1, f2
)
est_mono.fit(X, y)
```

---

## 11. Provenance & reporting

Discovery emits a `CompositeProvenance` audit trail; `report_to_markdown` renders
a stakeholder-ready report (summary, discovered-specs table, metrics matrix,
decision trail, per-spec audit, rejected candidates, ensemble metadata).

```python
from mlframe.training.composite import report_to_markdown
from mlframe.training.composite.spec import CompositeSpec

md = report_to_markdown(
    target_col="target",
    specs=disc.specs_,                 # list[CompositeSpec]
    failures=disc.filter_drops(),      # rejected candidates + reason
)
print(md)
```

`CompositeSpec` is the frozen, picklable record (transform name, base column(s),
fitted params, MI numbers) that drives the suite-level post-hoc wrapping via
`CompositeTargetEstimator.from_fitted_inner`.

---

## See also

- `docs/composite_config_reference.md` — every `CompositeTargetDiscoveryConfig` field.
- `docs/composite_targets_tutorial.ipynb` — end-to-end notebook.
- `docs/calibration_policy.md` — conformal / calibration policy.
- `docs/MULTI_OUTPUT.md`, `docs/MULTI_TARGET_REGRESSION.md` — vector-target depth.

## Advanced discovery: opt-in steps (default ON)

Three discovery steps with measured value are now **enabled by default** (each
compute-bounded by its caps; set the flag `False` for the fastest fit):

- **Auto transform-chaining** (`auto_chain_discovery_enabled=True`): composes
  `residual → tail-compression` chains and *appends to `specs_`* only the chains
  that beat both single stages on held-out RMSE (empty when none win). The
  winning chain is selected like any other spec, so you get it automatically.
- **Region-adaptive transform selection** (`region_adaptive_enabled=True`): picks
  the best transform per quantile region of the base; surfaces on
  `discovery.region_adaptive_specs_` (an informational artefact — it does not
  alter `specs_`).
- **Interaction-base discovery** (`interaction_base_discovery_enabled=True`):
  surfaces `a OP b` interaction bases whose synergy MI beats both marginals on
  `discovery.interaction_bases_`.

```python
disc = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig(enabled=True)).fit(df, "y", feats, train_idx)
disc.specs_                     # includes any winning auto-chain spec
disc.region_adaptive_specs_     # per-region best-transform artefacts
disc.interaction_bases_         # interaction-base candidates + synergy scores
```

Other opt-in discovery signals: `transform_waic_validation_enabled` (WAIC/LOO
ranking signal alongside MI-gain), `auto_base_structural_boost` (near-affine /
low-card / monotone base hints, default ON). `suggest_discovery_config(df, y, feats)`
picks sensible defaults from the data; `discover_and_wrap(...)` runs the whole
flow and returns a fitted estimator + report. `discover_incremental(prior, new_df)`
warm-starts on appended data.

## Uncertainty quantification at a glance

| Need | Use |
|------|-----|
| Constant-width band, any model | `calibrate_conformal` → `predict_interval` |
| Adaptive width (heteroscedastic) | `calibrate_conformal_cqr` → `predict_interval_cqr` |
| Per-group coverage | `calibrate_conformal_mondrian` → `predict_interval_mondrian` |
| Covariate shift | `calibrate_conformal_weighted` → `predict_interval_weighted` |
| Full quantiles | `CompositeQuantileEstimator.predict_quantile` |

---

# Extended surface (post-wave-10)

Everything below is importable from `mlframe.training.composite` unless a
fully-qualified submodule path is shown. Each section gives: one-line summary,
when to use, and a short verified snippet. All constructor / function
signatures below were read from the source modules under
`src/mlframe/training/composite/`.

## A. Estimator families

These are sibling wrappers to `CompositeTargetEstimator`, each for a different
target family. All are sklearn-compatible (`fit` / `predict`) and never copy or
down-convert the feature frame.

### `CompositeGLMEstimator` — log-link Poisson / Gamma / Tweedie residual

Models a GLM residual over a positive base mean; for counts, durations, and
insurance-style targets. (Already covered in §6 above — kept here for the menu.)

```python
from mlframe.training.composite import CompositeGLMEstimator
glm = CompositeGLMEstimator(base_estimator=..., family="poisson", tweedie_power=1.5)
```

### `CompositeQuantileEstimator` — native pinball-quantile heads

One inner per quantile fit on the transform `T`, inverted to y-scale, with
per-row non-crossing. Use when you need calibrated conditional quantiles rather
than a single point estimate. (Constructor + snippet in §4 above.)

### `CompositeMultiOutputEstimator` — per-column vector target

One composite per column of a `(n, K)` target. Use for joint multi-output
regression where each output wants its own transform / base. (See §7.)

### `CompositeDistributionEstimator` — full predictive distribution (CRPS)

One sentence: a dense-quantile composite (delegating to
`CompositeQuantileEstimator`) that exposes the *whole* predictive distribution
— CDF, sampling, and a proper CRPS score. When to use: you need calibrated
densities / probabilistic scoring, not just a band.

```python
from mlframe.training.composite import CompositeDistributionEstimator
de = CompositeDistributionEstimator(
    base_estimator=GradientBoostingRegressor(loss="quantile"),
    transform_name="linear_residual", base_column="lag",
)  # quantiles=None -> a dense default grid
de.fit(X, y)
de.predict(X)                              # median point estimate
de.predict_quantile(X)                     # (n, len(grid)) ascending
de.predict_cdf(X, y_grid=[-1.0, 0.0, 1.0]) # (n, 3) P(Y <= g)
de.sample(X, n_samples=100, random_state=0)
crps = de.crps(X, y, reduce="mean")        # proper score (lower=better)
```

### `CompositeSurvivalEstimator` — right-censored time-to-event (AFT-style)

One sentence: a log-time AFT-style composite over a log base column with a 0/1
event indicator and Harrell's concordance. When to use: right-censored
durations / survival times. `censoring="aware"` requires scikit-survival;
`"observed_only"` / `"auto"` fall back to event-only fitting.

```python
from mlframe.training.composite import CompositeSurvivalEstimator
sv = CompositeSurvivalEstimator(base_estimator=..., base_column="log_base",
                                censoring="auto")
sv.fit(X, time, event=event01)            # event=1 observed, 0 censored
sv.predict(X)                              # predicted (positive) time
```

### `CompositePanelEstimator` — entity fixed-effects (longitudinal)

One sentence: subtracts a train-only shrunken per-entity offset (toward the
global mean, `w_e = n_e/(n_e+alpha)`) then learns the within-entity residual.
When to use: panel / longitudinal data with strong entity-level intercepts.

```python
from mlframe.training.composite import CompositePanelEstimator
pe = CompositePanelEstimator(inner_estimator=..., entity_column="user_id",
                             shrinkage_alpha=10.0)
pe.fit(X, y)                               # entity_column read off X
pe.predict(X)                              # unseen entities -> global mean offset
pe.predict(X, entity_id=some_ids)          # override entity ids explicitly
```

### `CompositeRankEstimator` — learning-to-rank within groups

One sentence: residualises the target against a base score *within each query
group* (rank / z-score / raw modes) and learns the within-group ordering
correction. When to use: ranking with a strong base-score prior. `fit` requires
a `group` array; default inner is LightGBM `LGBMRanker` (pairwise fallback).

```python
from mlframe.training.composite import CompositeRankEstimator, ndcg_at_k
rk = CompositeRankEstimator(base_column="base_score", residual_mode="rank")
rk.fit(X, y, group=group_sizes)
scores = rk.predict(X, group=group_sizes)
ndcg = ndcg_at_k(y, scores, group_sizes, k=10)
```

### `OrthogonalizedCompositeEstimator` — double-ML / Neyman-orthogonal base

One sentence: cross-fits nuisance models for `E[y|X]` and `E[base|X]`, then fits
the inner on the orthogonalized (residual-on-residual) signal — debiased base
effect. When to use: you want an unbiased base coefficient under confounding.

```python
from mlframe.training.composite import OrthogonalizedCompositeEstimator
oc = OrthogonalizedCompositeEstimator(base_column="lag", inner_estimator=...,
                                      n_folds=5, random_state=0)
oc.fit(X, y); oc.predict(X)
```

### `BaggedCompositeEstimator` — bootstrap-bagged epistemic ensemble

One sentence: fits `n_estimators` composites on bootstrap resamples with
decorrelated inner seeds; `predict` averages, and the per-member spread gives an
epistemic uncertainty band. When to use: variance reduction + cheap epistemic
intervals without conformal calibration.

```python
from mlframe.training.composite import BaggedCompositeEstimator
bc = BaggedCompositeEstimator(base_estimator=composite_proto, n_estimators=10,
                              bootstrap=True, random_state=0, n_jobs=1)
bc.fit(X, y); bc.predict(X)
```

### `MissingAwareComposite` — NaN-base robustness

One sentence: wraps a single-base composite, dropping fit-time rows whose base
exceeds `max_missing_frac` NaN and imputing the base at predict (median/mean),
without copying the full frame. When to use: the base column has missing values
at inference time.

```python
from mlframe.training.composite import MissingAwareComposite
ma = MissingAwareComposite(composite=CompositeTargetEstimator(...),
                           max_missing_frac=0.5, impute_strategy="median")
ma.fit(X, y); ma.predict(X_with_nan_base)
```

### `TailCompositeEstimator` — extreme-value (GPD) tail quantiles

One sentence: fits a body point-composite plus a Generalized-Pareto tail on the
held-out residual exceedances, giving valid quantiles far past the data range.
When to use: extreme high/low quantiles (risk, VaR-style tails). Prefer a
held-out `residual_X` / `residual_y` so the threshold is not optimistic.

```python
from mlframe.training.composite import TailCompositeEstimator
tc = TailCompositeEstimator(base_estimator=..., transform_name="diff",
                            base_column="lag", threshold_pct=0.9, two_sided=True)
tc.fit(X_tr, y_tr, residual_X=X_cal, residual_y=y_cal)
tc.predict(X)                              # body point prediction
tc.predict_tail_quantile(X, q=0.995)       # GPD-extrapolated extreme quantile
tc.tail_params_                            # fitted (xi, beta, threshold) per side
```

### `CompositeOrRawStacker` — composite-vs-raw meta-stacker

One sentence: cross-fits both a composite and a raw inner on the same data and
learns a non-negative (NNLS) blend from OOF predictions — degrades gracefully to
whichever wins. When to use: you are unsure the composite helps and want an
auto-hedged ensemble.

```python
from mlframe.training.composite import CompositeOrRawStacker
st = CompositeOrRawStacker(base_estimator=..., transform_name="diff",
                           base_column="lag", n_splits=5, random_state=0)
st.fit(X, y); st.predict(X)
```

> `CompositeQRFEstimator` is referenced in the roadmap but is **not** present in
> the current source tree; it is intentionally omitted here until it lands.

## B. Uncertainty menu

The split / CQR / Mondrian / weighted conformal methods bind onto
`CompositeTargetEstimator` (covered in §4). The additions below extend the menu
to classification sets, GLM, multi-output, online, distributional, Venn-Abers,
and extreme-value coverage. Every `calibrate_*` call must use rows the inner
never trained on.

### Online / adaptive conformal (ACI) — streaming coverage

One sentence: an Adaptive Conformal Inference controller on
`CompositeTargetEstimator` that updates the radius online so long-run coverage
tracks `1-alpha` even under drift. When to use: streaming / non-exchangeable
data where a fixed split-conformal radius decays.

```python
est.init_aci(alpha=0.1, gamma=0.05, buffer_n=500, warmup_residuals=resid_cal)
lo, hi = est.predict_interval_online(X)         # current adaptive radius
est.update_conformal(x_new, y_new)              # observe -> advance controller
est.get_aci_state()                             # diagnostic snapshot
```

### Conformal classification SETS (LAC / APS)

One sentence: calibrated prediction *sets* (not single labels) on
`CompositeClassificationEstimator`, via least-ambiguous (`"lac"`) or adaptive
(`"aps"`) scores. When to use: classification where you need a set guaranteed to
contain the true label with probability `>= 1-alpha`.

```python
clf.calibrate_conformal_set(X_cal, y_cal, alpha=0.1, score="lac")
sets = clf.predict_set(X, alpha=0.1, score="lac")   # per-row label set
```

### Venn-Abers — calibrated binary probability intervals

One sentence: an Inductive Venn-Abers calibrator on
`CompositeClassificationEstimator` that returns a probability *interval*
`(p_low, p_high)` plus a calibrated point proba. When to use: you need honest
binary probability calibration with a width that flags uncertainty.

```python
clf.calibrate_venn_abers(X_cal, y_cal)
p_lo, p_hi = clf.predict_proba_interval(X)          # positive-class interval
proba = clf.predict_proba_venn_abers(X)             # (n, 2) calibrated
```

### GLM conformal — variance-scaled intervals

One sentence: a variance-function-scaled split-conformal radius on
`CompositeGLMEstimator`, so the band width *grows with the predicted mean*. When
to use: heteroscedastic count / positive targets where a constant band is wrong.

```python
glm.calibrate_conformal_glm(X_cal, y_cal, alpha=0.1)
lo, hi = glm.predict_interval_glm(X, alpha=0.1)
```

### Multi-output conformal — per-column bands

One sentence: per-column split-conformal on `CompositeMultiOutputEstimator`
(each output calibrated independently, failed columns degenerate to their
fallback). When to use: vector-target intervals.

```python
mo.calibrate_conformal(X_cal, Y_cal, alpha=0.1)
lower, upper = mo.predict_interval(X, alpha=0.1)    # both (n, K)
```

### Distributional CRPS + extreme-value tails

See `CompositeDistributionEstimator.crps` and
`TailCompositeEstimator.predict_tail_quantile` in section A — the distributional
proper-score path and the GPD tail-quantile path of the uncertainty menu.

### Menu (extended)

| Need | Use |
|------|-----|
| Streaming / drift-robust coverage | `init_aci` → `predict_interval_online` + `update_conformal` |
| Classification prediction sets | `calibrate_conformal_set` → `predict_set` |
| Binary probability interval | `calibrate_venn_abers` → `predict_proba_interval` |
| GLM (mean-scaled) interval | `calibrate_conformal_glm` → `predict_interval_glm` |
| Multi-output bands | `CompositeMultiOutputEstimator.calibrate_conformal` → `predict_interval` |
| Proper density score | `CompositeDistributionEstimator.crps` |
| Extreme tail quantile | `TailCompositeEstimator.predict_tail_quantile` |

## C. Workflow helpers

### `suggest_discovery_config` — data-driven config + rationale

One sentence: inspects a bounded sample of the frame and returns a populated
`CompositeTargetDiscoveryConfig` plus a per-field rationale dict. When to use:
you want sensible discovery defaults without hand-tuning. (Snippet in §2.)

```python
cfg, rationale = suggest_discovery_config(df, "target", feature_cols,
                                          sample_n=20_000, seed=0)
```

### `discover_and_wrap` — discovery → fitted estimator in one call

One sentence: runs the full discovery flow and returns a fitted estimator +
provenance report (`DiscoverAndWrapResult`), optionally calibrating a
split-conformal radius on a disjoint holdout. When to use: the one-call happy
path from raw frame to deployable composite.

```python
res = discover_and_wrap(df, "target", feature_cols, train_idx=train_idx,
                        calibrate_conformal=True, conformal_alpha=0.1,
                        holdout_idx=val_idx)
res.estimator.predict(X_new)
```

### `optimize_composite` — joint (transform, inner-HPO) search

One sentence: jointly searches the transform choice and inner hyperparameters by
CV (Optuna when available, else random search) and returns the winning fitted
composite + trial log. When to use: you want the best (transform, params) pair
rather than a fixed transform.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
res = optimize_composite(
    X, y, base_column="lag",
    transform_candidates=("diff", "linear_residual", "ratio"),
    inner_factory=lambda: HistGradientBoostingRegressor(),
    n_trials=30, cv=5, time_ordering=None, prefer_optuna=True,
)
res.estimator.predict(X)                   # winning all-rows-fitted composite
```

### `stability_select_specs` — bootstrap selection frequency

One sentence: re-runs discovery on `n_replicates` row-subsamples and returns
each spec's selection frequency plus the stable subset above `freq_threshold`.
When to use: you want only specs that survive resampling (guard against
seed-lucky discoveries).

```python
res = stability_select_specs(
    discovery_factory=lambda: CompositeTargetDiscovery(cfg),
    df=df, target="target", feature_cols=feature_cols, train_idx=train_idx,
    n_replicates=5, subsample_frac=0.8, freq_threshold=0.6, random_state=0,
)
```

### `discover_incremental` — warm-start on appended data

One sentence: cheaply decides REUSE (prior specs still hold) vs REDISCOVER (DGP
drifted) by re-scoring kept specs' MI gain on a sample of the new frame. When to
use: data grows incrementally and you want to avoid a full re-screen each batch.
Imported from the discovery subpackage:

```python
from mlframe.training.composite.discovery import discover_incremental
decision = discover_incremental(prior_disc, new_df, "target", feature_cols)
if decision.reuse:
    specs = decision.specs                 # prior specs carried forward
```

### `engineer_temporal_bases` — strictly-causal temporal base columns

One sentence: builds a family of strictly-causal lag / rolling / diff base
columns from the target series (sorted by time, mapped back to original row
order). When to use: time-series discovery where you need candidate AR bases.

```python
from mlframe.training.composite import engineer_temporal_bases
bases = engineer_temporal_bases(df, "target", time_column="ts",
                                lags=(1, 2, 3), rolling_windows=(3, 5),
                                ops=("lag", "rolling_mean", "diff"))
# dict[name -> ndarray aligned with df's row order]
```

### Default-ON discovery steps

`auto_chain_discovery_enabled`, `region_adaptive_enabled`,
`interaction_base_discovery_enabled` are ON by default (see "Advanced discovery"
above): auto-chaining appends winning residual→tail chains to `specs_`,
region-adaptive surfaces per-region best transforms on
`discovery.region_adaptive_specs_`, and interaction-base discovery surfaces
synergy bases on `discovery.interaction_bases_`.

## D. Interpretability

### `explain_prediction` — per-row base + residual decomposition

One sentence: returns a `pandas.DataFrame` decomposing each composite prediction
into its base level and learned-residual contribution. When to use: row-level
"why this prediction" attribution for a fitted (additive/multiplicative)
composite. Base-free unary transforms raise.

```python
from mlframe.training.composite import explain_prediction, attribution_summary
df_expl = explain_prediction(est, X)        # per-row base / residual / total
summary = attribution_summary(est, X)        # aggregate base-vs-residual shares
```

### `composite_report` — one explainability report (markdown / html)

One sentence: assembles a single human-readable report (config, attribution,
prediction range, fallback rate, interval coverage when `X`/`y` given) via one
`predict`. When to use: a stakeholder-facing summary of a fitted composite.

```python
from mlframe.training.composite import composite_report
md = composite_report(est, X, y, fmt="markdown")   # or fmt="html"
```

### `composite_model_card` — structured model card

One sentence: builds a structured model-card dict (identity, provenance, params,
readiness, plus evaluation / attribution / leakage when `X`/`y` given). When to
use: governance / model-registry metadata.

```python
from mlframe.training.composite import composite_model_card
card = composite_model_card(est, X, y, target_col="target")
```

## E. Production

### `CompositeDriftMonitor` — deployed-model drift watch

One sentence: a read-only monitor that sketches base / prediction / residual
distributions at train time and flags PSI / KS / residual-shift drift on new
batches, recommending a refit. When to use: monitoring a deployed composite for
input or residual drift.

```python
from mlframe.training.composite import CompositeDriftMonitor
mon = CompositeDriftMonitor(est, base_psi_threshold=0.25)
report = mon.check(X_new, y_new)            # alerts + recommend_update flag
```

### `detect_base_target_leakage` — base-is-y guard

One sentence: tests whether a base column is a near-deterministic function of the
*current* target (Spearman + residual + lag probes), catching a leaking base
before it inflates discovery. When to use: validating a candidate base on
temporal data before trusting its MI gain.

```python
from mlframe.training.composite import detect_base_target_leakage
verdict = detect_base_target_leakage(y, base, time_ordering=ts)
verdict["leaking"]                          # bool + the probe details
```

### `PurgedTimeSeriesSplit` / `make_purged_cv` — leakage-safe CV

One sentence: forward-walk CV with a purge gap and embargo so no train row shares
a label window / short-range autocorrelation with its test fold. When to use:
honest time-series CV for scoring composites.

```python
from mlframe.training.composite import PurgedTimeSeriesSplit, make_purged_cv
cv = PurgedTimeSeriesSplit(n_splits=5, purge=10, embargo=0.01)
for tr, te in cv.split(X):
    ...
```

### `export_serving_spec` / `load_serving_spec` — dependency-light serving

One sentence: serialise a fitted composite's forward/inverse + clip/fallback
params to a JSON-able dict and rebuild a pure-callable y-scale predict (the inner
model is served separately). When to use: lightweight production serving without
shipping the full Python estimator.

```python
from mlframe.training.composite import export_serving_spec, load_serving_spec
spec = export_serving_spec(est)             # json.dumps-able
predict = load_serving_spec(spec, inner_predict=my_inner_raw_predict)
y_hat = predict(X)
```

### `compare_models` / `should_promote` — champion-challenger governance

One sentence: paired-bootstrap (or Wilcoxon) comparison of a fitted challenger
vs champion on held-out data, with `should_promote` adding a significance + min
practical-effect gate. When to use: deciding whether to promote a retrained
model.

```python
from mlframe.training.composite import compare_models, should_promote
cmp = compare_models(champion, challenger, X_hold, y_hold, metric="rmse")
decision = should_promote(champion, challenger, X_hold, y_hold,
                          metric="rmse", alpha=0.05, min_effect=0.0)
decision["promote"], decision["reason"]
```

### `CompositeFeatureGenerator` — OOF composite-prediction feature

One sentence: a sklearn transformer that turns a discovered spec into one
leakage-free OOF feature column (`fit_transform` on train, `transform` on new
data via a single all-train wrapper). When to use: stacking a composite's
prediction as a feature into a downstream model.

```python
from mlframe.training.composite import CompositeFeatureGenerator
gen = CompositeFeatureGenerator(spec=disc.specs_[0], n_splits=5, time_aware=False)
X_train_aug = gen.fit_transform(X_train, y_train)   # adds OOF column
X_new_aug = gen.transform(X_new)
```
