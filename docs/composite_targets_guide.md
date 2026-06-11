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
