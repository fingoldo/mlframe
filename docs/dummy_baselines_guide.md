# Dummy baselines

Pre-training trivial-baseline floor. Runs once per `(target_type, target_name)`
in `train_mlframe_models_suite` (and per-target in `train_mlframe_ranker_suite`)
*after* `BaselineDiagnostics` and *before* per-target model training. Answers
the operator question:

> Is this task even hard? How much lift does my trained model actually have
> over a constant prediction / a per-group historical mean / a TS-naive lag?

Output lands at `metadata["dummy_baselines"][target_type][target_name]` and
the verdict line is logged at INFO before model training starts. After all
models train, a suite-end cross-target verdict block emits with canonical
UPPERCASE WARN tokens that grep cleanly out of any log shipping pipeline.

## Sit-alongside relationship with `BaselineDiagnostics`

The two modules are complementary, not redundant:

- **`BaselineDiagnostics`**: "is the target *predictable* from these
  features at all?" — quick-fit headline metric + top-K feature ablation +
  init_score residual baseline.
- **`dummy_baselines`** (this module): "is the *task* even hard?" —
  comparison to trivial reference predictors that ignore the features.

Both run before model training; neither is sufficient alone.

## Quickstart

```python
from mlframe.training.configs import DummyBaselinesConfig
from mlframe.training.core import train_mlframe_models_suite

models, metadata = train_mlframe_models_suite(
    df=df,
    target_name="TVT",
    model_name="experiment_1",
    features_and_targets_extractor=fte,
    mlframe_models=["cb", "lgb", "xgb"],
    # default ON for all 6 target_types — opt out via:
    dummy_baselines_config=DummyBaselinesConfig(enabled=False),
)

# Read back per-target reports
db = metadata.get("dummy_baselines", {})
for tt, by_name in db.items():
    for tn, rep in by_name.items():
        print(tt, tn, rep["strongest"], rep["primary_metric"])
        # rep["data"] — full per-baseline x per-split metrics dict
        # rep["extras"]["paired_bootstrap"] — Δ vs runner-up + 95% CI + P
        # rep["extras"]["bootstrap_ci"] — strongest CI when n was small
        # rep["extras"]["per_output_strongest"] — multi-output regression block
        # rep["extras"]["ts_diagnostics"] — ACF peaks + step-size + used periods
        # rep["extras"]["per_group"] — coverage_pct + repeat_entity_rate
        # rep["plot_path"] — overlay PNG path (None if suppressed / no plot_file)

# Failures (try/except wrapped — never blocks training):
failures = metadata.get("dummy_baselines_failures", {})
```

For learning-to-rank, the same parameter is wired on `train_mlframe_ranker_suite`:

```python
from mlframe.training.ranker_suite import train_mlframe_ranker_suite

train_mlframe_ranker_suite(
    df=...,
    target_name="...",
    model_name="...",
    features_and_targets_extractor=fte,
    dummy_baselines_config=DummyBaselinesConfig(...),
)
```

## Sample log output

Default INFO level (≤ 4 lines per target):

```
[DUMMY_BASELINES] target='TVT' regression ts_period=7 n_train=4_091_828 (finite=4_091_828) n_val=461_278 (finite=461_278) n_test=524_396 (finite=524_396)
[DUMMY_BASELINES] target='TVT' strongest=seasonal_naive_p7 (ts) val_RMSE=497.66 lift_vs_mean=+23.0% (n_baselines=9, full table at DEBUG)
[DUMMY_BASELINES] target='TVT' Delta_RMSE vs runner-up (naive_lag7) = -11.74 [95% bootstrap CI: -14.20, -9.31]; beats runner-up in 87% of resamples
[DUMMY_BASELINES] target='TVT' overlay plot saved: data/models/TVT/basic/baseline_TVT_seasonal_naive_p7.png
```

Suite-end summary block (after all models train):

```
[DUMMY_BASELINES] CROSS-TARGET VERDICT
target          strongest_dummy        dummy_metric    best_model  model_metric    lift   verdict
TVT             seasonal_naive_p7      val_RMSE=497.66 cb          val_RMSE=14.20  35.05x TASK_NON_TRIVIAL_AND_MODELS_HEALTHY
EGFDU           median                 val_RMSE=12.50  xgb         val_RMSE=12.40  1.01x  MODELS_BARELY_BEAT_TRIVIAL
[DUMMY_BASELINES] WARN BEST_MODEL_BELOW_DUMMY target='EGFDU' lift=1.01x — investigate label encoding, target leak, train/test contamination.
[DUMMY_BASELINES] HEALTH: 1/2 targets — see WARN lines above
```

To see the full per-baseline × per-split metrics table for one target, raise
the logger level for `mlframe.training.dummy_baselines` to `DEBUG`.

## Per-target catalog

### REGRESSION / QUANTILE_REGRESSION

Constant predictors:
- `mean` — `train_y.mean()` broadcast.
- `median` — `np.median(train_y)` broadcast (right-skewed targets).
- `quantile_p25`, `quantile_p75` — empirical 25/75-th percentile of train_y.

Group-aware:
- `per_group_mean` (or `per_group_historical_mean (ts)` when timestamps are
  monotonic across train/val/test) — fit per-category target mean on the
  highest-cardinality categorical that *passes* the gates:
  - **Cardinality cap**: skip if `n_unique(cat_col) > 0.5 * n_train`
    (row-id-like → silent perfect-prediction oracle).
  - **Coverage check**: log `val_coverage_pct = mean(val_cat in train_cat)`;
    exclude from strongest-pick if coverage < 50% on either split.
  - **Entity-overlap diagnostic**: log
    `repeat_entity_rate = fraction of val rows whose group has >=5 train labels`;
    annotate row label as `per_group_mean (high entity overlap — measures
    re-appearance, not generalization)` when > 0.5.
  - **NaN-key handling**: cat values coerced via
    `pd.Series(col).astype(str).fillna("__NULL__")` so polars Categorical /
    Enum / pandas categorical / object share one path; numeric columns use
    a fast path that bypasses stringification (~13× faster on 1M rows).

Time-series (only when timestamps are present and *temporally monotonic*
across train/val/test):
- `naive_last` — constant `y_train[-1]` (suppressed when `n_val > inferred_period`
  to avoid degenerating into the `mean` baseline).
- `naive_lag7`, `naive_lag30` — constant `y_train[-7]` / `y_train[-30]`.
- `seasonal_naive_pP` — for each ACF-detected period P, predicts
  `y_train[-P + (k mod P)]` for the k-th val/test row.
- `rolling_mean_w7`, `rolling_mean_w30` — constant `y_train[-W:].mean()`,
  emitted only when ACF detected a peak ≥ W (otherwise dropped — no
  information beyond `mean`).
- `linear_extrap` — OLS slope/intercept on the train tail (capped at 10000
  rows for cost), extrapolated to val/test timestamps.

For `quantile_regression` with `quantile_alphas=[0.1, 0.5, 0.9]` passed
explicitly, a dedicated dispatcher emits multi-output `(N, K)` predictions:
- `quantile_alpha_{a:.3f}` per α — constant prediction = empirical α-th
  percentile of train_y (clamped to `[1e-3, 1-1e-3]` for boundary).
- `quantile_alpha_0.500 (=median by construction)` — self-consistency note.
- `median_for_all` — single `np.median(train_y)` broadcast across all α.
- `multi_quantile_empirical` — the j-th α-th percentile in the j-th column
  (the proper multi-quantile constant baseline).

Headline metric for quantile_regression: `val_pinball_mean` (mean over
non-boundary α in `[0.05, 0.95]`); per-α `val_pinball@{a:.3f}` columns
retained for full table inspection.

### BINARY_CLASSIFICATION

- `prior` — constant `train_prior` probabilities (minimises log_loss at base rate).
- `most_frequent` — argmax(class_counts) one-hot.
- `all_zeros`, `all_ones` — symmetric constant baselines.
- `uniform` — 0.5 / 0.5 (binary) or 1/K (multiclass).
- `stratified` — random sampling at train prior, **20 deterministic seeds
  averaged** (`mean ± std` reported); single-seed estimates are
  indistinguishable from real signal at `n_val < 2000`.
- `per_group_prior` — per-category positive rate (same gates as regression
  `per_group_mean`).
- TS-only: `naive_last_class`, `rolling_majority_w24`.

**Headline metric = `log_loss`** (D5). Constant baselines all collapse to
`AUC = 0.5` by construction, so AUC cannot discriminate them. AUC is shown
as a secondary column with annotation.

### MULTICLASS_CLASSIFICATION

`prior`, `most_frequent`, `uniform`. Headline = `log_loss`.

### MULTILABEL_CLASSIFICATION

`all_zero`, `all_one`, `per_label_prior`, `per_label_most_frequent`.
Columns explicitly named: `val_log_loss_macro`, `val_log_loss_micro`,
`test_log_loss_macro`, `test_log_loss_micro`. Headline = macro log-loss
(skips all-constant labels in the macro mean). Sample-AUC deliberately
out of scope (cost > value).

Macro/micro log-loss are computed via a numba-JIT'd kernel
(`@njit(parallel=True, fastmath=True)`) — ~57× faster than the per-label
sklearn loop on n=10⁵, K=10. Optional dep — graceful fallback to sklearn
on import failure.

### LEARNING_TO_RANK

- `random_within_query` — random scores within each query group, **10
  deterministic seeds averaged** (`mean ± std` reported).
- `identity_input_order` — predict scores in feature-row order
  (1 / rank-within-group); tests if the upstream system has already sorted.
- `mean_relevance` — constant `train_y.mean()` (within-query ordering
  defined as input order via stable `np.argsort`).
- `most_recent_first` (TS-only) — rank by recency within group.

Headline = `NDCG@10`. Group sanity gate hard-fails on misaligned
`group_ids` length (caller bug, not runtime degraded condition).

`popularity` baseline DROPPED — mlframe's LTR API exposes only
`group_ids` (= qid), not per-row doc-id. Without an FTE-level `doc_field`,
popularity-by-doc-frequency cannot be computed. Documented as a known
absence rather than a silent gap.

## How time-series detection works

Activates when ALL of:
1. `getattr(features_and_targets_extractor, "ts_field", None)` is set.
2. `timestamps` (from FTE.transform) is not None.
3. **Temporal monotonicity** check passes:
   `timestamps[train].max() <= timestamps[val].min()` AND
   `timestamps[val].max() <= timestamps[test].min()`. Interleaved random
   splits → TS baselines skipped + INFO line emitted.

Period candidates assembled via:
- **Step inference**: median of `np.diff(np.unique(timestamps))` →
  hourly / daily / weekly / monthly defaults.
- **ACF peaks**: top-2 peaks of `acf(np.diff(y_train, n=1))` above
  `2 / sqrt(n)` (statsmodels significance threshold). First-differencing
  removes linear trend without STL fit cost; raw-series ACF is otherwise
  dominated by trend autocorrelation, not seasonality.
- **Stratified contiguous-window sample** for `n_train > 50 000` —
  avoids tail-only bias toward the regime of the most-recent rows
  (e.g. promotional / holiday weeks for retail).
- **Period filter**: keep only `2 <= P <= n_train // 4` (Nyquist-ish minimum
  for 4 full cycles in train).
- **User override** via `config.ts_extra_periods`.

Logged once per target:

```
[dummy-baselines] ts_periods: step=daily defaults={1,7,30,365}; acf_peaks={7,14}; using={1,7,14,30,365}
```

## Strongest-pick + paired-bootstrap robustness

After `argmin/argmax` on the primary metric, a **non-degeneracy gate**
runs on the reference split (val first, test fallback):
- classification: `len(unique(y_ref)) >= 2 AND n_ref >= 10`;
- regression / quantile: `np.std(y_ref) > 1e-9 AND n_ref >= 10`;
- LTR: `n_groups_ref >= 2 AND any(group_size > 1)`.

Failure → `strongest=None`, no plot, log explains why.

When `min(n_val, n_test) < bootstrap_ci_threshold` (default 2000), a
**paired bootstrap (1000 resamples)** runs against the runner-up baseline:
- Δ_metric = `strongest_metric - runner_up_metric` (sign-corrected by
  minimize/maximize convention).
- 95% CI from percentiles of resampled Δ.
- `P(strongest beats runner-up)` = fraction of resamples where strongest wins.

If `P < strongest_min_beat_runner_up_prob` (default 0.7), the verdict line
is annotated `(beats runner-up in {pct}% of resamples — TIE, treat as
noise)` and the overlay plot is suppressed. Above the n-threshold the
paired bootstrap is skipped entirely (point-estimate signal-to-noise is
high enough that paired bootstrap is just expensive ceremony).

A separate `bootstrap_ci` block under the same n-threshold provides a
95% CI on the strongest baseline's primary metric (1000 resamples,
~1s cost on n=10⁴).

## Configuration

```python
DummyBaselinesConfig(
    enabled=True,                                # default ON
    apply_to_target_types=frozenset({           # opt-out per target_type
        "regression", "binary_classification",
        "multiclass_classification", "multilabel_classification",
        "learning_to_rank", "quantile_regression",
    }),

    # User-defined TS periods (in addition to ACF auto-detect + step-inference defaults)
    ts_extra_periods=(),                         # e.g. (365,) for annual seasonality

    # Per-group leakage gates
    per_group_max_cardinality_ratio=0.5,         # skip cat if n_unique > 0.5 * n_train
    per_group_min_val_coverage_pct=50.0,         # exclude from strongest if coverage < 50%
    per_group_high_overlap_threshold=0.5,        # entity-overlap diagnostic threshold

    # n_repeats for stochastic baselines
    stratified_n_repeats=20,                     # binary stratified
    random_within_query_n_repeats=10,            # LTR random_within_query

    # Bootstrap CI / paired-bootstrap
    bootstrap_ci_threshold=2000,                 # CI fires only when min(n_val,n_test) < this
    bootstrap_ci_n_resamples=1000,
    paired_bootstrap_n_resamples=1000,
    strongest_min_beat_runner_up_prob=0.7,       # below this → TIE annotation

    # Suite-end alarm threshold
    best_model_min_lift=1.5,                     # lift < 1.5x → WARN BEST_MODEL_BELOW_DUMMY

    # Reproducibility
    random_state=42,
)
```

## Operator alarm WARN tokens

Four canonical UPPERCASE tokens, grep-able from any log:

| Token | Trigger | Suggested investigation |
|---|---|---|
| `[DUMMY_BASELINES] WARN BEST_MODEL_BELOW_DUMMY` | best model's primary metric lift < `best_model_min_lift` (default 1.5x) | label encoding, target leak, train/test contamination |
| `[DUMMY_BASELINES] WARN ALL_BASELINES_BELOW_RANDOM` | binary: every classifier baseline AUC < 0.5 | check `target_label_encoder` direction; check sign of `cost_function` |
| `[DUMMY_BASELINES] WARN TS_BEATS_TREES` | strongest TS baseline beats best model on val | verify `val_placement='forward'`; check for leaked-from-future feature columns |
| `[DUMMY_BASELINES] WARN PARTIAL_FAILURE` | per-baseline failures occurred (NaN rows) | review per-baseline failure reason in `metadata["dummy_baselines_failures"]` |

The suite-end summary block emits these after all models complete training.

## Multi-output regression (D4)

For 2D target arrays `y.shape = (N, K)`, the dispatcher runs the regression
path per output and aggregates:

- **Per-output strongest-pick block** — one line per output, naming the
  strongest baseline + its primary metric value + scale-free normalized
  metric (`RMSE / std(y_train_per_target)`).
- **Cross-output normalized strongest-pick** — picks the baseline whose
  mean normalized RMSE is lowest across outputs. Surfaces in the suite-end
  verdict table as ONE row per multi-output target (not K rows).

```
[dummy-baselines] target='Y' (multi-output regression, K=3) per-output strongest:
  Y[0]: median  (RMSE=12.50, normalized=0.41)
  Y[1]: naive_lag7 (RMSE=8.30, normalized=0.55)
  Y[2]: mean    (RMSE=2.10, normalized=0.50)
[dummy-baselines] target='Y' cross-output normalized strongest=median (mean_normalized_RMSE=0.49)
```

Overlay plot generated for `Y[0]` only (one-plot-per-target invariant).

## When the diagnostic is skipped

Per-baseline failures are recorded as NaN rows + `failed=True` flag and
logged once per `(metric_fn.__name__, error_type)` — never silently
swallowed. Block-level skip reasons:

- `config.enabled=False` — explicitly disabled.
- `target_type='X' not in apply_to_target_types=...` — caller opted out.
- Object-dtype target incompatible with the target_type (e.g. string
  values for regression) — D8 gate.
- Both val and test targets have <2 finite values — D9 gate.
- Non-degeneracy gate fails on both reference splits — `strongest=None`,
  table still emitted for inspection.
- `internal_error: ...` — anything else; outer try/except in
  `train_mlframe_models_suite` records to `metadata["dummy_baselines_failures"]`.

## Output schema

`metadata["dummy_baselines"][target_type][target_name]` is the result of
`BaselineReport.to_dict()`:

```python
{
    "schema_version": "1.0",                  # for forward-compat (D14)
    "target_type": "regression",
    "target_name": "TVT",
    "data": {                                 # per-baseline x per-metric (NaN scrubbed to None)
        "mean":               {"val_RMSE": 645.34, "val_MAE": 518.21, "test_RMSE": 646.10, "test_MAE": 518.95, "failed": False},
        "median":             {"val_RMSE": 647.81, ...},
        "seasonal_naive_p7":  {"val_RMSE": 497.66, ...},
        ...
    },
    "strongest": "seasonal_naive_p7 (ts)",
    "primary_metric": "val_RMSE",
    "ts_period_used": 7,                      # None for non-TS targets
    "plot_path": "data/.../baseline_TVT_seasonal_naive_p7.png",
    "elapsed_s": 1.23,
    "n_train": 4_091_828, "n_val": 461_278, "n_test": 524_396,
    "n_train_finite": 4_091_828, "n_val_finite": 461_278, "n_test_finite": 524_396,
    "extras": {                               # target-type-specific diagnostics
        "ts_diagnostics":      {"step": "daily", "acf_peaks": [7, 14], "used": [1, 7, 14, 30, 365]},
        "per_group":           {"cat_col": "user_id", "val_coverage_pct": 92.3, "test_coverage_pct": 91.8, "repeat_entity_rate": 0.42, "n_groups_train": 14523},
        "paired_bootstrap":    {"runner_up": "naive_lag7", "delta": -11.74, "delta_ci": [-14.20, -9.31], "p_strongest_beats": 0.87, "split_used": "val"},
        "bootstrap_ci":        {"val": [495.0, 497.66, 500.3], "test": [496.5, 498.93, 501.2]},
        "per_output_strongest": [...],         # multi-output regression
        "cross_output_strongest": {...},       # multi-output regression
        "tie": False,                          # paired-bootstrap robustness flag
        "quantile_alphas":     [0.1, 0.5, 0.9],
        "quantile_n_eff_val":  {0.1: 8, 0.5: 50, 0.9: 92},
        "n_classes": 5,                        # multiclass / binary
    },
}
```

JSON-serializable: NaN values are scrubbed to None (D15) so
`json.dumps(report.to_dict())` succeeds.

## Pure w.r.t. inputs (sweep-orchestrator memoization)

`compute_dummy_baselines` is pure with respect to `(target_type, train_y,
val_y, test_y, timestamps, group_ids, cat_features_chosen)`. Hyperparam-
sweep orchestrators can trivially memoize via:

```python
import hashlib

def _baseline_inputs_hash(target_type, train_y, val_y, test_y, timestamps=None, group_ids=None):
    h = hashlib.sha256()
    h.update(str(target_type).encode())
    for arr in (train_y, val_y, test_y, timestamps, group_ids):
        h.update(np.ascontiguousarray(arr).tobytes() if arr is not None else b"None")
    return h.hexdigest()
```

(Same recipe is exposed as the private `_baseline_inputs_hash` helper in
the module.)

## Limitations

- **Quantile per-α requires explicit `quantile_alphas` kwarg** at call
  site. Without it, `quantile_regression` routes through the regression
  path (mean/median/p25/p75 + RMSE/MAE) — adequate constant-prediction
  floor but doesn't surface per-α structure.
- **LTR popularity baseline** unimplementable on the current FTE
  protocol (no `doc_field`); deferred until / unless that lands.
- **Sample-AUC for multilabel** is deliberately out of scope (cost > value).
- **Plot rendering only for the strongest baseline** (per-target); 5
  targets ≤ 5 plots, not N×baselines.
- **Per-cell metric isolation only** — block-level errors during
  predictor construction (e.g. ACF on a constant series) still skip
  individual TS baselines but are surfaced via INFO logs.

## Profiling + smoke

- `python -m mlframe.training._profile_dummy_baselines` — cProfile harness;
  prints per-target wall-time + top-30 cumulative-time entries.
- `python mlframe/training/_smoke_dummy_baselines_e2e.py` — end-to-end
  smoke through `train_mlframe_models_suite` with one lgb-target.

Wall time: ~1s/target on 1M-row × 1-target after the numba pass; ~0.04s/target
on multilabel n=10⁵, K=10. Extrapolated 5M-row × 5-target ≈ 5s — well inside
the original 30-120s plan budget. cProfile attribution overhead inflates
apparent pandas/sklearn-internal call timings ~10-13× vs standalone wall-time
microbench; the profile harness docstring documents this so future re-runs
don't re-flag attribution noise as real hotspots.
