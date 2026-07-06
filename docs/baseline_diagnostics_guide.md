# Baseline diagnostics

Cheap pre-training diagnostic that runs once per `(target_type, target_name)`
in `train_mlframe_models_suite` and surfaces three answers operators ask
*after* a model is trained:

1. What is the headline metric of a quick fit?
2. Which features are dominant — and by how many percentage points?
3. Does native residual learning (`init_score` / `base_margin`) already
   capture the dominant signal?

Output lands at `metadata["baseline_diagnostics"][target_type][target_name]`
and is logged via `format_baseline_diagnostics_report` at INFO level
before per-target training starts.

## Why this exists

The original motivation: a data scientist predicting `TVT` on ~100 features
only learned that `TVT_prev` dominated feature importance after the full
suite finished training. Hours of compute to surface a one-line diagnostic.

This module surfaces the dominant feature and quantifies its contribution
*before* per-target training begins, so you can act on it:

- accept the dominance (the model is doing the right thing);
- drop the feature (lift other features into FI);
- configure composite-target discovery (residual learning).

## What it does

```python
from mlframe.training.core import train_mlframe_models_suite

models, metadata = train_mlframe_models_suite(
    df=df,
    target_name="TVT",
    model_name="experiment_1",
    features_and_targets_extractor=fte,
    mlframe_models=["lgb", "xgb"],
    # default ON for regression / binary classification — opt out via
    # baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False)
    # (BaselineDiagnosticsConfig lives in mlframe.training.configs); with
    # enabled=False, metadata["baseline_diagnostics"] is not populated at all.
)

# Read it back
report = metadata["baseline_diagnostics"]["regression"]["TVT"]
print(report["composite_recommendation"])      # 'high_potential' / 'marginal' / 'unlikely_to_help' / 'skipped'
print(report["ablation"][0])                    # top dominant feature, e.g. {'feature': 'TVT_prev', 'metric_after_drop': ..., 'delta_pct': 672.4, 'rank': 1}
print(report["init_score_baseline"])            # native residual baseline (or None)
```

### Sample log output

```
[BaselineDiagnostics] target='TVT' (regression) RMSE_raw=11497.6592 sample_n=50000 elapsed=42.3s
[BaselineDiagnostics] Ablation (drop -> Δ%, positive = drop hurt):
  rank=1 TVT_prev                Δ%=+20.51 RMSE_after_drop=13854.2102
  rank=2 X                       Δ%=+0.04  RMSE_after_drop=11502.1308
  rank=3 dY                      Δ%=+0.02  RMSE_after_drop=11500.0011
  ...
[BaselineDiagnostics] init_score(TVT_prev) RMSE=11489.32 Δ%=-0.07 vs raw
[BaselineDiagnostics] composite_recommendation=high_potential
  reason: top ablation Δ%=20.51 >= 5.00 (strong dominant feature); init_score baseline still off raw by -0.07% (residual has structure)
```

## How it works

1. **Sample**. Random 50 000-row sample (configurable). Larger frames are
   subsampled to keep the diagnostic under one minute.
2. **Quick fit**. One LightGBM model on an 80/20 random holdout with
   `n_estimators=100`. Records headline metric (`RMSE` for regression,
   `AUC` for binary) and `feature_importances_`. (Was 200; a 6-scenario x
   3-seed bench showed the dominant-feature verdict is identical at 100 and
   the ablation runs ~1.8x faster — see
   `src/mlframe/training/baselines/_benchmarks/bench_ablation_n_estimators_provisioning.py`.)
3. **Ablation**. Top-K features by FI are each dropped in turn (sequential,
   independent — *not* cumulative), the model is refit on the reduced
   feature set, and Δ% is measured against the raw fit. Sign convention:
   positive Δ% always means "dropping this feature hurt performance".
4. **`init_score` baseline** (regression and binary classification). Top-1 dominant
   feature is passed as `init_score` to a fresh LightGBM. The model now
   learns only the residual. If the resulting metric is within
   `init_score_optimal_threshold_pct` of the raw fit, native residual
   learning already captures the dominant signal.
5. **Recommendation**. Three-way classifier:

   | Recommendation | Condition |
   |---|---|
   | `high_potential` | max ablation Δ% ≥ `high_potential_min_dominance_pct` AND init_score baseline did NOT close the gap |
   | `marginal` | max ablation Δ% in `[marginal_threshold_pct, high_potential_min_dominance_pct)` |
   | `unlikely_to_help` | max ablation Δ% < `marginal_threshold_pct` OR init_score baseline matches raw within threshold |
   | `skipped` | config disabled / unsupported target_type / degenerate inputs |

## Configuration

```python
BaselineDiagnosticsConfig(
    enabled=True,                                 # default ON
    ablation_top_k=5,                             # drop top-5 by FI
    quick_model_n_estimators=100,
    quick_model_num_leaves=31,
    quick_model_learning_rate=0.05,
    init_score_top_k=1,                           # 1 = single-feature, K>1 = OLS combiner
    init_score_apply_to_target_types=("regression", "binary_classification"),  # binary supported via a logit-space init_score
    sample_n=50_000,                              # None = use full train
    high_potential_min_dominance_pct=5.0,         # Δ% threshold for "dominant"
    init_score_optimal_threshold_pct=1.0,         # init_score within 1pct of raw -> already optimal
    marginal_threshold_pct=2.0,
    apply_to_target_types=("regression", "binary_classification"),
    random_state=42,
)
```

## When the diagnostic is skipped

The component never raises into the training pipeline; on any internal
error it returns a `skipped=True` report with `skip_reason` set and emits
a single WARNING log. Skip reasons in the wild:

- `config.enabled=False` — explicitly disabled.
- `target_type='X' not in apply_to_target_types=...` — multiclass /
  multilabel / LtR / quantile_regression are out of scope (init_score
  semantics don't carry).
- `no feature_cols provided` — degenerate input.
- `length mismatch X=N vs y=M` — caller passed misaligned arrays.
- `raw quick-fit metric is non-finite` — typically a constant target or
  a near-empty sample.
- `internal_error: ...` — anything else (LightGBM unavailable, fit
  exception, etc.).

## Limitations

- **Sequential ablation**, not joint. Two features that are individually
  small but jointly large will not surface as dominant. Cost of joint
  ablation grows combinatorially; out of scope here.
- **Sample bias**. The diagnostic runs on a random 50 000-row sample by
  default. For temporal datasets this is *not* a temporal split — set
  `sample_n=None` to use the full train, or pre-sort and use a smaller
  sample if recency matters more than statistical noise.
- **`init_score` baseline now covers regression AND binary classification.**
  Binary's `init_score` is a logit offset: top-K dominant features are
  LR-combined into a probability-scale score, converted to logit, and
  passed as LightGBM's `init_score=` so the booster learns the residual
  logit.
- **Multiclass / multilabel skipped**. The native residual story breaks
  down (no scalar `y - base`). Future composite-target discovery will
  decide whether to extend this.
