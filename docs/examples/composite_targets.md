# Composite-target discovery: usage examples

Copy-pasteable recipes for `train_mlframe_models_suite` with the
composite-target feature. Three complexity tiers (zero-config, minimal
opt-in, production) plus how to read results back, render a stakeholder
report, and roll back if things break.

For the conceptual walkthrough see
[`docs/composite_targets_tutorial.ipynb`](../composite_targets_tutorial.ipynb).
For the diagnostic that runs by default see
[`docs/baseline_diagnostics_guide.md`](../baseline_diagnostics_guide.md).

## Decision tree: which tier do I need?

```
What's the task?
├── Multilabel / multiclass / LtR / quantile -> NOT supported.
│   metadata["composite_target_failures"] will say why.
├── Regression
│   ├── 100+ features, no idea which is dominant       -> Tier 1 (auto-discovery)
│   ├── BaselineDiagnostics says "unlikely_to_help"     -> stay at Tier 0
│   ├── Online single-row predict + tight latency SLA   -> Tier 2 + max_inference_components
│   ├── Regulated / audit / bit-exact reproducibility   -> Tier 2 + deterministic_screening_models
│   └── Production -- everything on                     -> Tier 2 (full stack)
└── Binary classification -> BaselineDiagnostics works;
                             composite-discovery skips with explicit reason.
```

## Tier 0: zero-config (default behaviour)

Composite-target discovery is OFF by default. ``BaselineDiagnostics`` runs
automatically for regression and binary classification (default ON, ~30-90s
overhead on a sample). No code change needed -- the metadata is just
populated for free.

```python
from mlframe.training.core import train_mlframe_models_suite

models, metadata = train_mlframe_models_suite(
    df=df,
    target_name="TVT",
    model_name="exp_baseline_only",
    features_and_targets_extractor=fte,
    mlframe_models=["lgb", "xgb"],
)

# Always available (default):
diag = metadata["baseline_diagnostics"]["regression"]["TVT"]
print(diag["composite_recommendation"])
# -> "high_potential" / "marginal" / "unlikely_to_help"

for entry in diag["ablation"]:
    print(f"  rank {entry['rank']}: drop {entry['feature']!r}"
          f" -> RMSE_after_drop={entry['metric_after_drop']:.3f}"
          f" (delta%={entry['delta_pct']:+.2f})")
# rank 1: drop 'TVT_prev' -> RMSE_after_drop=13854.21 (delta%=+20.51)
# rank 2: drop 'X'        -> RMSE_after_drop=11502.13 (delta%=+0.04)
# ...
```

Use this output to decide whether to opt into composite mode. If the
recommendation is `unlikely_to_help`, skip Tier 1 entirely.

## Tier 1: minimal opt-in (recommended starting point)

Enable discovery and the cross-target ensemble with sensible defaults.
Auto-base picks the dominant feature; the full default transform set is tried.

```python
from mlframe.training.configs import CompositeTargetDiscoveryConfig
from mlframe.training.core import train_mlframe_models_suite

cfg = CompositeTargetDiscoveryConfig(
    enabled=True,
    cross_target_ensemble_strategy="oof_weighted",
)

models, metadata = train_mlframe_models_suite(
    df=df,
    target_name="TVT",
    model_name="exp_composite",
    features_and_targets_extractor=fte,
    mlframe_models=["lgb", "xgb"],
    composite_target_discovery_config=cfg,  # NEW kwarg
)

# Cross-target ensemble (y-scale predictions out of the box):
ensemble_entry = models["regression"]["_CT_ENSEMBLE__TVT"][0]
y_pred = ensemble_entry.model.predict(X_new)
```

Inside the defaults: `base_candidates="auto"`, the full default transform
set (24 entries -- residual/ratio/unary/chain families; see
`CompositeTargetDiscoveryConfig.transforms` or `list_transforms()`), bin-based
MI (`mi_estimator="bin"`), hybrid screening (`screening="hybrid"`),
`auto_skip_on_baseline_optimal=False`.

## Tier 2: production config (all optimisations on)

```python
from mlframe.training.configs import CompositeTargetDiscoveryConfig

cfg = CompositeTargetDiscoveryConfig(
    enabled=True,

    # Smart skip when the diagnostic says composite won't help.
    auto_skip_on_baseline_optimal=True,

    # Two-stage screening: cheap MI prefilter -> tiny-model rerank.
    screening="hybrid",                 # default; "mi" for MI-only, "tiny_model" for rerank-only
    mi_estimator="bin",                 # default; 38x faster MI than the opt-in kNN Kraskov path
    tiny_model_n_jobs=3,                # parallel CV folds (Phase B)
    top_k_after_mi=8,
    top_m_after_tiny=3,

    # Cross-target ensemble + honest validation.
    cross_target_ensemble_strategy="linear_stack",  # Ridge stack
    oof_holdout_frac=0.2,                           # honest 20% holdout
    max_inference_components=3,                     # latency cap for online predict

    # Optional: deterministic mode for regulated / CI use cases.
    deterministic_screening_models=False,           # default OFF; True for bit-equality
    random_state=42,
)
```

Notes:

- `mi_estimator="bin"` is the default bin-based MI; 38x faster than the
  opt-in kNN Kraskov path but biased low on heavy-tail distributions. For
  near-Gaussian targets it picks the same dominant base. Opt into
  `mi_estimator="knn"` if you have power-law / heavy-tail targets.
- `oof_holdout_frac=0.2` re-fits a clone of every component on 80% of
  train and predicts on the held-out 20%. Honest holdout drives the
  ensemble weights / stacking and the validation gate. Cost: one
  extra fit per component.
- `max_inference_components=3` trims the ensemble to the top-3
  components by absolute weight at build time. Useful when you have
  K=8 wrappers and a 1-row online predict SLA in milliseconds.
- `deterministic_screening_models=True` injects per-family determinism
  flags (LightGBM `deterministic+force_row_wise`, XGB `tree_method="hist"`,
  CatBoost `boosting_type="Plain"`) so run-to-run results are bit-equal.
  5-10% slower per fit; default OFF.

## Reading results from `metadata`

```python
# Discovered specs with justification numbers
specs = metadata["composite_target_specs"]["regression"]["TVT"]
# [
#   {"name": "TVT-linres-TVT_prev",
#    "transform_name": "linear_residual",
#    "base_column": "TVT_prev",
#    "fitted_params": {"alpha": 0.952, "beta": -1.5, ...},
#    "mi_gain": 0.5203, "mi_y": 0.0453, "mi_t": 0.5655,
#    "valid_domain_frac": 1.0, "n_train_rows": 1200},
#   ...
# ]

# Rejected candidates with reasons
failures = metadata["composite_target_failures"]["regression"]["TVT"]
# [{"name": "...", "reason": "valid_domain_frac=0.42 < 0.7"},
#  {"name": "...", "reason": "mi_gain=-0.04 <= eps=0.01"}, ...]

# Y-scale RMSE/MAE per composite per split (NOT T-scale). Empty by default because
# `skip_wrap_pass_predict=True` (default) skips the wrap-pass y-scale predict() calls
# for speed; pass `skip_wrap_pass_predict=False` in the config above to populate this,
# or call `mlframe.training.core._phase_composite_post.recover_composite_y_scale_metrics`
# on demand after the fact.
y_metrics = metadata["composite_target_y_scale_metrics"]["regression"]
for composite_name, entries in y_metrics.items():
    for e in entries:
        test = e["metrics"].get("test", {})
        print(f"  {composite_name}: test_RMSE={test.get('RMSE', float('nan')):.4f}")

# Cross-target ensemble: strategy + per-component weights
ens = metadata["composite_target_ensemble"]["regression"]["TVT"]
# {"strategy": "linear_stack",
#  "component_names": ["raw#0", "TVT-linres-TVT_prev#0"],
#  "weights": [0.3, 0.7],
#  "notes": {"intercept": 0.5, ...}}

# Schema bump for forward-compat checks
metadata["schema_version"]  # == 2

# Env signature for debugging version drift between save / load time
metadata["composite_target_env_signature"]
# {"numpy": "1.26.4", "lightgbm": "4.5.0", "sklearn": "1.6.0", ...}
```

## Stakeholder-ready Markdown report

```python
from mlframe.training.composite import CompositeSpec, report_to_markdown

spec_dicts = metadata["composite_target_specs"]["regression"]["TVT"]
spec_objs = [
    CompositeSpec(**{k: v for k, v in s.items()
                     if k in {"name", "target_col", "transform_name",
                              "base_column", "fitted_params",
                              "mi_gain", "mi_y", "mi_t",
                              "valid_domain_frac", "n_train_rows"}})
    for s in spec_dicts
]
md = report_to_markdown(
    target_col="TVT",
    specs=spec_objs,
    failures=metadata["composite_target_failures"]["regression"]["TVT"],
    ensemble_metadata=metadata["composite_target_ensemble"]["regression"]["TVT"],
)
print(md)
# Markdown table of specs with mi_gain / valid_frac
# + per-spec audit-trail paragraph (forward formula, justification)
# + rejected candidates table
# + cross-target ensemble component weights
```

## Production rollback / kill switch

If composite mode breaks production, set the env var BEFORE the
`train_mlframe_models_suite` call. The suite reads it at config-resolution
time and forces `enabled=False` regardless of the config object the
caller passed.

```bash
export MLFRAME_DISABLE_COMPOSITE=1
python my_training_script.py
```

Roll back in seconds without a code change. Useful when a downstream
consumer (a serving layer, a dashboard, a regression test) trips on
the new metadata schema and you need to ship a hotfix.

## When NOT to enable composite mode

The `BaselineDiagnostics` recommendation is the canonical signal:

| recommendation | meaning | action |
|---|---|---|
| `high_potential` | Dominant feature exists AND init_score baseline doesn't fully extract it | Enable composite (Tier 1+) |
| `marginal` | Some dominance but small; init_score may capture much of it | A/B test composite vs raw on a held-out set |
| `unlikely_to_help` | No dominant feature OR init_score already matches raw | Stay at Tier 0; `auto_skip_on_baseline_optimal=True` makes this automatic |
| `skipped` | target_type unsupported (multiclass / LtR / quantile) | Composite not applicable |

## Per-target-type behaviour

| target_type | Composite-mode behaviour |
|---|---|
| `REGRESSION` | Full support: discovery, wrapper, ensemble, OOF, all strategies |
| `BINARY_CLASSIFICATION` | `BaselineDiagnostics` runs (default ON) AND now produces a meaningful `init_score_baseline` via logit offset (LR-combined top-K dominant features -> sigmoid -> logit -> LightGBM init_score). The composite-discovery + wrapper layers themselves still skip with `binary_classification_unsupported_init_score_logit_offset` -- but the actionable diagnostic (init_score baseline AUC vs raw AUC) is delivered. |
| `MULTICLASS_CLASSIFICATION` | Skipped with `multiclass_unsupported_no_residual_semantics` |
| `MULTILABEL_CLASSIFICATION` | Skipped with `multilabel_classification_unsupported` |
| `LEARNING_TO_RANK` | Skipped with `ltr_unsupported_pairwise_breaks_with_residual` |
| `QUANTILE_REGRESSION` | Skipped with `quantile_regression_unsupported_per_quantile_inverse_undefined` |

### Binary classification: actionable use of the init_score baseline

The `init_score_baseline` field of the diagnostic IS the actionable
output for binary tasks. Read it like this:

```python
diag = metadata["baseline_diagnostics"]["binary_classification"]["my_target"]
isb = diag["init_score_baseline"]
if isb is not None:
    # Native LightGBM/XGB init_score offset gives AUC=isb["metric"]
    # vs raw AUC=diag["headline_metric"]["value"].
    # If isb is within ~1% of raw, you can deploy the native
    # init_score path directly:
    #   lgb.LGBMClassifier(...).fit(X, y, init_score=logit(prior_p))
    # ...without needing the full composite-discovery layer.
    print(f"raw AUC = {diag['headline_metric']['value']:.4f}")
    print(f"init_score AUC = {isb['metric']:.4f} (using {isb['feature_used']!r})")
    print(f"delta = {isb['delta_vs_raw_pct']:+.2f}%")
```

If the init_score baseline matches raw within `init_score_optimal_threshold_pct`
(default 1%), the recommendation will be `unlikely_to_help` -- meaning
"native init_score offset is enough; don't over-engineer with composite
wrapping". You ship the simpler init_score path and skip the composite
infrastructure.

## Benchmarks + profiling

```bash
# Run the synthetic benchmark suite (S1-S16 scenarios, JSON + Markdown leaderboard)
python benchmarks/composite_target_benchmark.py --fast

# Profile every composite-target feature with cProfile + wall-time calibration
python benchmarks/composite_profile.py --feature all --reps 5
python benchmarks/composite_profile.py --feature wrapper_predict
```

## End-to-end example: TVT walkthrough

A full Jupyter walkthrough on synthetic TVT-style data is at
[`docs/composite_targets_tutorial.ipynb`](../composite_targets_tutorial.ipynb).
16 cells covering data synthesis, baseline diagnostics, composite opt-in,
spec inspection, baseline-vs-ensemble RMSE comparison, and the
`report_to_markdown` rendering.
