# Honest diagnostics guide

The ``honest_diagnostics`` aggregator (``src/mlframe/training/honest_diagnostics.py``) consolidates the audit-trail diagnostics that previously lived as scattered per-target logs into a single structured report. It is wired into ``finalize`` and default ON via ``ReportingConfig.honest_estimator_diagnostics``.

## What it aggregates

For every trained estimator in the suite, the aggregator emits:

1. **Top-line metric with 95% bootstrap CI** — ``RMSE`` (regression) or ``ROC-AUC`` / ``log_loss`` (classification), with percentile CI from ``evaluation.bootstrap.bootstrap_metric`` (``n_bootstrap=1000``).
2. **OOF metric vs test metric delta** — exposes optimistic-OOF / honest-holdout gap. Configurable threshold WARNs when the gap exceeds a fraction of the metric's typical scale (default 10%).
3. **Dummy-baseline delta** — improvement vs the strongest parameter-free dummy (median / mean / mode / RandomGuess / lag_predict / AR(1)-failsafe). Negative delta = the trained model is worse than free dummies on honest holdout, which trips a hard WARN and pushes the estimator below the dummy-floor gate.
4. **Calibration ECE** — for classifiers, the OOF ECE under the policy-chosen calibrator (see ``docs/calibration_policy.md``).
5. **Provenance trail** — the hyperparams that drove the trained instance, sourced from ``training/provenance.py``. Includes seeds, splitter family, calibrator policy, feature-selection mode, ensemble flavour.
6. **DeLong p-value for AUC pairwise** (classification only) — for the top-2 classifiers in each slot, runs ``evaluation.bootstrap.delong_test`` so the user knows whether the win is statistically significant or noise.

## Why "honest"

The aggregator is opinionated about which numbers count: **only honest-holdout metrics drive verdicts**. Validation-set metrics are biased upward (the early-stopping detector saw them) and are stamped as such in the report but do not gate ensemble inclusion. OOF metrics are used only when no honest holdout exists (small-n regime, ``trainset_only`` mode).

This mirrors the project memory rule ``feedback_ml_val_test_oof_terminology``: val=ES detector (biased), test/OOS/holdout=honest estimate (model never saw), OOF=CV test-analog.

## Output shape

The entry point is ``run_honest_diagnostics(ctx, models, metadata=None) -> dict``, called from the suite's ``finalize`` phase and stamped into ``metadata["honest_diagnostics"]``. It returns a single ``dict`` with four artefact blocks keyed by ``"{target_type}/{target_name}/{model_name}"`` (not one row per estimator at the top level):

```python
{
    "ts": "2026-07-06T02:44:10+00:00",
    "bootstrap_ci": {
        "regression/y_target_0/lgbm": {
            "rmse": {"point": 11.63, "ci_lo": 11.21, "ci_hi": 12.08},
        },
        # classification entries instead carry "roc_auc" / "brier" / "log_loss" / "ece",
        # each as {"point": ..., "ci_lo": ..., "ci_hi": ...}; an entry with no
        # test_target/test_probs on the model becomes {"status": "skipped", "reason": "..."}
    },
    "drift_psi": {
        "status": "skipped",  # or "ok" with a per-column PSI table, when ctx.train_df/val_df/test_df are set
        "reason": "ctx.train_df is None",
    },
    "calibration": {
        "regression/y_target_0/lgbm": {
            "status": "skipped",
            "reason": "no oof_probs on model entry",
            "probs_posthoc_calibrated": None,
        },
    },
    "provenance": {
        "status": "ok",
        "n_steps": 0,
        "table": "(no provenance recorded)",
        "raw": {},
    },
    "reports_dir": None,  # set when ctx.data_dir / ctx.models_dir are configured
}
```

Each model entry passed in ``models`` (``{target_type: {target_name: [model_entry, ...]}}``) needs ``test_target`` + ``test_probs`` for the bootstrap block, and ``oof_probs`` for the calibration block; missing attributes degrade that block to a `"status": "skipped"` entry rather than raising.

## Configuration

The single knob is the on/off toggle ``ReportingConfig.honest_estimator_diagnostics`` (default ON):

```python
from mlframe.training.configs import ReportingConfig

config = ReportingConfig(
    honest_estimator_diagnostics=True,   # default ON
)
```

To disable entirely:

```python
config = ReportingConfig(honest_estimator_diagnostics=False)
```

## Reading the report

A representative table row for a regression target with a lag_predict floor:

```
target=y_target_0  estimator=lgbm  metric=RMSE
  test=11.63 [11.21, 12.08]   oof=11.45 [11.02, 11.91]
  honest_gap=+1.55% (ok)
  dummy_floor=lag_predict(11.58)  delta=-0.05  WARN: trained model loses to lag-predict on honest holdout
  provenance: GroupKFold(n=5) + MRMR(k=auto) + CT_ENSEMBLE_NNLS, seed=42
```

The ``WARN`` line is the actionable signal: the ensemble gate will drop this estimator from the final blend (per ``ct_ensemble_dummy_floor_enabled``, see ``docs/dummy_baselines_guide.md``).

## Limitations

- **Multiclass calibration ECE**: surfaced as ``None`` pending the multiclass calibration policy fix-up (Wave-10+).
- **DeLong test requires binary** ``y_true``. Multiclass classification gets per-class one-vs-rest DeLong only if the caller explicitly asks; default behaviour skips DeLong for multiclass.
- **Bootstrap CI cost**: ~50-200ms per estimator. For very large suites (20+ targets × 5 classifiers) this adds ~10-20s to finalize; disable the aggregator (``honest_estimator_diagnostics=False``) if wall-time matters.

## Related modules

- ``src/mlframe/training/honest_diagnostics.py`` — the aggregator.
- ``src/mlframe/evaluation/bootstrap.py`` — ``bootstrap_metric`` + ``delong_test``.
- ``src/mlframe/calibration/policy.py`` — ``pick_best_calibrator``. See ``docs/calibration_policy.md``.
- ``src/mlframe/training/provenance.py`` — provenance trail (AP14).
- ``src/mlframe/training/dummy_baselines.py`` — dummy floor producers. See ``docs/dummy_baselines_guide.md``.

## Sensors

- ``tests/training/test_honest_diagnostics_aggregator.py`` — covers schema, gap-warn threshold, dummy-floor delta, calibration ECE wiring, DeLong invocation.
- ``tests/evaluation/test_bootstrap.py`` — covers underlying ``bootstrap_metric`` + ``delong_test``.

## Commit reference

Landed in commit ``58586198`` (W9A ``feat(training): honest_diagnostics aggregator + ReportingConfig.honest_estimator_diagnostics default ON``) on 2026-05-24.
