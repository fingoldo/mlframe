# Honest diagnostics guide (AP13)

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

The aggregator returns a ``dict`` of the following shape (one entry per trained estimator):

```python
{
    "estimator_name": "lgbm__y_target_0",
    "target": "y_target_0",
    "metric": "RMSE",
    "test": {"point": 11.63, "lo": 11.21, "hi": 12.08},
    "oof":  {"point": 11.45, "lo": 11.02, "hi": 11.91},
    "honest_gap_pct": 1.55,
    "honest_gap_warn": False,
    "dummy_floor": {"name": "lag_predict", "rmse": 11.58, "delta": -0.05, "warn": True},
    "calibration": {"calibrator": "NoCal", "ece": None},  # regression -> None
    "provenance": {
        "splitter": "GroupKFold(n=5)",
        "feature_selection": "MRMR(k=auto)",
        "ensemble_flavour": "CT_ENSEMBLE_NNLS",
        "seed": 42,
    },
    "delong": None,  # filled only for classification top-2 pairs
}
```

The structured dict is rendered into a one-pager text table at the bottom of the standard reporting output, and persisted to ``honest_diagnostics.json`` next to the model bundle if ``ReportingConfig.persist_honest_diagnostics_json`` is True (default).

## Configuration

```python
from mlframe.training.configs import ReportingConfig

config = ReportingConfig(
    honest_estimator_diagnostics=True,                    # AP13 default ON
    honest_gap_warn_threshold=0.10,                       # default 10%
    honest_diagnostics_n_bootstrap=1000,                   # default
    honest_diagnostics_alpha=0.05,                         # 95% CI
    honest_diagnostics_delong_top_k=2,                     # top-2 pairs per slot
    persist_honest_diagnostics_json=True,                  # default
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
- **Bootstrap CI cost**: ~50-200ms per estimator at default ``n_bootstrap=1000``. For very large suites (20+ targets × 5 classifiers) this adds ~10-20s to finalize. Reduce ``honest_diagnostics_n_bootstrap`` to 200 if wall-time matters.

## Related modules

- ``src/mlframe/training/honest_diagnostics.py`` — the aggregator.
- ``src/mlframe/evaluation/bootstrap.py`` — ``bootstrap_metric`` + ``delong_test``.
- ``src/mlframe/calibration/quality.py`` — ``pick_best_calibrator``. See ``docs/calibration_policy.md``.
- ``src/mlframe/training/provenance.py`` — provenance trail (AP14).
- ``src/mlframe/training/dummy_baselines.py`` — dummy floor producers. See ``docs/dummy_baselines_guide.md``.

## Sensors

- ``tests/training/test_honest_diagnostics_aggregator.py`` — covers schema, gap-warn threshold, dummy-floor delta, calibration ECE wiring, DeLong invocation.
- ``tests/evaluation/test_bootstrap.py`` — covers underlying ``bootstrap_metric`` + ``delong_test``.

## Commit reference

Landed in commit ``58586198`` (W9A ``feat(training): honest_diagnostics aggregator + ReportingConfig.honest_estimator_diagnostics default ON``) on 2026-05-24.
