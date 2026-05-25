# Calibration policy guide (AP12)

Probability calibration in mlframe is governed by ``pick_best_calibrator`` (``src/mlframe/calibration/quality.py``), a small policy that selects between ``Sigmoid`` (Platt scaling), ``Isotonic`` (PAV), and ``NoCal`` (pass-through) by comparing **out-of-fold Expected Calibration Error (ECE)** with bootstrap confidence intervals.

## When this fires

The policy is consulted at end-of-training, after the OOF predictions for every classifier in the ensemble are available. Defaults are picked so the policy works out-of-the-box without requiring user configuration; advanced users override via ``ReportingConfig``.

## Decision rule

Given OOF ``y_true`` and ``y_pred_proba`` for each candidate calibrator C in ``{NoCal, Sigmoid, Isotonic}``:

1. Compute ``ece_C = expected_calibration_error(y_true, y_pred_proba_C, n_bins=10)``.
2. Bootstrap a 95% CI on ``ece_C`` via ``evaluation.bootstrap.bootstrap_metric`` (``n_bootstrap=1000``, percentile method).
3. Pick the calibrator whose **upper CI bound** is lowest. This is the conservative choice: prefer the calibrator that statistically dominates on calibration without overfitting to a single OOF realisation.
4. Tie-breaker (within 1 standard error): prefer ``NoCal > Sigmoid > Isotonic`` (Occam — fewer learned parameters wins on ties).

## Why ECE-with-CI rather than point ECE

Point-ECE on n<~5000 OOF rows is noisy; calibrators that fit a few extra parameters (Isotonic) routinely show a 0.5-1pp improvement that washes out under resampling. Bootstrap CI exposes that noise so the policy doesn't repeatedly flip between calibrators on minor data changes.

## Reliability plot

Every selection generates a reliability plot rendered via ``reporting/renderers/matplotlib.py``. The plot overlays the three candidate calibration curves alongside the ideal diagonal, and annotates the chosen calibrator. The plot is included in standard reporting output (controlled by ``ReportingConfig.render_calibration_plot``; default ``True``).

## Configuration

```python
from mlframe.training.configs import ReportingConfig

config = ReportingConfig(
    pick_calibrator_policy="oof_ece_bootstrap_ci",   # default
    pick_calibrator_n_bins=10,                        # default
    pick_calibrator_n_bootstrap=1000,                  # default
    pick_calibrator_alpha=0.05,                        # 95% CI default
    render_calibration_plot=True,                      # default
)
```

To opt out and pin a specific calibrator:

```python
config = ReportingConfig(pick_calibrator_policy="force_sigmoid")
# or "force_isotonic" / "force_nocal"
```

## Limitations

- **Binary classification only** in the current implementation. Multiclass calibration is one-vs-rest in the underlying sklearn classes but the policy does not currently produce per-class CIs; pinning ``NoCal`` is the safer default for multiclass until that is wired.
- **Requires OOF predictions**. Classifiers trained without CV (i.e. with ``train_eval_score=False`` shortcuts) cannot be calibrated by this policy — the calibrator falls back to ``NoCal`` and a WARN is logged.
- **n<200 rows**: the bootstrap CI is wide enough that the upper-bound tiebreaker effectively always picks ``NoCal``. This is by design — there is not enough data to fit a calibrator meaningfully on tiny samples.

## Related modules

- ``src/mlframe/calibration/quality.py`` — ``pick_best_calibrator``, ``expected_calibration_error``, ``make_custom_calibration_plot``.
- ``src/mlframe/calibration/post.py`` — post-hoc ``Sigmoid``/``Isotonic`` wrappers around the chosen calibrator.
- ``src/mlframe/evaluation/bootstrap.py`` — generic ``bootstrap_metric`` + ``delong_test`` consumed by the policy.
- ``src/mlframe/training/honest_diagnostics.py`` — aggregator that surfaces the calibration verdict alongside the rest of the honest-OOF diagnostics. See ``docs/honest_diagnostics_guide.md``.

## Sensors

- ``tests/calibration/test_pick_best_calibrator_policy.py`` — covers the bootstrap-CI selection logic, tie-breaker behaviour, ``NoCal``/``Sigmoid``/``Isotonic`` ordering, multiclass fallback.
- ``tests/evaluation/test_bootstrap.py`` — covers the generic ``bootstrap_metric`` helper that the policy delegates to.

## Commit reference

Landed in commit ``783eae4c`` (W9A ``feat(calibration): pick_best_calibrator policy with OOF ECE + bootstrap CI``) on 2026-05-24.
