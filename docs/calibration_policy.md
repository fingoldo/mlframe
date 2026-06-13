# Calibration policy guide (AP12)

Probability calibration in mlframe is governed by ``pick_best_calibrator`` (``src/mlframe/calibration/policy.py``), a small policy that selects between ``Sigmoid`` (Platt scaling), ``Isotonic`` (PAV), and ``NoCal`` (pass-through) by comparing **out-of-fold Expected Calibration Error (ECE)** with bootstrap confidence intervals.

## When this fires

The policy is consulted at end-of-training, after the OOF predictions for every classifier in the ensemble are available. Defaults are picked so the policy works out-of-the-box without requiring user configuration.

## Signature

```python
from mlframe.calibration.policy import pick_best_calibrator

result = pick_best_calibrator(
    probs, y,                 # optional diagnostic-only held-out probs/labels (not used for the decision)
    oof_probs, oof_y,         # OOF probs/labels that drive the selection
    candidates=None,          # default: all of NoCal / Sigmoid / Isotonic
    n_bootstrap=...,          # bootstrap reps for the ECE CI
    n_bins=10,                # ECE bin count (DEFAULT_ECE_NBINS)
    random_state=0,
)
# -> {"chosen": <name>, "ece_mean": float, "ece_ci": (lo, hi),
#     "alternatives": {...}, "rule": <selection-rule>, "n_oof": int, "plot_path": Optional[str]}
```

ECE itself is computed by ``_ece_score(y_true, p_pred, n_bins=15)`` in the same module; the full Brier reliability / resolution / uncertainty decomposition is available via ``mlframe.metrics.core.compute_ece_and_brier_decomposition``.

## Decision rule

Given OOF ``oof_y`` and ``oof_probs`` for each candidate calibrator C in ``{NoCal, Sigmoid, Isotonic}``:

1. Compute ``ece_C = _ece_score(oof_y, calibrated_probs_C, n_bins=15)``.
2. Bootstrap a CI on ``ece_C``.
3. Pick the calibrator whose **upper CI bound** is lowest. This is the conservative choice: prefer the calibrator that statistically dominates on calibration without overfitting to a single OOF realisation.
4. Tie-breaker (overlapping CIs): prefer ``NoCal > Sigmoid > Isotonic`` (Occam — fewer learned parameters wins on ties).

## Why ECE-with-CI rather than point ECE

Point-ECE on n<~5000 OOF rows is noisy; calibrators that fit a few extra parameters (Isotonic) routinely show a 0.5-1pp improvement that washes out under resampling. Bootstrap CI exposes that noise so the policy doesn't repeatedly flip between calibrators on minor data changes.

## Reliability plot

A reliability plot can be emitted by ``pick_best_calibrator`` when ``emit_plot=True`` (with an optional ``plot_path=``); the resolved path is returned under the ``plot_path`` key of the result dict. The plot overlays the candidate calibration curves alongside the ideal diagonal and annotates the chosen calibrator.

## Limitations

- **Binary classification only** in the current implementation. Multiclass calibration is one-vs-rest in the underlying sklearn classes but the policy does not currently produce per-class CIs; pinning ``NoCal`` is the safer default for multiclass until that is wired.
- **Requires OOF predictions**. Classifiers trained without CV (i.e. with ``train_eval_score=False`` shortcuts) cannot be calibrated by this policy — the calibrator falls back to ``NoCal`` and a WARN is logged.
- **n<200 rows**: the bootstrap CI is wide enough that the upper-bound tiebreaker effectively always picks ``NoCal``. This is by design — there is not enough data to fit a calibrator meaningfully on tiny samples.

## Related modules

- ``src/mlframe/calibration/policy.py`` — ``pick_best_calibrator``, ``_ece_score``, the reliability-plot emitter.
- ``src/mlframe/metrics/core.py`` — ``compute_ece_and_brier_decomposition`` (Brier REL/RES/UNC decomposition).
- ``src/mlframe/calibration/post.py`` — post-hoc ``Sigmoid``/``Isotonic`` wrappers around the chosen calibrator.
- ``src/mlframe/evaluation/bootstrap.py`` — generic ``bootstrap_metric`` + ``delong_test`` consumed by the policy.
- ``src/mlframe/training/honest_diagnostics.py`` — aggregator that surfaces the calibration verdict alongside the rest of the honest-OOF diagnostics. See ``docs/honest_diagnostics_guide.md``.

## Sensors

- ``tests/calibration/test_pick_best_calibrator_policy.py`` — covers the bootstrap-CI selection logic, tie-breaker behaviour, ``NoCal``/``Sigmoid``/``Isotonic`` ordering, multiclass fallback.
- ``tests/evaluation/test_bootstrap.py`` — covers the generic ``bootstrap_metric`` helper that the policy delegates to.

## Commit reference

Landed in commit ``783eae4c`` (W9A ``feat(calibration): pick_best_calibrator policy with OOF ECE + bootstrap CI``) on 2026-05-24.
