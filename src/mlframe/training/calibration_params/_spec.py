"""Which calibration-step callables get a generated strict-parameter model.

``ThresholdOptimizerParams`` / ``IsotonicRiskParams`` / ``ConfidenceShrinkageParams`` are rendered from the signatures of
``optimize_decision_threshold`` / ``isotonic_overfit_risk`` / ``apply_confidence_shrinkage`` with ``pyutilz.dev.signature_models``, so the
``*_kwargs`` the finalize phase forwards cannot drift from what those functions accept (``tests/training/test_calibration_params_in_sync.py``).
The data arguments the finalize phase supplies itself are excluded.
"""

from __future__ import annotations

from typing import Tuple

from .._param_model_spec import SelectorSpec

SPECS: Tuple[SelectorSpec, ...] = (
    SelectorSpec(
        "threshold_optimizer", "ThresholdOptimizerParams", "mlframe.calibration.threshold_optimizer", "optimize_decision_threshold",
        # ``metric_fn`` is passed by the finalize phase itself (balanced accuracy unless the caller supplies one), so the config adds it as an
        # optional field instead of the signature's required parameter.
        exclude=("y_true", "y_proba", "metric_fn"),
    ),
    SelectorSpec("isotonic_risk", "IsotonicRiskParams", "mlframe.calibration.isotonic_risk", "isotonic_overfit_risk", exclude=("calib_p", "calib_y")),
    SelectorSpec(
        "confidence_shrinkage", "ConfidenceShrinkageParams", "mlframe.calibration.confidence_shrinkage", "apply_confidence_shrinkage",
        exclude=("preds", "confidences"),
    ),
)
