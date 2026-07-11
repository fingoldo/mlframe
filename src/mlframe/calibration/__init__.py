"""Probability calibration quality and post-hoc calibrators.

The ``post`` submodule pulls heavy optional deps (netcal, pycalib,
ml_insights, betacal, venn-abers); it is NOT eager-imported. Use:

    from mlframe.calibration.post import ...

Submodules:
    quality        - calibration quality assessment (reliability diagrams, ECE, ...).
    post           - post-hoc calibration methods (isotonic, Platt, beta, Venn-Abers, ...).
    probabilities  - probability transformations and diagnostics.

iter631: ``quality`` is NOT eager-imported. quality.py cascades through
matplotlib + properscoring + sklearn (~2s per process); the suite reaches
calibration symbols via ``mlframe.calibration.policy`` (the honest-diagnostics
ECE wrapper) which loads its own narrow deps. PEP 562 ``__getattr__`` lazy-
resolves both ``quality`` and ``probabilities`` symbols on first attribute
access so external ``from mlframe.calibration import X`` callers keep
working at the cost of a one-time deferred import.
"""

from __future__ import annotations

from typing import Any

from mlframe.calibration.ensembling import odds_ratio_combine
from mlframe.calibration.isotonic_risk import isotonic_overfit_risk
from mlframe.calibration.confidence_shrinkage import compute_oof_confidence, apply_confidence_shrinkage
from mlframe.calibration.threshold_optimizer import optimize_decision_threshold, apply_decision_threshold
from mlframe.calibration.group_bias_correction import fit_group_bias_correction, apply_group_bias_correction
from mlframe.calibration.smoothed_override import apply_smoothed_override
from mlframe.calibration.asymmetric_rescale import fit_asymmetric_rescale, apply_asymmetric_rescale
from mlframe.calibration.group_zero_sum_constraint import apply_group_zero_sum_constraint
from mlframe.calibration.sticky_state_persistence_floor import apply_sticky_state_persistence_floor, optimize_persistence_floor
from mlframe.calibration.prediction_band_correction import find_prediction_band_shift, apply_prediction_band_correction


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # ``importlib.import_module`` bypasses ``_handle_fromlist``'s package-
    # attribute check which would otherwise re-enter __getattr__ and recurse.
    # Resolution order: quality (broader surface) first, then probabilities.
    import importlib
    _q = importlib.import_module("mlframe.calibration.quality")
    if hasattr(_q, name):
        val = getattr(_q, name)
        globals()[name] = val
        return val
    _p = importlib.import_module("mlframe.calibration.probabilities")
    if hasattr(_p, name):
        val = getattr(_p, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    import importlib
    _q = importlib.import_module("mlframe.calibration.quality")
    _p = importlib.import_module("mlframe.calibration.probabilities")
    return sorted(set(globals().keys()) | {n for n in dir(_q) if not n.startswith("_")} | {n for n in dir(_p) if not n.startswith("_")})
