"""Opt-in, cost-gated training diagnostics that require extra model fits.

Unlike the cheap ``baselines`` diagnostics (one quick fit), the helpers here
deliberately retrain K times (a learning curve is K refits by construction),
so each is OFF by default behind its own config and wired by the integrator
only when the operator asks for it.
"""
from __future__ import annotations

from mlframe.training.diagnostics.learning_curve import (
    LearningCurveConfig,
    LearningCurveResult,
    compute_learning_curve,
    learning_curve_panel,
)

__all__ = [
    "LearningCurveConfig",
    "LearningCurveResult",
    "compute_learning_curve",
    "learning_curve_panel",
]
