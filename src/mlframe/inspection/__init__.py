"""Model-agnostic interpretation utilities (PZAD interpretability lecture).

Mirrors ``sklearn.inspection`` for the post-hoc, model-agnostic interpretation primitives that are genuinely
absent from sklearn. Currently:

    interaction - Friedman & Popescu's H-statistic: how much of a feature PAIR's joint effect on a trained
                  model is non-additive interaction (built on the suite's existing PDP machinery).

PDP / ICE panels live in ``mlframe.reporting.charts.pdp_ice``; SHAP / permutation importance are provided by the
``shap`` package and ``mlframe.feature_selection`` respectively.
"""

from __future__ import annotations

from mlframe.inspection.interaction import (
    friedman_h_statistic,
    pairwise_interaction_strength,
)

__all__ = ["friedman_h_statistic", "pairwise_interaction_strength"]
