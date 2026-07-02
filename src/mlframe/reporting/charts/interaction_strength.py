"""Friedman-Popescu H-statistic interaction heatmap: a model-diagnostics panel beside PDP/ICE.

Reuses the same PDP machinery the PDP/ICE panels use (``mlframe.inspection.pairwise_interaction_strength`` is built on
``compute_pdp`` / ``compute_pdp_2d``), so it belongs with the other model-explanation charts. The panel shows the symmetric
``(F, F)`` matrix of pairwise interaction strengths in [0, 1]; the largest off-diagonal cells are the feature pairs whose
joint model effect is most non-additive - exactly the pairs worth an explicit engineered interaction feature.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec

# H-statistic is O(F^2) 2-D PDP surfaces; cap the feature count so the panel stays bounded on wide frames.
DEFAULT_MAX_INTERACTION_FEATURES: int = 8


def interaction_strength_panel(
    model: Any,
    X: Any,
    features: Sequence[Union[int, str]],
    *,
    grid: int = 20,
    sample: int = 2000,
    seed: int = 0,
) -> HeatmapPanelSpec:
    """HeatmapPanelSpec of the pairwise Friedman-Popescu H-statistic (interaction strength in [0, 1]) over ``features``."""
    from mlframe.inspection import pairwise_interaction_strength

    feats = list(features)
    M = pairwise_interaction_strength(model, X, feats, grid=grid, sample=sample, seed=seed)
    labels = tuple(str(f) for f in feats)
    return HeatmapPanelSpec(
        matrix=M,
        row_labels=labels,
        col_labels=labels,
        title="Pairwise interaction strength (Friedman-Popescu H)",
        colormap="magma",
        cell_text=M,
        text_format=".2f",
        colorbar_label="H (0 additive .. 1 pure interaction)",
    )


def compose_interaction_strength_figure(
    model: Any,
    X: Any,
    features: Sequence[Union[int, str]],
    *,
    max_features: int = DEFAULT_MAX_INTERACTION_FEATURES,
    grid: int = 20,
    sample: int = 2000,
    seed: int = 0,
    suptitle: str = "Feature interaction strength",
) -> FigureSpec:
    """One-panel FigureSpec wrapping :func:`interaction_strength_panel`, capping to the top ``max_features`` features."""
    top = list(features)[: max(2, int(max_features))]
    panel = interaction_strength_panel(model, X, top, grid=grid, sample=sample, seed=seed)
    return FigureSpec(suptitle=suptitle, panels=((panel,),), figsize=(6.0, 5.0))


__all__ = [
    "DEFAULT_MAX_INTERACTION_FEATURES",
    "interaction_strength_panel",
    "compose_interaction_strength_figure",
]
