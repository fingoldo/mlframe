"""Cluster 41 regression: scale_pos_weight is written only for binary targets (audit 2026-11-20, finding 41-F33).

XGBoost ignores ``scale_pos_weight`` for multiclass, so writing it for every imbalanced target was a no-op knob. The other imbalance knobs
(class_weight, auto_class_weights) are unchanged.
"""

import numpy as np

from mlframe.training.targets._target_distribution_analyzer_classif import _classification_balance_pathologies


def _run(counts):
    """Run the class-balance check on per-class counts and return the knob overrides."""
    knobs: dict = {}
    _classification_balance_pathologies(
        np.asarray(counts, dtype=float),
        {},
        [],
        int(np.sum(counts)),
        knobs,
        lambda *a, **k: None,
        list(range(len(counts))),
    )
    return knobs


def test_multiclass_imbalance_does_not_set_scale_pos_weight():
    """A three-class imbalanced target sets no scale_pos_weight (41-F33)."""
    knobs = _run([900, 50, 50])
    assert "scale_pos_weight" not in knobs.get("xgb_kwargs", {})
    assert knobs["lgb_kwargs"]["class_weight"] == "balanced"


def test_binary_imbalance_still_sets_scale_pos_weight():
    """Control: a two-class imbalanced target keeps scale_pos_weight (41-F33)."""
    knobs = _run([900, 50])
    assert knobs["xgb_kwargs"]["scale_pos_weight"] > 1.0
