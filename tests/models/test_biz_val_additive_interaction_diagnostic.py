"""biz_value test for ``models.additive_interaction_diagnostic``.

The win: on a purely additive dataset (no true feature interactions), the ``num_leaves=2`` additive-only
model should nearly match the full model's CV score (high ``additive_signal_ratio``, no interaction
engineering recommended). On a purely interaction-driven dataset (target = product of two features, zero
marginal/additive signal), the additive model should score far worse than the full model (low/negative
ratio, interaction engineering correctly recommended) -- the diagnostic must actually distinguish the two
regimes, not just report a number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.models.additive_interaction_diagnostic import additive_interaction_diagnostic


def test_biz_val_additive_interaction_diagnostic_distinguishes_additive_from_interaction_signal():
    rng = np.random.default_rng(0)
    n = 3000

    X_additive = rng.normal(0, 1, (n, 4))
    y_additive = X_additive[:, 0] * 2 + np.sin(X_additive[:, 1] * 2) + X_additive[:, 2] ** 2 + rng.normal(0, 0.2, n)

    X_interaction = rng.normal(0, 1, (n, 4))
    y_interaction = X_interaction[:, 0] * X_interaction[:, 1] * 3 + rng.normal(0, 0.2, n)

    splits = list(KFold(5, shuffle=True, random_state=0).split(X_additive))

    result_additive = additive_interaction_diagnostic(X_additive, y_additive, splits, metric_fn=r2_score, objective="regression")
    result_interaction = additive_interaction_diagnostic(X_interaction, y_interaction, splits, metric_fn=r2_score, objective="regression")

    assert result_additive["additive_signal_ratio"] > 0.9, result_additive
    assert result_additive["recommend_interaction_engineering"] is False

    assert result_interaction["additive_signal_ratio"] < 0.5, result_interaction
    assert result_interaction["recommend_interaction_engineering"] is True

    assert result_additive["additive_signal_ratio"] > result_interaction["additive_signal_ratio"]
