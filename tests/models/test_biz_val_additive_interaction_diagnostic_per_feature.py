"""biz_value test for the opt-in per-feature LOFO decomposition of ``additive_interaction_diagnostic``.

The win: given a dataset where only a KNOWN pair of features (0 and 1) drives a pure interaction term
(``X0 * X1``, zero marginal/additive signal on its own -- same construction as the module's base biz_value
test) while three other features (2, 3, 4) are purely additive, the ``per_feature_report=True`` output must
rank features 0 and 1 at the top of ``interaction_contribution`` -- i.e. it must correctly ISOLATE the
interacting features from the additive ones, not just report a global ratio.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.models.additive_interaction_diagnostic import additive_interaction_diagnostic


def test_biz_val_additive_interaction_diagnostic_per_feature_isolates_interacting_pair():
    """Biz val additive interaction diagnostic per feature isolates interacting pair."""
    rng = np.random.default_rng(0)
    n = 1500

    X = rng.normal(0, 1, (n, 5))
    y = (
        4.0 * X[:, 0] * X[:, 1]  # pure interaction, zero marginal effect for features 0 and 1
        + 1.5 * X[:, 2]
        + np.sin(X[:, 3] * 2)
        + 0.8 * X[:, 4] ** 2
        + rng.normal(0, 0.2, n)
    )

    splits = list(KFold(3, shuffle=True, random_state=0).split(X))
    feature_names = [f"f{i}" for i in range(5)]

    result = additive_interaction_diagnostic(
        X,
        y,
        splits,
        metric_fn=r2_score,
        objective="regression",
        per_feature_report=True,
        per_feature_names=feature_names,
    )

    assert "per_feature_interaction_report" in result
    report = result["per_feature_interaction_report"]
    assert len(report) == 5

    # descending sort by interaction_contribution must hold.
    contributions = [row["interaction_contribution"] for row in report]
    assert contributions == sorted(contributions, reverse=True)

    top2_features = {report[0]["feature"], report[1]["feature"]}
    known_interacting = {"f0", "f1"}
    top2_precision = len(top2_features & known_interacting) / 2.0
    assert top2_precision == 1.0, report

    # the interacting features' contribution must clearly separate from the purely-additive ones.
    interacting_min = min(row["interaction_contribution"] for row in report if row["feature"] in known_interacting)
    additive_max = max(row["interaction_contribution"] for row in report if row["feature"] not in known_interacting)
    assert interacting_min > additive_max, report
