"""The exhaustive interaction pass scores subsets of cardinality at most 3, not every subset of a narrow feature set.

With P=16 features and no ``max_card`` the old pass scored all 65,535 subsets, each copying an n x r x r slice of the interaction tensor.
The bounded pass scores the 696 subsets of size 1 to 3, and the greedy pass that always runs covers larger interacting sets.
"""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions as mod


def test_exhaustive_interaction_pass_scores_only_small_subsets(monkeypatch):
    """The number of subset-loss calls at P=16 stays far below the 65,535 subsets of the full powerset."""
    rng = np.random.default_rng(4)
    n, P = 60, 16
    Phi = rng.normal(size=(n, P, P))
    Phi = (Phi + np.transpose(Phi, (0, 2, 1))) / 2.0
    base = np.zeros(n)
    y = rng.integers(0, 2, n).astype(float)

    calls = []
    real_loss = mod.interaction_subset_loss

    def _counting_loss(*args, **kwargs):
        """Count each subset-loss call and delegate to the real scorer."""
        calls.append(1)
        return real_loss(*args, **kwargs)

    monkeypatch.setattr(mod, "interaction_subset_loss", _counting_loss)
    mod.interaction_top_n(Phi, base, y, classification=True, metric="logloss", top_n=5, exhaustive_max=16)

    assert len(calls) < 2000, len(calls)
