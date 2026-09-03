"""FS_BORUTA_ROOT-5 (2026-08-05 audit): ``TentativeRoughFix`` must clear ``self.tentative`` after resolving
every tentative feature into ``accepted``/``rejected``.

Before this fix, ``self.tentative`` was left unchanged -- any caller reading it directly (or
``Subset(tentative=True)``) saw already-resolved features as still undecided.
"""

from __future__ import annotations

import pandas as pd


def test_tentative_rough_fix_clears_self_tentative():
    """After TentativeRoughFix, self.tentative must be empty (every feature is resolved)."""
    from mlframe.feature_selection.boruta_shap import BorutaShap

    sel = BorutaShap(model=None, importance_measure="gini", classification=True, n_trials=1, random_state=0, verbose=False)
    sel.history_x = pd.DataFrame({"f1": [0.0, 0.05, 0.5], "f2": [0.0, 0.0, 0.0], "Max_Shadow": [0.0, 0.1, 0.2]})
    sel.tentative = ["f1", "f2"]
    sel.rejected = []
    sel.accepted = []

    sel.TentativeRoughFix()

    assert list(sel.tentative) == [], f"expected self.tentative to be cleared, got {sel.tentative!r}"
    assert set(sel.accepted) | set(sel.rejected) == {"f1", "f2"}
