"""FS_BORUTA_ROOT-1 (2026-08-05 audit): ``TentativeRoughFix`` must exclude the phantom row-0 initializer
from ``self.history_x`` before computing the tentative-vs-shadow median comparison.

``create_importance_history`` seeds ``self.history_x = np.zeros(self.ncols)`` (a single all-zero row)
BEFORE the trial loop runs; each trial then ``np.vstack``s its real per-trial z-scored importances onto it,
so ``self.history_x`` ends up with N+1 rows for N trials -- row 0 is not a real trial. ``_io_plot.py``'s
``results_to_csv``/plot already strip this phantom row via ``.iloc[1:]``, but ``TentativeRoughFix`` read
``self.history_x`` directly, splicing a literal 0.0 into both the tentative feature's and the shadow's
median -- since importances are z-scored (normalize=True, the default) around 0, this measurably biases
the medians toward 0 and can flip the accept/reject decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class _DummyXGBLike:
    """Stand-in whose ``type(...).__name__`` contains ``XGB``; keeps ``check_model`` happy without a real fit."""

    feature_importances_ = np.zeros(2)

    def fit(self, X: Any, y: Any, **kw: Any) -> "_DummyXGBLike":
        """No-op fit stub; returns self unchanged."""
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Returns ``np.zeros(len(X))``."""
        return np.zeros(len(X))


def test_tentative_rough_fix_excludes_phantom_row0_from_median():
    """Row 0 (phantom zero-init) must not participate in the tentative-vs-shadow median comparison.

    Fixture: a 2-real-trial history where including the phantom row-0 flips the accept/reject decision
    relative to excluding it -- median(f1[1:])=median([0.05,0.5])=0.275 > median(Max_Shadow[1:])=
    median([0.1,0.2])=0.15 -> ACCEPT (correct); median(f1)=median([0,0.05,0.5])=0.05 <
    median(Max_Shadow)=median([0,0.1,0.2])=0.1 -> REJECT (the pre-fix, phantom-row-biased outcome).
    """
    from mlframe.feature_selection.boruta_shap import BorutaShap

    sel = BorutaShap(
        model=_DummyXGBLike(),
        importance_measure="gini",
        classification=True,
        n_trials=1,
        random_state=0,
        verbose=False,
    )
    sel.history_x = pd.DataFrame({"f1": [0.0, 0.05, 0.5], "Max_Shadow": [0.0, 0.1, 0.2]})
    sel.tentative = ["f1"]
    sel.rejected = []
    sel.accepted = []

    sel.TentativeRoughFix()

    assert list(sel.accepted) == ["f1"], f"expected f1 to be ACCEPTED once the phantom row-0 is excluded, got accepted={sel.accepted!r} rejected={sel.rejected!r}"
    assert len(sel.rejected) == 0, f"expected nothing rejected, got rejected={sel.rejected!r}"
