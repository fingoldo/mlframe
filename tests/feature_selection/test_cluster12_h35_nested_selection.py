"""Cluster 12 regression: H35 picks its config on a held-out part, not on the rows its members were fit on (audit 2026-11-20, finding 12-F34).

H35.fit first fits the base selector on 75% of the rows, picks (vote, expand) by AUC on the remaining 25% (whose features are built through the
augment path), then refits on all rows with the chosen config. The base fit must therefore run twice, first on the training part only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid"


def test_h35_fits_members_on_a_training_part_first():
    """The base fit sees 75% of the rows first, then all rows for the final refit (12-F34)."""
    pytest.importorskip("lightgbm")
    sys.path.insert(0, str(BENCH))
    rb = pytest.importorskip("round3_hybrid_refine_bench")  # the bench needs its sibling helpers on the path
    from mlframe.feature_selection.hybrid_selector import HybridSelector

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = (X["f0"].to_numpy() + 0.5 * rng.normal(size=n) > 0).astype(int)
    lengths = []
    original = HybridSelector.fit

    def _spy(self, Xf, yf):
        """Record the row count of every base fit, then delegate."""
        lengths.append(len(Xf))
        return original(self, Xf, yf)

    HybridSelector.fit = _spy
    try:
        sel = rb.H35().fit(X, y)
    finally:
        HybridSelector.fit = original
    assert lengths[0] == 150 and lengths[-1] == n
    assert len(sel.raw_selected_) >= 1
