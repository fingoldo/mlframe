"""The CMIM and triplet-cross candidate scorers stop at MRMR.fit's FE deadline instead of scoring every candidate.

Both ran without consulting the deadline, so max_runtime_mins did not bound them: a fit given a 0.6 minute budget sat in these
scorers until the 10 minute test watchdog fired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _orthogonal_cmim_fe as cmim
from mlframe.feature_selection.filters import _orthogonal_triplet_fe as triplet
from mlframe.feature_selection.filters._fe_deadline import fe_deadline_scope
from timeit import default_timer as timer


def _frame(n: int = 4000):
    """Four standard-normal columns and a binary target driven by the product of the first two."""
    rng = np.random.default_rng(0)
    raw = pd.DataFrame({f"x{i}": rng.standard_normal(n) for i in range(4)})
    y = ((raw["x0"] * raw["x1"] + 0.5 * rng.standard_normal(n)) > 0).astype(int).to_numpy()
    return raw, y, rng


def test_cmim_scorer_returns_no_rows_once_the_deadline_has_passed():
    """An expired deadline stops the CMIM scorer before it scores any candidate; without one all three are scored."""
    raw, y, rng = _frame()
    eng = pd.DataFrame({"x0__sq": raw["x0"] ** 2, "x1__cube": raw["x1"] ** 3, "x2__noise": rng.standard_normal(len(raw))})
    assert len(cmim.score_features_by_cmim(raw, eng, y)) == 3
    with fe_deadline_scope(timer() - 1.0):
        assert len(cmim.score_features_by_cmim(raw, eng, y)) == 0


def test_triplet_cmi_scorer_returns_no_rows_once_the_deadline_has_passed():
    """An expired deadline stops the triplet scorer before it scores any candidate; without one both are scored."""
    raw, y, _ = _frame()
    eng = pd.DataFrame({"x0*x1*x2__t1": raw["x0"] * raw["x1"] * raw["x2"], "x0*x1*x3__t1": raw["x0"] * raw["x1"] * raw["x3"]})
    assert len(triplet.score_triplet_cross_basis_by_cmi(raw, eng, y, n_perm=19)) == 2
    with fe_deadline_scope(timer() - 1.0):
        assert len(triplet.score_triplet_cross_basis_by_cmi(raw, eng, y, n_perm=19)) == 0
