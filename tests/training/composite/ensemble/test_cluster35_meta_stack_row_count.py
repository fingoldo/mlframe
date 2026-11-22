"""Audit 2026-11-20 finding 35-F51: n_oof_rows counts the rows the meta-stacker fit used.

That is finite in y AND in every OOF column. A row with a NaN in one component column was counted before, though the fit dropped it.
"""

import numpy as np

from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble
from mlframe.training.composite.ensemble._stackers import build_meta_stack_ensemble


class _Stub:
    """A component placeholder; the stacker never predicts through it in this test."""


def test_row_count_excludes_rows_with_a_nan_component():
    """One NaN in y and one NaN in a component column leave 58 of 60 rows, and the note says 58."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([1.0, 0.5, -0.3]) + 0.05 * rng.normal(size=60)
    X[5, 1] = np.nan
    y[9] = np.nan
    ens = build_meta_stack_ensemble(CompositeCrossTargetEnsemble, [_Stub()] * 3, ["a", "b", "c"], X, y, stacker="ridge")
    assert ens.notes["n_oof_rows"] == 58
