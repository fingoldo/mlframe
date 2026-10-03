"""Auto-escalation fits order-dependent warps, so it must stay off for nominal multiclass targets and on for binary and continuous ones."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_auto_escalation import run_fe_auto_escalation
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._inputs import _is_nominal_multiclass_target
from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.mark.parametrize(
    "y, expected",
    [
        (np.tile(np.arange(4), 50), True),
        (np.tile(np.arange(3), 50).astype(np.int8), True),
        (np.tile(np.arange(2), 50), False),
        (np.tile([False, True], 50), False),
        (np.random.default_rng(0).normal(size=200), False),
        (np.arange(200), False),
        (np.zeros((200, 2), dtype=np.int64), False),
    ],
)
def test_nominal_multiclass_detection(y, expected):
    """Three to twenty distinct integer labels are nominal; binary, continuous, many-valued integer and multi-output targets are not."""
    assert _is_nominal_multiclass_target(y) is expected


def _escalate(nominal: bool):
    """Run escalation on one failed pair of an X0*X1 problem with the nominal flag set as requested and return (admitted, info)."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["x0", "x1", "x2"])
    y = np.sin(3.7 * X["x0"].to_numpy()) * X["x1"].to_numpy() + 0.1 * rng.normal(size=n)
    classes_y = np.digitize(y, np.quantile(y, np.linspace(0, 1, 11)[1:-1])).astype(np.int32)
    sel = MRMR(verbose=0, random_seed=0)
    sel.feature_names_in_ = list(X.columns)
    sel._fe_escalation_nominal_target_ = nominal
    out = run_fe_auto_escalation(
        sel, failed_pairs=[((0, 1), 1.0)], X=X, cols=list(X.columns), classes_y=classes_y, pair_maxt_floor=0.0, admitted_pool={}, verbose=0
    )
    return out, sel.fe_escalation_info_


def test_escalation_skipped_for_nominal_target():
    """The nominal flag makes escalation return nothing, propose nothing and record why."""
    out, info = _escalate(True)
    assert out == []
    assert info["proposed"] == 0
    assert "nominal" in info["skipped"]


def test_escalation_still_runs_for_ordinal_target():
    """Without the flag the same pair is processed, so the guard does not disable escalation globally."""
    _, info = _escalate(False)
    assert "skipped" not in info
    assert info["eligible_pairs"] == [("x0", "x1")]
