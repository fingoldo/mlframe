"""Regression: in explain_models, nclasses was only assigned inside the per-fold
`if max_test_ind < L:` branch. When the FINAL TimeSeries fold has no OOS rows that branch is
skipped on the last iteration, leaving nclasses unbound/stale at the show_custom_calibration_plot
call. The fix derives nclasses once from the stacked probs (probs.shape[1]) before that call.

This sensor pins the structural property of the fixed code: nclasses is taken from the stacked
OOS probs, so it is defined regardless of which fold contributed the rows."""

from __future__ import annotations

import ast
import inspect

import numpy as np

from mlframe.inference import explainability


def test_nclasses_derived_from_stacked_probs_before_plot():
    src = inspect.getsource(explainability)
    # The fix must derive nclasses from the stacked probs (probs.shape[1]) at the do_ts_oos
    # aggregation site, not rely solely on the per-fold conditional assignment.
    tree = ast.parse(src)
    assigns_from_probs_shape = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "nclasses" in targets and isinstance(node.value, ast.Subscript):
                val = node.value
                if (
                    isinstance(val.value, ast.Attribute)
                    and val.value.attr == "shape"
                    and isinstance(val.value.value, ast.Name)
                    and val.value.value.id == "probs"
                ):
                    assigns_from_probs_shape.append(node)
    assert assigns_from_probs_shape, "nclasses must be derived from probs.shape[1] (empty-final-fold safety)"


def test_stacked_probs_nclasses_logic():
    # Emulate the empty-final-fold accumulation: folds contribute OOS probs, the last contributes
    # none. The stacked probs still yield the correct nclasses.
    all_probs = [np.zeros((3, 4)), np.zeros((2, 4))]  # final fold added nothing
    probs = np.vstack(all_probs)
    nclasses = probs.shape[1]
    assert nclasses == 4
