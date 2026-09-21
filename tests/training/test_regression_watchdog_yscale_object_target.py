"""Regression: the composite wrap-pass watchdog must still check a model whose target slice is an object array.

``np.isfinite`` raises on an object-dtype slice, and the old watchdog swallowed that at DEBUG, so a schema-drifted target
(numbers stored as objects) silently switched the check off. The watchdog now casts the target to float64 once; a target
that cannot be cast reports that the check could not run, at WARNING.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.training.core._composite_wrap_watchdog import run_wrap_watchdog

from tests.training.composite.estimator.test_wrap_watchdog_oracle import _grouped_wrapper, _watchdog_warnings


def test_object_dtype_numeric_target_is_still_checked(caplog, monkeypatch):
    """Numbers stored as objects: a drifted prediction is still caught, so the check ran."""
    wrapper, spec, df, y = _grouped_wrapper()
    y_obj = y.astype(object)
    try:
        np.isfinite(y_obj)
        raised = False
    except TypeError:
        raised = True
    assert raised, "an object-dtype target must trip a bare np.isfinite"
    orig = wrapper.predict
    monkeypatch.setattr(wrapper, "predict", lambda X, inner_X=None: np.asarray(orig(X)) + 0.5)
    with caplog.at_level(logging.WARNING):
        run_wrap_watchdog(wrapper, spec, df, y_obj, composite_name="c", split_name="val")
    assert any("y-MAE" in m for m in _watchdog_warnings(caplog))


def test_genuinely_nonnumeric_target_is_reported_not_swallowed(caplog):
    """A target that cannot be cast to float reports that the check could not run."""
    wrapper, spec, df, y = _grouped_wrapper()
    with caplog.at_level(logging.WARNING):
        run_wrap_watchdog(wrapper, spec, df, np.array(["a"] * len(y), dtype=object), composite_name="c", split_name="val")
    assert any("could not run" in m for m in _watchdog_warnings(caplog))
