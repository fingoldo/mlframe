"""Regression: the composite OOD watchdog must compute y-scale on an object/string target slice.

Pre-fix ``np.std(_y_split[np.isfinite(_y_split)])`` ran on the raw (possibly object-dtype) target
slice; a non-float slice raised, the watchdog swallowed it at DEBUG, and the OOD check was silently
disabled. ``_watchdog_y_scale`` casts to float64 once and never raises.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.core._phase_composite_wrapping import _watchdog_y_scale


def test_object_dtype_numeric_target_yields_finite_scale():
    # numeric values stored in an object array (the schema-drift case that broke np.std/np.isfinite).
    y = np.array([1.0, 2.0, 3.0, 4.0, np.nan], dtype=object)
    # Pre-fix shape raised on the object array:
    try:
        np.isfinite(y)
        raised = False
    except TypeError:
        raised = True
    assert raised, "object-dtype target must trip the pre-fix np.isfinite path"

    scale = _watchdog_y_scale(y)
    assert np.isfinite(scale) and scale > 0


def test_float_target_matches_plain_std():
    y = np.array([1.0, 5.0, 9.0, np.inf])
    expected = float(np.std(y[np.isfinite(y)])) or 1.0
    assert _watchdog_y_scale(y) == expected


def test_genuinely_nonnumeric_target_falls_back_to_one():
    y = np.array(["a", "b", "c"], dtype=object)
    assert _watchdog_y_scale(y) == 1.0
