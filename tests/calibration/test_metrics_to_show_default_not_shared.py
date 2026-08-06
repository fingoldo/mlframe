"""CALIBRATION-8: estimate_calibration_quality_binned / show_classifier_calibration must not use the
module-level METRICS_TO_SHOW dict directly as a mutable default argument -- a mutable default is shared
across every call that doesn't pass metrics_to_show explicitly, so any future in-place mutation of it
(e.g. a caller doing ``metrics_to_show.pop(...)``) would silently corrupt the default for all other
callers for the lifetime of the process.
"""

from __future__ import annotations

import numpy as np

from mlframe.calibration.quality import METRICS_TO_SHOW, estimate_calibration_quality_binned, show_classifier_calibration


def test_estimate_calibration_quality_binned_default_is_not_the_module_dict_object():
    """The default resolves to METRICS_TO_SHOW's *contents*, but must not be that literal object."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200).astype(np.float64)
    y_pred = rng.uniform(0, 1, 200)
    _, _, _, metrics = estimate_calibration_quality_binned(y_true, y_pred)
    assert set(metrics.keys()) == set(METRICS_TO_SHOW.keys())


def test_show_classifier_calibration_default_is_not_the_module_dict_object():
    """Same guard for show_classifier_calibration's own metrics_to_show default."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, 200).astype(np.float64)
    y_pred = rng.uniform(0, 1, 200)
    perf = show_classifier_calibration(y_true, y_pred, title="t", skip_plotting=True)
    assert isinstance(perf, dict)
    assert set(perf.keys()) == set(METRICS_TO_SHOW.keys())


def test_mutating_a_caller_supplied_metrics_to_show_does_not_leak_into_next_default_call():
    """Passing a caller-owned dict (even if the caller mutates it afterward) must not affect a
    later call that relies on the default -- proves the default isn't aliased to caller state."""
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 2, 200).astype(np.float64)
    y_pred = rng.uniform(0, 1, 200)

    custom = dict(METRICS_TO_SHOW)
    custom.pop(next(iter(custom)))
    estimate_calibration_quality_binned(y_true, y_pred, metrics_to_show=custom)

    _, _, _, metrics = estimate_calibration_quality_binned(y_true, y_pred)
    assert set(metrics.keys()) == set(METRICS_TO_SHOW.keys())
