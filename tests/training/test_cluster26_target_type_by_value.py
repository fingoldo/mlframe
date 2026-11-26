"""Cluster 26 regression: a target_type given as an enum value resolves to that type (audit 2026-11-20, finding 26-F37).

The string branch resolved only member names, so a value such as ``"multiclass_classification"`` fell through to the shape-based guess and the
caller's type was ignored. The branch now tries the value first, then the name.
"""

import numpy as np

from mlframe.training import _calibration_models as cm
from mlframe.training.configs import TargetTypes


class _Base:
    """A bare classifier stand-in; the calibrator reads only ``classes_`` from it."""


def _wrapper_type(target_type: str, labels: np.ndarray):
    """Fit the multi-target calibrated wrapper for ``target_type`` on ``labels`` and return the type it resolved to."""
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.05, 0.95, labels.shape)
    wrapper = cm._fit_calibrated_wrapper(object(), _Base(), probs, labels, target_type, True)
    return wrapper._target_type


def test_value_string_resolves_to_the_enum_member():
    """The enum value resolves to the matching TargetTypes member (26-F37)."""
    assert TargetTypes("multilabel_classification") is TargetTypes.MULTILABEL_CLASSIFICATION


def test_value_string_overrides_the_shape_guess():
    """A 2-D label array with the value ``multiclass_classification`` resolves to MULTICLASS, not the MULTILABEL the shape implies (26-F37)."""
    labels = np.random.default_rng(1).integers(0, 2, (200, 3))
    assert _wrapper_type("multiclass_classification", labels) is TargetTypes.MULTICLASS_CLASSIFICATION


def test_value_string_agreeing_with_the_shape_still_resolves():
    """Control: a value that matches the label shape resolves to that type (26-F37)."""
    labels = np.random.default_rng(2).integers(0, 2, (200, 3))
    assert _wrapper_type("multilabel_classification", labels) is TargetTypes.MULTILABEL_CLASSIFICATION
