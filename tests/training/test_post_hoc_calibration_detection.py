"""The post-hoc calibration wrapper lives under the bundle's ``.model``, not on the bundle itself.

`entry.model = wrapped` is what training stamps, while the predict path tested `type(model_obj).__name__` - the bundle's
own class - so every member answered "not calibrated", the mixed-calibration WARN could never fire, and a deployed
reliability diagram diverged from each member's with nothing in the log.
"""

from __future__ import annotations

import types

from mlframe.training.core.predict import _is_post_hoc_calibrated_model


class _PostHocCalibratedModel:  # names matter: the detector matches on the class name
    pass


class _PostHocMultiCalibratedModel:
    pass


class _PlainEstimator:
    pass


def test_a_wrapper_under_the_bundle_is_detected():
    bundle = types.SimpleNamespace(model=_PostHocCalibratedModel())
    assert _is_post_hoc_calibrated_model(bundle) is True


def test_a_multi_output_wrapper_under_the_bundle_is_detected():
    bundle = types.SimpleNamespace(model=_PostHocMultiCalibratedModel())
    assert _is_post_hoc_calibrated_model(bundle) is True


def test_a_bare_wrapper_is_still_detected():
    """The pre-existing shape stays supported: some callers pass the wrapper itself."""
    assert _is_post_hoc_calibrated_model(_PostHocCalibratedModel()) is True


def test_an_uncalibrated_bundle_is_not_detected():
    assert _is_post_hoc_calibrated_model(types.SimpleNamespace(model=_PlainEstimator())) is False
    assert _is_post_hoc_calibrated_model(_PlainEstimator()) is False
    assert _is_post_hoc_calibrated_model(None) is False
