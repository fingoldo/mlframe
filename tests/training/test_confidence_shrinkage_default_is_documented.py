"""`apply_confidence_shrinkage` is ON by default, and the documentation must say so.

The class comment stated "Default ON ... enabled unconditionally" while the suite kwarg and the phase docstring both
said "default OFF"; the config is constructed for every run, so a caller who passed no `regression_calibration_config`
and read "Opt-in ... default OFF" still got every regression model's test predictions pulled toward a neutral value.
"""

from __future__ import annotations

import inspect


from mlframe.training._reporting_configs import RegressionCalibrationConfig


def test_the_default_is_on():
    """The behaviour the documentation below has to describe: the config ships with shrinkage enabled."""
    assert RegressionCalibrationConfig().apply_confidence_shrinkage is True


def test_the_point_recalibration_default_is_still_off():
    """The other half of the config - the one the "opt-in" wording actually describes - stays off."""
    assert RegressionCalibrationConfig().point == "off"


def test_no_documentation_still_calls_the_shrinkage_default_off():
    """The implementing function's own docstring must not tell a reader the step is opt-in."""
    from mlframe.training.core import _phase_finalize_calibration as phase

    doc = inspect.getdoc(phase._apply_confidence_shrinkage_to_regression) or ""
    assert "ON by default" in doc
    assert "default OFF" not in doc


def test_the_suite_kwarg_documents_the_two_defaults_separately():
    """The suite entry point is where most callers read the default, so its parameter section has to carry it too."""
    from mlframe.training.core import _main_train_suite as suite

    doc = inspect.getdoc(suite.train_mlframe_models_suite) or ""
    section = doc[doc.index("regression_calibration_config:") :][:600]
    assert "apply_confidence_shrinkage" in section and "ON by default" in section
