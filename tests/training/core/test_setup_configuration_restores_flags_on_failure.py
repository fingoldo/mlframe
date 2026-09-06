"""A `setup_configuration` that raises must not leave its process-wide overrides flipped.

The four overrides (residual-audit reporting, inline display, format subfolders, calibration colormap) are
applied near the top of `setup_configuration`, but the snapshot `_phase_finalize` restores them from is only
written into `ctx.artifacts` at the very end -- and `ctx` does not exist until ~280 lines after the first
flip. Any raise in between left the flags on the failed suite's values with no snapshot in existence for
anyone to act on, for the lifetime of the thread. In a test process that is every later test, which is the
symptom `_process_flag_scope.py`'s docstring already records for the previous instance of this defect.

A caller that never receives a ctx can never restore them, so the unwind belongs here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlframe.training.core import _phase_config_setup as pcs
from mlframe.training.evaluation import _get_residual_audit_enabled, _set_residual_audit_enabled


class _Boom(RuntimeError):
    """The failure injected partway through setup."""


class _Config(SimpleNamespace):
    """A config stub that answers every attribute setup_configuration reaches for.

    Most reads go through ``getattr(cfg, name, default)``, but a few (``enable_crash_reporting``) are
    direct, so a bare SimpleNamespace raises on the success path before reaching the assertion.
    """

    def __getattr__(self, name: str) -> bool:
        """Anything this test did not set explicitly reads as off."""
        return False


def _setup(**overrides):
    """Call `setup_configuration` with the all-None config set the other tests in this tree use."""
    kwargs = dict(
        preprocessing_config=None,
        pipeline_config=None,
        feature_types_config=None,
        split_config=None,
        hyperparams_config=None,
        behavior_config=None,
        reporting_config=None,
        output_config=None,
        outlier_detection_config=None,
        feature_selection_config=None,
        confidence_analysis_config=None,
        baseline_diagnostics_config=None,
        dummy_baselines_config=None,
        quantile_regression_config=None,
        composite_target_discovery_config=None,
        feature_handling_config=None,
        model_name="m",
        target_name="t",
        mlframe_models=None,
        verbose=0,
    )
    kwargs.update(overrides)
    return pcs.setup_configuration(**kwargs)


@pytest.fixture(autouse=True)
def residual_audit_restored():
    """Leave the real flag exactly as this test found it, whatever the test does to it."""
    prior = _get_residual_audit_enabled()
    yield
    _set_residual_audit_enabled(prior)


def test_a_raise_after_the_flip_does_not_leave_residual_audit_flipped(monkeypatch):
    """The exact gap: the flip happens, the snapshot never does, and the exception escapes."""
    _set_residual_audit_enabled(True)

    def _explode(*args, **kwargs):
        """Stand in for any of the ~280 lines of setup between the flip and the snapshot."""
        raise _Boom("setup failed after the overrides were applied")

    assert hasattr(pcs, "_build_suite_common_params_dict"), "the injection point has been renamed; this test has lost its subject"
    monkeypatch.setattr(pcs, "_build_suite_common_params_dict", _explode)

    with pytest.raises(_Boom):
        _setup(behavior_config=_Config(report_residual_audit=False))

    assert _get_residual_audit_enabled() is True, "setup_configuration failed with the residual-audit override still flipped"


def test_the_flip_still_happens_on_the_success_path():
    """The unwind must only run on the failure path; a successful setup still applies the override."""
    _set_residual_audit_enabled(True)
    ctx = _setup(behavior_config=_Config(report_residual_audit=False))
    assert _get_residual_audit_enabled() is False, "the override no longer reaches the process on the success path"
    assert ctx.artifacts["_process_flag_prior_residual_audit"] is True, "the snapshot _phase_finalize restores from is missing"


def test_the_calibration_colormap_override_is_also_unwound(monkeypatch):
    """The three thread-local overrides share the failure path; this one is furthest from the first flip."""
    colors = pytest.importorskip("mlframe.reporting.colors")
    if not hasattr(colors, "set_calibration_cmap"):
        pytest.skip("this build of mlframe.reporting.colors has no calibration-cmap override to restore")

    prior = colors.get_calibration_cmap_override()
    try:
        monkeypatch.setattr(pcs, "_build_suite_common_params_dict", lambda *a, **k: (_ for _ in ()).throw(_Boom("boom")))
        with pytest.raises(_Boom):
            _setup(reporting_config=_Config(calibration_colormap="viridis"))
        assert colors.get_calibration_cmap_override() == prior, "the calibration-colormap override survived a failed setup"
    finally:
        colors.set_calibration_cmap(prior)
