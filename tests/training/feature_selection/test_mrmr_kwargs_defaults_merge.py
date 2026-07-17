"""Sensor tests for MRMR kwargs default merge / verbose propagation (A-P2-001 / A-P2-002 / A-Low-010)."""

from mlframe.training.core._setup_helpers import _initialize_training_defaults


def test_caller_kwargs_extend_defaults_not_replace():
    """Caller-supplied mrmr_kwargs must shallow-merge with defaults; max_runtime_mins must survive."""
    _, _, mrmr_kwargs = _initialize_training_defaults(
        common_params_dict=None,
        rfecv_models=None,
        mrmr_kwargs={"verbose": 0},
        suite_verbose=0,
    )
    assert mrmr_kwargs["verbose"] == 0, "caller-supplied verbose must override default"
    assert mrmr_kwargs["max_runtime_mins"] == 300, "max_runtime_mins default must survive caller override (A-P2-001 shallow-merge)"
    assert mrmr_kwargs["fe_max_steps"] == 1
    assert mrmr_kwargs["n_workers"] >= 1


def test_default_verbose_tracks_suite_verbose():
    """When caller passes no verbose, MRMR verbose must follow suite_verbose."""
    _, _, mrmr_kwargs = _initialize_training_defaults(
        common_params_dict=None,
        rfecv_models=None,
        mrmr_kwargs=None,
        suite_verbose=0,
    )
    assert mrmr_kwargs["verbose"] == 0, "default MRMR verbose must track suite_verbose=0"

    _, _, mrmr_kwargs2 = _initialize_training_defaults(
        common_params_dict=None,
        rfecv_models=None,
        mrmr_kwargs=None,
        suite_verbose=2,
    )
    assert mrmr_kwargs2["verbose"] == 2


def test_psutil_none_safe_fallback(monkeypatch):
    """psutil.cpu_count(logical=False) returning None must not raise TypeError (A-Low-010)."""
    import psutil

    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: None)
    _, _, mrmr_kwargs = _initialize_training_defaults(
        common_params_dict=None,
        rfecv_models=None,
        mrmr_kwargs=None,
    )
    assert mrmr_kwargs["n_workers"] >= 1
