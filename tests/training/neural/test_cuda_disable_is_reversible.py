"""Hiding CUDA process-wide after an invalidated context must be undoable, not a one-way door."""

import os

import pytest

torch = pytest.importorskip("torch")

from mlframe.training.neural.base import _cuda_fallback as cf


@pytest.fixture(autouse=True)
def _leave_cuda_as_found():
    prior_env, prior_fn, prior_state = os.environ.get("CUDA_VISIBLE_DEVICES"), torch.cuda.is_available, cf._CUDA_DISABLED_STATE
    yield
    cf._CUDA_DISABLED_STATE = prior_state
    torch.cuda.is_available = prior_fn
    if prior_env is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = prior_env


def test_the_disable_is_recorded_and_can_be_undone(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    original = torch.cuda.is_available
    cf._CUDA_DISABLED_STATE = None

    cf._disable_cuda_globally()
    assert cf.cuda_is_disabled_for_this_process()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and torch.cuda.is_available() is False

    assert cf.restore_cuda_visibility() is True
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert torch.cuda.is_available is original
    assert not cf.cuda_is_disabled_for_this_process()


def test_restoring_an_unset_variable_removes_it_again(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cf._CUDA_DISABLED_STATE = None
    cf._disable_cuda_globally()
    assert cf.restore_cuda_visibility() is True
    assert "CUDA_VISIBLE_DEVICES" not in os.environ, "a variable that was never set must not be left behind as empty"


def test_restoring_when_nothing_was_disabled_is_a_no_op():
    cf._CUDA_DISABLED_STATE = None
    assert cf.restore_cuda_visibility() is False


def test_a_second_disable_keeps_the_original_state(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    cf._CUDA_DISABLED_STATE = None
    cf._disable_cuda_globally()
    cf._disable_cuda_globally()  # a later estimator hits the same path
    cf.restore_cuda_visibility()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3", "the second disable must not record the already-disabled state"


def test_a_new_suite_restores_cuda_visibility(monkeypatch):
    """The disable is about the moment, not the host: a new suite gets the GPU back, and gives it up again if it must."""
    import mlframe.training.core._phase_config_setup as pcs

    cf._CUDA_DISABLED_STATE = None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cf._disable_cuda_globally()
    assert cf.cuda_is_disabled_for_this_process()

    pcs._restore_cuda_visibility_for_a_new_suite()
    assert not cf.cuda_is_disabled_for_this_process()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_the_restore_does_not_import_torch_when_nothing_disabled_it(monkeypatch):
    """The suite must not pull torch into a run that never touches it just to ask this question."""
    import sys

    import mlframe.training.core._phase_config_setup as pcs

    monkeypatch.delitem(sys.modules, "mlframe.training.neural.base._cuda_fallback", raising=False)
    pcs._restore_cuda_visibility_for_a_new_suite()
    assert "mlframe.training.neural.base._cuda_fallback" not in sys.modules
