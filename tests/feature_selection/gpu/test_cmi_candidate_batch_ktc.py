"""The candidate-batch CMI scorer of the CMI-greedy FE construction takes its host-or-device choice from a kernel-tuning entry, with the former fixed floors as the pre-sweep fallback."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _mi_greedy_cmi_fe_ktc as K


def test_entry_is_registered_with_a_work_axis():
    """The tuner entry exists and sweeps the candidate-row work."""
    assert K._SPEC is not None
    assert list(K._SWEEP_ROWS) == sorted(K._SWEEP_ROWS) and K._SWEEP_ROWS[0] <= K.DEVICE_FALLBACK_ROWS


def test_fallback_floors_without_a_sweep(monkeypatch):
    """When the tuner cannot answer, both floors apply: at least 64 candidates and 1e6 candidate-rows."""
    monkeypatch.setattr(K, "_SPEC", None)
    assert K.candidate_batch_device_chosen(20_000, 100)
    assert not K.candidate_batch_device_chosen(20_000, 10)  # too few candidates
    assert not K.candidate_batch_device_chosen(1_000, 100)  # too little work


def test_a_failing_choose_uses_the_floors(monkeypatch):
    """An exception from the tuner is not an exception for the caller."""

    class _Boom:
        """A tuner entry whose choose() raises."""

        def choose(self, **_kw):
            """Raise as a broken cache would."""
            raise RuntimeError("cache unavailable")

    monkeypatch.setattr(K, "_SPEC", _Boom())
    assert K.candidate_batch_device_chosen(30_000, 128)
    assert not K.candidate_batch_device_chosen(30_000, 8)


def test_sweep_bodies_agree_on_the_selection_view():
    """The host and device bodies of the sweep give the same selection view at the smallest swept work (the property the sweep enforces before it records a time)."""
    pytest.importorskip("cupy")
    inputs = K._sweep_inputs({"n_samples": K._SWEEP_ROWS[0]})
    host, dev = K._sweep_host(*inputs), K._sweep_device(*inputs)
    assert host.shape == dev.shape
    assert np.array_equal(host, dev)
