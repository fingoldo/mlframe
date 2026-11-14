"""The device quantile of the binned-aggregate edges takes its host-or-device choice from a kernel-tuning entry, with the former 200000-row floor as the pre-sweep fallback."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _binned_agg_cheap_mi as M


def test_entry_is_registered():
    """The tuner entry exists and sweeps column lengths around the former floor."""
    assert M._SPEC is not None
    assert min(M._SWEEP_ROWS) < M._DEVICE_QUANTILE_MIN_N < max(M._SWEEP_ROWS)


def test_fallback_floor_without_a_sweep(monkeypatch):
    """When the tuner cannot answer, the former floor decides."""
    monkeypatch.setattr(M, "_SPEC", None)
    assert M.device_quantile_chosen(M._DEVICE_QUANTILE_MIN_N)
    assert not M.device_quantile_chosen(M._DEVICE_QUANTILE_MIN_N - 1)


def test_a_failing_choose_uses_the_floor(monkeypatch):
    """An exception from the tuner is not an exception for the caller."""

    class _Boom:
        """A tuner entry whose choose() raises."""

        def choose(self, **_kw):
            """Raise as a broken cache would."""
            raise RuntimeError("cache unavailable")

    monkeypatch.setattr(M, "_SPEC", _Boom())
    assert M.device_quantile_chosen(10 * M._DEVICE_QUANTILE_MIN_N)
    assert not M.device_quantile_chosen(10)


def test_sweep_bodies_give_the_same_edges_with_ties():
    """Host and device edges of a column with tied values are equal (the property the sweep enforces before it records a time)."""
    pytest.importorskip("cupy")
    (x,) = M._sweep_inputs({"n_samples": 50_000})
    assert np.array_equal(M._sweep_host(x), M._sweep_device(x))
