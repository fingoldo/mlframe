"""The binned-aggregate and noise-gate fallbacks to a slower path say so (throttled warning) instead of a debug line."""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the per-key log counts so each test sees its first warning."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def test_group_selection_device_failure_warns(monkeypatch, caplog):
    """A failing strict-resident probe logs a warning and still scores on the CPU path."""
    import mlframe.feature_selection.filters._gpu_strict_fe as gs
    from mlframe.feature_selection.filters._binned_agg_cheap_mi import _cheap_mi_group_selection

    def boom():
        """Fail the probe."""
        raise RuntimeError("probe fault")

    monkeypatch.setattr(gs, "fe_gpu_strict_resident_enabled", boom)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"g": rng.normal(size=300)})
    with caplog.at_level(logging.WARNING):
        out = _cheap_mi_group_selection(X, ["g"], rng.integers(0, 2, 300))
    assert "g" in out
    assert any("probe fault" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_candidate_codes_resident_failure_warns(monkeypatch, caplog):
    """A failing resident binning logs a warning and falls back to the host binner."""
    import mlframe.feature_selection.filters._gpu_strict_fe as gs
    from mlframe.feature_selection.filters._binned_numeric_agg_redundancy import _candidate_codes

    monkeypatch.setattr(gs, "fe_gpu_strict_resident_enabled", lambda: (_ for _ in ()).throw(RuntimeError("resident fault")))
    with caplog.at_level(logging.WARNING):
        out = _candidate_codes(np.arange(10.0), 4, lambda col, nbins: "host")
    assert out == "host"
    assert any("resident fault" in r.getMessage() for r in caplog.records)


def test_device_gate_declined_warning_names_last_error(monkeypatch, caplog):
    """The 'declined twice' warning carries the exception of the failed device pass."""
    from mlframe.feature_selection.filters import _binned_numeric_agg_cands as cands

    def fake_local(*a, **k):
        """Fail like a device fault."""
        raise RuntimeError("pool exhausted")

    fake = types.ModuleType("mlframe.feature_selection.filters._binned_numeric_agg_resident")
    fake.local_mi_gate_binagg_resident = fake_local
    monkeypatch.setitem(sys.modules, "mlframe.feature_selection.filters._binned_numeric_agg_resident", fake)
    monkeypatch.setattr(cands, "_release_device_caches", lambda: None)

    def fit_fn(*a, **k):
        """Recipe-only fit stand-in."""
        return None, {"c": {"group_col": "g", "agg_col": "a"}}

    X = pd.DataFrame({"g": [1.0, 2.0], "a": [1.0, 2.0]})
    with caplog.at_level(logging.WARNING):
        res = cands.device_born_candidates(fit_fn, X, np.zeros(2), ["g"], ["a"], ("mean",), 10, 5, 0, set(), lambda names, raw: names, True, None)
    assert res[0] == "unavailable"
    assert any("pool exhausted" in r.getMessage() for r in caplog.records)


def test_noise_gate_backend_lookup_failure_warns(monkeypatch, caplog):
    """A failing per-host tuned lookup logs a warning and uses the size heuristic."""
    import mlframe.feature_selection.filters._batch_mi_noise_gate_tuning as t
    from pyutilz.performance.kernel_tuning import cache as cache_mod

    class _Boom:
        """Cache stand-in whose lookup fails."""

        @staticmethod
        def load_or_create():
            """Fail the load."""
            raise RuntimeError("cache fault")

    monkeypatch.setattr(cache_mod, "KernelTuningCache", _Boom)
    with caplog.at_level(logging.WARNING):
        choice = t._batch_mi_noise_gate_backend_choice(10, 10)
    assert choice == "cpu"
    assert any("cache fault" in r.getMessage() for r in caplog.records)
