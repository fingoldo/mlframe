"""The ShapProxiedFS clustering GPU width gates take their decision from the per-host kernel_tuning_cache sweep, not a hardcoded width."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_cluster_su as su_mod
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_gpu_tuning as tuning


def test_fallback_choice_follows_the_calibrated_widths():
    """Before any sweep has run the old constants decide, and only they."""
    assert tuning._dense_fallback_choice(tuning.DENSE_FALLBACK_MIN_FEATURES - 1) == "cpu"
    assert tuning._dense_fallback_choice(tuning.DENSE_FALLBACK_MIN_FEATURES) == "gpu"
    assert tuning._su_fallback_choice(tuning.SU_FALLBACK_MIN_FEATURES - 1) == "cpu"
    assert tuning._su_fallback_choice(tuning.SU_FALLBACK_MIN_FEATURES) == "gpu"


@pytest.mark.parametrize("spec_name, gate", [("_DENSE_SPEC", tuning.dense_gpu_pays_off), ("_SU_SPEC", tuning.su_gpu_pays_off)])
def test_gate_returns_the_measured_choice_and_survives_a_tuner_failure(monkeypatch, spec_name, gate):
    """The gate answers with the tuner's choice at the real (unsnapped) size; a tuner exception falls back to the width constant."""
    spec = getattr(tuning, spec_name)
    seen = {}

    def choose(**dims):
        """Record the dims the gate asked about and answer gpu."""
        seen.update(dims)
        return "gpu"

    monkeypatch.setattr(spec, "choose", choose)
    assert gate(123, 4567) is True
    assert seen == {"f": 123, "n": 4567}

    monkeypatch.setattr(spec, "choose", lambda **dims: "cpu")
    assert gate(10**6, 10) is False

    def boom(**dims):
        """A tuner that cannot answer."""
        raise RuntimeError("tuner unavailable")

    monkeypatch.setattr(spec, "choose", boom)
    assert gate(10**6, 10) is True  # far above either fallback width
    assert gate(1, 10) is False


def test_su_route_uses_the_tuner_unless_a_width_is_given(monkeypatch):
    """``gpu_min_features=None`` defers to the measured choice; an explicit width overrides it either way."""
    monkeypatch.setattr(su_mod, "cluster_su_gpu_available", lambda: True)
    monkeypatch.setattr(su_mod, "_gpu_free_memory_bytes", lambda: 10**12)
    kwargs = dict(n_features=100, n_samples=1000, max_n_bins=10)

    monkeypatch.setattr(su_mod, "su_gpu_pays_off", lambda f, n: False)
    assert su_mod._should_route_su_gpu(**kwargs) is False
    assert su_mod._should_route_su_gpu(**kwargs, gpu_min_features=50) is True

    monkeypatch.setattr(su_mod, "su_gpu_pays_off", lambda f, n: True)
    assert su_mod._should_route_su_gpu(**kwargs) is True
    assert su_mod._should_route_su_gpu(**kwargs, gpu_min_features=500) is False


def test_dense_cpu_probe_counts_the_edges_above_threshold():
    """The CPU sweep variant counts exactly the upper-triangle pairs with |corr| above the sweep threshold."""
    (Z,) = tuning._make_dense_inputs({"n": 400, "f": 60})
    expected = sum(1 for i in range(60) for j in range(i + 1, 60) if abs(float(Z[:, i] @ Z[:, j]) / 400) > tuning._SWEEP_THRESHOLD_DENSE)
    assert expected > 0
    assert abs(float(tuning._dense_edges_cpu_count(Z)) - expected) <= 2


def test_dense_gpu_probe_matches_the_cpu_probe():
    """Both sweep variants must report the same edge count, or the sweep's equivalence check would reject the GPU variant."""
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device")
    (Z,) = tuning._make_dense_inputs({"n": 2000, "f": 300})
    gpu = tuning._dense_edges_gpu_count(Z)
    assert abs(float(gpu) - float(tuning._dense_edges_cpu_count(Z))) <= 2


def test_su_cpu_probe_runs_on_the_packed_inputs():
    """The SU sweep inputs pack and the CPU variant finds the planted dependent pairs."""
    packed = tuning._make_su_inputs({"n": 1500, "f": 40})
    assert float(tuning._su_edges_cpu_count(*packed)) > 0
    assert np.isfinite(packed[4]).all()
