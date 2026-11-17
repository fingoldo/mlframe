"""Chunk-08 audit regressions: DCD / device helpers must log the fallbacks they take and not crash inside their own handlers."""
import logging
import sys
import types

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts

PKG = "mlframe.feature_selection.filters"


@pytest.fixture(autouse=True)
def _reset_throttles():
    """Forget throttle counts so each test sees its first log line."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def test_warm_marginal_entropies_failure_before_loop_does_not_raise_unbound_local():
    """A failure before the column loop (state without the scratch slots) is swallowed, not an UnboundLocalError from the handler."""
    from mlframe.feature_selection.filters._dcd_pair_su_batch import _warm_marginal_entropies

    state = types.SimpleNamespace(column_entropy_cache={})
    fd = np.zeros((10, 3), dtype=np.int32)
    _warm_marginal_entropies(state, [(0, 1)], fd, np.array([2, 2, 2]), np.int32)
    assert state.column_entropy_cache == {}


def test_pair_entropy_choose_failure_is_logged(monkeypatch, caplog):
    """A failing kernel-tuning choose() falls to the row floor and says so."""
    from mlframe.feature_selection.filters import _dcd_pair_su_gpu as m

    class Boom:
        """Spec whose choose() raises."""

        def choose(self, **kw):
            """Always raise."""
            raise RuntimeError("tuner down")

    monkeypatch.setattr(m, "_SPEC", Boom())
    with caplog.at_level(logging.WARNING):
        assert m.entropy_device_chosen(10, 10) is False
    assert "choose() failed" in caplog.text


def test_device_quantile_failure_is_logged(monkeypatch, caplog):
    """A device failure falls to the host np.quantile and says so."""
    from mlframe.feature_selection.filters import _device_quantile as m

    monkeypatch.setitem(m._STATE, "verified", True)

    def boom(x, qs):
        """Raise like a device fault."""
        raise RuntimeError("device fault")

    monkeypatch.setattr(m, "_device_quantile_raw", boom)
    with caplog.at_level(logging.WARNING):
        assert m.device_quantile(np.arange(5.0), np.array([0.5])) is None
    assert "device quantile failed" in caplog.text


def test_dcor_backend_selection_failure_is_logged(monkeypatch, caplog):
    """A failing tuning-cache lookup falls to the work crossover and says so."""
    from mlframe.feature_selection.filters import _dcor_gpu as m
    from mlframe.feature_selection.filters import _gpu_policy
    from pyutilz.performance.kernel_tuning import cache as kcache

    fake_cp = types.SimpleNamespace(cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(getDeviceCount=lambda: 1)))
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setattr(_gpu_policy, "gpu_globally_disabled", lambda: False)
    monkeypatch.delenv("MLFRAME_DCOR_DEVICE", raising=False)

    def boom():
        """Raise like a corrupt cache."""
        raise RuntimeError("cache down")

    monkeypatch.setattr(kcache.KernelTuningCache, "load_or_create", staticmethod(boom))
    with caplog.at_level(logging.WARNING):
        assert m.dcor_backend(10, 10) == "cpu"
    assert "dCor backend selection failed" in caplog.text


def test_auto_swap_method_fold_failure_is_logged(monkeypatch, caplog):
    """A combiner that fails on every fold is reported instead of silently scoring 0.0."""
    from mlframe.feature_selection.filters import _cluster_aggregate
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import DCDState, _select_swap_method_auto

    rng = np.random.default_rng(0)
    n = 60
    fd = np.column_stack([rng.integers(0, 3, n), rng.integers(0, 2, n)]).astype(np.int32)
    state = DCDState(factors_data=fd, target_indices=np.array([1]), quantization_nbins=3)

    def boom(*a, **k):
        """Raise like a degenerate fold."""
        raise ValueError("degenerate")

    monkeypatch.setattr(_cluster_aggregate, "_derive_weights", boom)
    with caplog.at_level(logging.WARNING):
        _select_swap_method_auto(state=state, Z=rng.standard_normal((n, 3)), target_y=None, member_names=("a", "b", "c"))
    assert "bake-off" in caplog.text


def test_recipe_edge_failure_is_logged(caplog):
    """Fit-time bin edges that cannot be derived are reported, since transform then re-quantiles."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import DCDState, SwapDecision
    from mlframe.feature_selection.filters._dynamic_cluster_discovery._dcd_swap import _register_engineered_recipe

    state = DCDState(cols=["a", "b"], quantization_nbins=3)
    recipe = {"members": [0, 1], "mean": [0.0, 0.0], "std": [1.0, 1.0], "signs": [1.0, 1.0], "weights": [0.7, 0.7], "method": "pca_pc1"}
    decision = SwapDecision(accept=True, aggregate_name="agg", binned_rep=np.array([0, 1, 2]), recipe_obj=recipe, rep_continuous=np.array(["x", "y", "z"]))
    with caplog.at_level(logging.WARNING):
        _register_engineered_recipe(state, decision, {})
    assert "bin edges not stored" in caplog.text
