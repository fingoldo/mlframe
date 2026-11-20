"""Every device-to-host fallback of the usability / triple-MI / unary-stack helpers logs a WARNING (a silent slow path hid regressions)."""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the per-key log counts so each test sees its first occurrence."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _warned(caplog, text):
    """Whether a WARNING record contains ``text``."""
    return any(r.levelno >= logging.WARNING and text in r.getMessage() for r in caplog.records)


def test_triple_mi_gpu_launch_failure_warns_and_returns_none(monkeypatch, caplog):
    """A device launch failure returns None (host kernel runs) and says so at WARNING."""
    from mlframe.feature_selection.filters import _triple_mi_gpu as mod

    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))

    class _Boom:
        """A module cache whose build fails."""

        def get(self, cp):
            """Fail like a broken compile."""
            raise RuntimeError("boom")

    monkeypatch.setattr(mod, "_MODULE", _Boom())
    with caplog.at_level(logging.WARNING):
        got = mod.triple_mi_gpu(np.zeros((3, 8), np.int8), np.zeros(8, np.int64), 4, 2)
    assert got is None
    assert _warned(caplog, "device triple MI failed")


def test_triple_mi_gpu_missing_cupy_warns(monkeypatch, caplog):
    """A missing cupy returns None and says so at WARNING."""
    from mlframe.feature_selection.filters import _triple_mi_gpu as mod

    monkeypatch.setitem(sys.modules, "cupy", None)
    with caplog.at_level(logging.WARNING):
        got = mod.triple_mi_gpu(np.zeros((3, 8), np.int8), np.zeros(8, np.int64), 4, 2)
    assert got is None
    assert _warned(caplog, "device triple MI unavailable")


def test_triple_mi_choose_failure_warns(monkeypatch, caplog):
    """A failing kernel-tuning choose() falls to the floor and says so at WARNING."""
    from mlframe.feature_selection.filters import _triple_mi_gpu as mod

    class _Spec:
        """A tuning spec whose choose() fails."""

        def choose(self, **kw):
            """Fail."""
            raise RuntimeError("no cache")

    monkeypatch.delenv("MLFRAME_TRIPLE_MI_DEVICE", raising=False)
    monkeypatch.setattr(mod, "_SPEC", _Spec())
    with caplog.at_level(logging.WARNING):
        assert mod.use_device_triples(10, 10) is False
    assert _warned(caplog, "KTC choose() failed")


def _fake_device_column():
    """A stand-in for a float32 device column and its op-code array."""
    return SimpleNamespace(dtype=np.dtype(np.float32), ndim=1, shape=(5,)), SimpleNamespace(shape=(3,))


def test_unary_stack_no_cushion_warns(monkeypatch, caplog):
    """Declining the stack for lack of VRAM says so at WARNING."""
    from mlframe.feature_selection.filters import _fe_gpu_vram, _unary_stack_cuda as mod

    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda nbytes: False)
    d_x, codes = _fake_device_column()
    with caplog.at_level(logging.WARNING):
        assert mod.unary_stack_cuda(d_x, codes, 1.0) is None
    assert _warned(caplog, "no VRAM cushion")


def test_unary_stack_failure_warns(monkeypatch, caplog):
    """A launch or compile failure falls back to the per-unary path and says so at WARNING."""
    from mlframe.feature_selection.filters import _fe_gpu_vram, _unary_stack_cuda as mod

    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda nbytes: True)
    d_x, codes = _fake_device_column()
    with caplog.at_level(logging.WARNING):
        assert mod.unary_stack_cuda(d_x, codes, 1.0) is None
    assert _warned(caplog, "unary_stack_cuda unavailable")


def test_resident_greedy_dispatch_failure_warns(monkeypatch, caplog):
    """A failing resident-greedy dispatch falls to the CPU greedy and says so at WARNING."""
    from mlframe.feature_selection.filters import _gpu_strict_fe, _usability_aware_selection as mod

    def _boom():
        """Fail like a broken probe."""
        raise RuntimeError("probe")

    monkeypatch.setattr(_gpu_strict_fe, "fe_gpu_strict_resident_enabled", _boom)
    with caplog.at_level(logging.WARNING):
        got = mod._usability_greedy_try_gpu_resident(
            [], np.zeros(3), w=0.5, K=1, seed=0, n_folds=2, mae_improve_rel=0.01, shortlist=1, shortlist_diversity_corr=0.9, classification=False,
        )
    assert got is None
    assert _warned(caplog, "GPU-resident greedy path unavailable")


def test_host_batched_newton_decline_warns(monkeypatch, caplog):
    """The sklearn refit taken after the batched Newton CV declines says so at WARNING."""
    from mlframe.feature_selection.filters import _usability_aware_selection as mod, _usability_greedy_clf_host_batched as hb

    def _decline(*a, **k):
        """Decline like a singular fold."""
        raise hb.HostBatchedFallbackError("singular fold")

    monkeypatch.setattr(hb, "cv_candidates_binary_host", _decline)
    rng = np.random.default_rng(0)
    y = (rng.random(80) > 0.5).astype(int)
    pool = [mod.UsableCandidate(f"c{i}", rng.standard_normal(80) + y * (i + 1) * 0.3, 0.1) for i in range(3)]
    with caplog.at_level(logging.WARNING):
        mod.usability_greedy(pool, y, K=1, classification=True, shortlist=3)
    assert _warned(caplog, "batched Newton CV of the usability greedy declined")


def test_recipe_build_failure_warns(monkeypatch, caplog):
    """A candidate dropped because its recipe cannot be built says so at WARNING."""
    from mlframe.feature_selection.filters import _usability_aware_selection as mod, engineered_recipes

    def _boom(**kw):
        """Fail like a recipe the machinery cannot build."""
        raise ValueError("no recipe")

    monkeypatch.setattr(engineered_recipes, "build_unary_binary_recipe", _boom)
    cand = mod.UsableCandidate("mul(identity(a),identity(b))", np.arange(10.0), 0.5, None, ("a", "b"), ("identity", "identity", "mul"))
    pool: list = []
    with caplog.at_level(logging.WARNING):
        mod._build_usability_cand_step2_sub_quantum_cpu([cand], lambda m: m, 4, 0.97, "medium", "minimal", 10, "quantile", np.int32, {}, None, np.float32, pool)
    assert pool == []
    assert _warned(caplog, "recipe build failed")


def test_clf_resident_fallback_warns(monkeypatch, caplog):
    """The classification resident twin falling to the CPU greedy says so at WARNING."""
    from mlframe.feature_selection.filters import _usability_greedy_clf_gpu_resident as mod

    monkeypatch.setattr(mod, "_import_cupy_unless_disabled", lambda: object())

    def _degenerate(y):
        """Fail like a non-convergent fit."""
        raise mod._ResidentClfFallbackError("no convergence")

    monkeypatch.setattr(mod, "_resident_binary_codes", _degenerate)
    pool = [SimpleNamespace(values=np.arange(4.0), mi=0.1)]
    with caplog.at_level(logging.WARNING):
        assert mod.usability_greedy_clf_gpu_resident(pool, np.array([0, 1, 0, 1])) is None
    assert _warned(caplog, "non-convergent/singular/degenerate device fit")


def test_regression_resident_fallback_warns(monkeypatch, caplog):
    """The regression resident twin falling to the CPU greedy says so at WARNING."""
    from mlframe.feature_selection.filters import _gpu_policy, _usability_greedy_gpu_resident as mod

    fake = types.ModuleType("cupy")
    fake.asarray = lambda a, *x, **k: a  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(_gpu_policy, "gpu_globally_disabled", lambda: False)

    def _degenerate(*a, **k):
        """Fail like a singular border."""
        raise mod._ResidentFallbackError("singular")

    monkeypatch.setattr(mod, "_shrink_shortlist_to_ram_budget", _degenerate)
    pool = [SimpleNamespace(values=np.arange(6.0), mi=0.1)]
    with caplog.at_level(logging.WARNING):
        assert mod.usability_greedy_gpu_resident(pool, np.arange(6.0)) is None
    assert _warned(caplog, "singular border / degenerate device fit")


def test_abscorr_gpu_failure_warns(monkeypatch, caplog):
    """A failing GPU |corr| falls back to the CPU value and says so at WARNING."""
    from mlframe.feature_selection.filters import _usability_gpu, _usability_greedy_steps as steps

    def _boom(u, v):
        """Fail like a device fault."""
        raise RuntimeError("device")

    monkeypatch.setattr(steps, "_GPU_USABILITY", lambda: True)
    monkeypatch.setattr(_usability_gpu, "gpu_abscorr", _boom)
    u = np.arange(10.0)
    with caplog.at_level(logging.WARNING):
        assert steps._abscorr(u, u * 2.0) == pytest.approx(1.0)
    assert _warned(caplog, "_abscorr: GPU path failed")


def test_abscorr_batch_gpu_failure_warns(monkeypatch, caplog):
    """A failing batched GPU |corr| in the shortlist falls back to the CPU loop and says so at WARNING."""
    from mlframe.feature_selection.filters import _usability_gpu, _usability_greedy_steps as steps

    def _boom(cols, v):
        """Fail like a device fault."""
        raise RuntimeError("device")

    monkeypatch.setattr(steps, "_GPU_USABILITY", lambda: True)
    monkeypatch.setattr(_usability_gpu, "gpu_abscorr_batch", _boom)
    rng = np.random.default_rng(0)
    pool = [SimpleNamespace(name=f"c{i}", values=rng.standard_normal(30), mi=0.1) for i in range(3)]
    y = rng.standard_normal(30)
    shortlist = steps._usability_greedy_step3_cheap_residual_aware(False, None, None, lambda i: pool[i].values, None, None, y, pool, 0.5, 0.1, 0.97, 3)
    with caplog.at_level(logging.WARNING):
        assert len(shortlist([])) >= 1
    assert _warned(caplog, "gpu_abscorr_batch failed")
