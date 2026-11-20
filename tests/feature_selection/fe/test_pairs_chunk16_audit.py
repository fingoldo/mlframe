"""Regression tests for the FE pair-search audit (chunk 16): operand-floor input checks, name-cache use, dead-column floor, ext-val kernel guards, silent-fallback logs."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_dispatch, _pairs_emit, _pairs_emit_extval, _pairs_gpu_gates
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_gates import _select_single_best
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_operand_floor import operand_uplift_null_bar
from mlframe.utils.log_throttle import reset_throttle_counts


def test_operand_uplift_null_bar_rejects_length_mismatch_instead_of_reading_out_of_bounds() -> None:
    """Operand codes shorter than the target used to index past the compiled kernel's arrays; they must raise."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 50)
    b = rng.integers(0, 4, 50)
    y = rng.integers(0, 2, 60)
    with pytest.raises(ValueError):
        operand_uplift_null_bar(a, b, y, floor=0.1)


def test_operand_uplift_null_bar_rejects_negative_codes() -> None:
    """A negative ordinal code indexes outside the joint table inside the njit kernel; it must raise."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 50)
    a[3] = -1
    b = rng.integers(0, 4, 50)
    y = rng.integers(0, 2, 50)
    with pytest.raises(ValueError):
        operand_uplift_null_bar(a, b, y, floor=0.1)


def test_select_single_best_uses_name_cache_on_the_default_path() -> None:
    """The caller-owned name cache decides the name tie-break on the rank-MI-primary path as well as the usability-primary one."""
    cols = ["a", "b"]
    cfg_add = (((0, "identity"), (1, "identity")), "add", 0)
    cfg_mul = (((0, "identity"), (1, "identity")), "mul", 1)
    perf = {cfg_add: 0.5, cfg_mul: 0.5}
    cache = {cfg_add: "zzz", cfg_mul: "aaa"}
    assert _select_single_best(perf, cols, name_cache=cache) == cfg_mul


def test_is_dead_column_keeps_a_tiny_scale_informative_column() -> None:
    """A column whose whole scale is 1e-12 is informative, not dead; a constant one still is."""
    rng = np.random.default_rng(1)
    tiny = (rng.standard_normal(500) * 1e-12).astype(np.float32)
    assert not _pairs_emit._is_dead_column(float(np.std(tiny)), tiny)
    const = np.full(500, 1e-12, dtype=np.float32)
    assert _pairs_emit._is_dead_column(float(np.std(const)), const)
    unit_const = np.full(500, 3.0)
    assert _pairs_emit._is_dead_column(float(np.std(unit_const)), unit_const)


def test_extval_host_buffer_rejects_param_a_shorter_than_the_buffer() -> None:
    """The ext-val njit kernel reads ``param_a[r]`` for every buffer row; a shorter operand must raise, not read past the array."""
    n_rows = 50
    factors = [np.arange(n_rows, dtype=np.float64), np.ones(n_rows)]
    op_codes = np.array([0, 1], dtype=np.int8)
    with pytest.raises(ValueError):
        _pairs_emit_extval._extval_host_buffer(np.ones(40), factors, op_codes, [np.multiply, np.add], 4, n_rows, lambda *a: None)


def test_dispatch_gpu_noise_gate_failure_logs_a_warning(monkeypatch, caplog) -> None:
    """A GPU noise-gate failure that drops to the CPU kernel is logged at WARNING, not debug."""
    reset_throttle_counts()
    monkeypatch.setattr(_pairs_dispatch, "_tuned_noise_gate_backend", lambda n, K: "gpu")

    def _boom(**kwargs):
        """Stand-in GPU twin that always fails."""
        raise RuntimeError("simulated GPU fault")

    monkeypatch.setattr(_pairs_dispatch, "_batch_mi_with_noise_gate_gpu", _boom)
    disc = np.zeros((20, 3), dtype=np.int8)
    classes_y = np.zeros(20, dtype=np.int64)
    with caplog.at_level(logging.WARNING):
        out = _pairs_dispatch._dispatch_batch_mi_with_noise_gate_impl_part1(
            0, False, np.array([1.0]), 20, 10, None, _pairs_dispatch._FeDispatchEnvGate(False, True), classes_y, 0.99, lambda: None, None, disc,
            np.full(3, 10, dtype=np.int64), classes_y, 3,
        )
    assert out == (False, None)
    assert any("GPU path failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_gpu_gate_probe_failure_logs_a_warning(monkeypatch, caplog) -> None:
    """A failing CUDA probe forces the CPU discretize path and says so at WARNING."""
    reset_throttle_counts()
    monkeypatch.delenv("MLFRAME_FE_GPU_DISCRETIZE", raising=False)
    import mlframe.feature_selection.filters._gpu_policy as gp

    def _boom():
        """Stand-in probe that always fails."""
        raise RuntimeError("probe fault")

    monkeypatch.setattr(gp, "cuda_available_for_run", _boom)
    with caplog.at_level(logging.WARNING):
        assert _pairs_gpu_gates._fe_gpu_discretize_enabled_uncached(1000, 100) is False
    assert any("cuda_available_for_run() check failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_batched_leader_usability_failure_logs_a_warning(caplog) -> None:
    """A failing batched |corr| capability falls to the per-leader path and says so at WARNING."""
    reset_throttle_counts()

    def _boom(*a):
        """Stand-in batched capability that always fails."""
        raise RuntimeError("batched fault")

    resolve = SimpleNamespace(abs_corr_many=_boom)
    with caplog.at_level(logging.WARNING):
        got = _pairs_emit._batched_leader_usability(resolve, [(0, "a", 0), (0, "b", 1)], np.zeros(4), np.ones(4, dtype=bool), True)
    assert got is None
    assert any("batched leader |corr| failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_extval_per_candidate_loop_fallback_logs_a_warning(caplog) -> None:
    """When even the minimal ext-val batch is refused by the RAM budget the pair drops to the per-candidate loop and says so."""
    reset_throttle_counts()

    class _FakeX:
        """Frame stand-in with only a length."""

        def __len__(self) -> int:
            """Row count large enough that one factor against every op exceeds the minimal-batch cap."""
            return 5_000_000

    resolve = lambda i: np.zeros(4)
    with caplog.at_level(logging.WARNING):
        out = _pairs_emit_extval._emit_pair_features_step2_read_here_once_part2(
            None, 0, True, lambda: False, resolve, None, SimpleNamespace(), None, {}, {"mul": np.multiply, "add": np.add}, ((0, "identity"), (1, "identity")),
            "mul", [1, 2], 0, np.random.default_rng(0), lambda v: np.zeros(4), _FakeX(), lambda *a, **k: (False, 0, 0), 1,
        )
    assert out[7] is False
    assert any("scoring one candidate at a time" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
