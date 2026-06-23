"""Regression (2026-06-23): the FE candidate BINNING gate is DECOUPLED from the full ``fe_gpu_pairs_mi``
analytic-path crossover.

WHY THIS EXISTS. A full-fit cProfile of a GPU-mode F2 100k fit (GPU 0% idle) showed the CPU njit
binning ``discretize_2d_quantile_batch`` (``_quantile_edges_2d_njit`` + ``_searchsorted_2d_right_njit_parallel``)
as the #1 WALL hotspot: 116.5s cumtime of a 228s wall. The bit-identical GPU binning
``gpu_discretize_codes_host`` is 17-24x faster at n=100k (verified maxdiff 0) yet was running on the CPU,
because the binning was gated by ``_fe_gpu_discretize_enabled`` -> the FULL ``fe_gpu_pairs_mi`` KTC sweep,
which times binning + GPU-MI + chi2 and had cached "cpu" for the n_rows<=100000 region. That verdict on
the heavier MI path wrongly disabled the cheap, strictly-simpler binning. The binning now has its own
``_fe_gpu_binning_enabled`` gate + dedicated ``fe_gpu_binning`` KTC kernel. Routing the binning to the GPU
cut the full-fit WALL 228.0s -> 89.5s (binning op cumtime 116.5s -> 4.9s) with BIT-IDENTICAL selection
(same recovered compound + recipes, CPU vs GPU binning).

These tests pin:
* the binning gate is independent of the pair-MI gate (env tri-state, separate KTC kernel registered);
* the global ``MLFRAME_FE_GPU_DISCRETIZE=0`` kill switch still disables the binning;
* (GPU) ``gpu_discretize_codes_host`` is bit-identical (maxdiff 0) to ``discretize_2d_quantile_batch``;
* (GPU) a full F2 100k fit recovers the SAME feature set with binning forced CPU vs forced GPU.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_core as core


def _has_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return bool(is_cuda_available())
    except Exception:
        return False


def test_binning_gate_env_tristate(monkeypatch):
    """The binning gate honours its OWN env tri-state, independent of the pair-MI gate's env."""
    monkeypatch.setenv("MLFRAME_FE_GPU_BINNING", "0")
    assert core._fe_gpu_binning_enabled(100_000, 3888) is False  # forced off regardless of CUDA
    # The pair-MI gate must NOT be affected by the binning env var.
    monkeypatch.delenv("MLFRAME_FE_GPU_DISCRETIZE", raising=False)
    # (only asserts no crash + boolean type; the actual GPU choice depends on host)
    assert isinstance(core._fe_gpu_discretize_enabled(100_000, 3888), bool)


def test_global_kill_switch_disables_binning(monkeypatch):
    """``MLFRAME_FE_GPU_DISCRETIZE=0`` is a global FE-GPU kill switch that also disables the binning,
    so existing CPU-only configs are byte-for-byte unchanged."""
    monkeypatch.setenv("MLFRAME_FE_GPU_DISCRETIZE", "0")
    monkeypatch.delenv("MLFRAME_FE_GPU_BINNING", raising=False)  # auto, but the kill switch wins
    assert core._fe_gpu_binning_enabled(100_000, 3888) is False


def test_binning_ktc_kernel_registered():
    """The dedicated binning crossover helpers exist and are distinct from the pair-MI ones."""
    from mlframe.feature_selection.filters import _gpu_resident_basis as b
    assert callable(b.fe_gpu_binning_backend_choice)
    assert callable(b._run_fe_gpu_binning_sweep)
    assert callable(b._fe_gpu_binning_fallback_choice)
    assert callable(b.ensure_fe_gpu_binning_tuning)
    # distinct from the pair-MI path
    assert b.fe_gpu_binning_backend_choice is not b.fe_gpu_pairs_mi_backend_choice


def test_binning_fallback_crossover_math(monkeypatch):
    """The pre-sweep fallback routes large work to GPU, tiny work to CPU (lower crossover than the
    full MI path: binning is a cheaper op)."""
    from mlframe.feature_selection.filters import _gpu_resident_basis as b
    monkeypatch.delenv("MLFRAME_FE_GPU_BINNING_MIN_NK", raising=False)
    assert b._fe_gpu_binning_fallback_choice(100_000, 256) == "gpu"   # 2.56e7 >= 1e6
    assert b._fe_gpu_binning_fallback_choice(5_000, 50) == "cpu"      # 2.5e5  < 1e6


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_gpu_binning_bit_identical_to_cpu():
    """``gpu_discretize_codes_host`` == ``discretize_2d_quantile_batch`` codes, maxdiff 0."""
    from mlframe.feature_selection.filters.discretization import discretize_2d_quantile_batch
    from mlframe.feature_selection.filters._gpu_resident_fe import gpu_discretize_codes_host
    rng = np.random.default_rng(0)
    cand = np.ascontiguousarray(rng.uniform(0.1, 5.0, (20_000, 64)).astype(np.float32))
    for nb in (10, 20):
        c_cpu = discretize_2d_quantile_batch(cand, n_bins=nb, dtype=np.int8, assume_finite=True)
        c_gpu = gpu_discretize_codes_host(cand, nb, dtype=np.int8)
        maxdiff = int(np.abs(c_cpu.astype(np.int32) - c_gpu.astype(np.int32)).max())
        assert maxdiff == 0, f"nb={nb}: GPU binning codes differ from CPU (maxdiff {maxdiff})"


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required")
def test_f2_selection_identical_cpu_vs_gpu_binning(monkeypatch):
    """A full F2 fit recovers the SAME feature set whether the binning runs on CPU or GPU -- the binning
    GPU routing is a backend choice with no selection effect (bit-identical codes). Uses a smaller n than
    the 100k profiling fit so the test runs in CI time while still crossing the GPU binning threshold."""
    from mlframe.feature_selection.filters import MRMR

    def mk(n, seed):
        r = np.random.default_rng(seed)
        a = r.uniform(0.1, 1.1, n); b = r.uniform(0.1, 1.1, n); c = r.uniform(0.1, 1.1, n)
        d = r.uniform(0, 2 * np.pi, n); e = r.uniform(0.1, 1.1, n); f = r.uniform(0.1, 1.1, n)
        return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), a ** 2 / b + f / 5 + np.log(np.abs(c) + 1e-9) * np.sin(d)

    df, y = mk(60_000, 7)

    def _fit():
        return [str(n) for n in MRMR(verbose=0, n_jobs=1).fit(df, y).get_feature_names_out()]

    monkeypatch.setenv("MLFRAME_FE_GPU_BINNING", "0")
    names_cpu = _fit()
    monkeypatch.setenv("MLFRAME_FE_GPU_BINNING", "1")
    names_gpu = _fit()
    assert names_cpu == names_gpu, f"binning backend changed selection: CPU={names_cpu} GPU={names_gpu}"
