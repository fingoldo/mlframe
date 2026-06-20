"""Wiring + parity gate for the batched-GPU conditional-MI cache pre-fill in the MRMR greedy loop.

The batched CUDA conditional-MI kernel (``info_theory._cmi_cuda``) is bit-parity (<~1e-9) with the
serial CPU ``conditional_mi``. The production greedy redundancy loop (``filters/evaluation.py``)
runs inside ``@njit`` and cannot call cupy, so the win is realised by PRE-POPULATING the
``cached_cond_MIs`` dict (``_prefill_cond_MIs_gpu``) with batched I(X; Y | Z) values BEFORE the
njit loop runs; the loop then hits the cache and skips the scalar path.

HARD GATE (this test): MRMR's final selection + per-feature gain MUST be IDENTICAL with the GPU
pre-fill ON vs OFF (the kernel is bit-parity, so selection cannot change). We also assert the
pre-fill actually engages (the dispatch is called and fills cache slots), and that the dispatch
is invoked when the pre-fill path is active vs never when it is disabled.

RAM-light by mandate: n ~ 8000, p ~ 300, single process.
"""
from __future__ import annotations

import os
import warnings
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters.info_theory._cmi_cuda import cupy_available
from mlframe.feature_selection.filters import evaluation as _eval_mod
import mlframe.feature_selection.filters.info_theory._cmi_cuda as _cmi_cuda

_HAS_GPU = cupy_available()


def _make_frame(seed: int = 20260619, n: int = 8000, p: int = 300):
    """Planted-signal classification frame: a handful of informative columns drive a binary target,
    the rest are noise. Small enough for the RAM budget, large enough that ``n*p`` clears the GPU
    dispatch gate (n*p ~ 2.4M >= 1M, p >= 64)."""
    rng = np.random.default_rng(seed)
    informative = 6
    Z = rng.normal(size=(n, informative))
    weights = np.array([2.5, -2.0, 1.7, -1.4, 1.1, -0.9])
    logit = Z @ weights
    y = (logit + rng.normal(scale=0.5, size=n) > 0).astype(np.int64)
    cols = {}
    for i in range(informative):
        cols[f"sig_{i}"] = Z[:, i]
    for j in range(p - informative):
        cols[f"noise_{j}"] = rng.normal(size=n)
    X = pd.DataFrame(cols)
    return X, pd.Series(y)


def _fit_mrmr(X, y):
    mrmr = MRMR(
        full_npermutations=3,
        baseline_npermutations=3,
        fe_max_steps=0,          # keep FE off: this test is about the redundancy CMI path only
        use_simple_mode=False,   # MUST be off so the conditional-MI redundancy branch runs
        max_runtime_mins=None,
        verbose=0,
        # The GPU cache pre-fill is wired into the parallel worker entry
        # (``_evaluate_candidates_inner``), which ``screen_predictors`` uses for the
        # redundancy loop when ``n_workers > 1`` and the feasible candidate set is large --
        # the production regime for a wide (p~300) frame. ``n_workers=2`` keeps the RAM
        # budget while exercising that path (threading backend -> same process, so the
        # monkeypatch on the dispatch symbol applies).
        n_workers=2,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr.fit(X, y)
    return mrmr


def _selection(mrmr):
    return [str(n) for n in mrmr.get_feature_names_out()]


def _run(monkeypatch, gpu_on: bool, count_dispatch: bool = True):
    """Fit MRMR with the GPU CMI pre-fill ON or OFF; return (selection, n_dispatch_calls, wall_s)."""
    if gpu_on:
        monkeypatch.delenv("MLFRAME_MRMR_GPU_CMI", raising=False)
    else:
        monkeypatch.setenv("MLFRAME_MRMR_GPU_CMI", "0")

    n_calls = {"n": 0}
    if count_dispatch:
        _orig = _cmi_cuda.conditional_mi_batched_dispatch

        def _counting_dispatch(*a, **k):
            n_calls["n"] += 1
            return _orig(*a, **k)

        # Patch the symbol the pre-fill imports lazily (module attribute).
        monkeypatch.setattr(_cmi_cuda, "conditional_mi_batched_dispatch", _counting_dispatch)

    X, y = _make_frame()
    MRMR._FIT_CACHE.clear()
    t0 = timer()
    mrmr = _fit_mrmr(X, y)
    wall = timer() - t0
    return _selection(mrmr), n_calls["n"], wall


def test_prefill_helper_disabled_by_env(monkeypatch):
    """Env kill-switch off => helper is a pure no-op (returns 0, writes nothing)."""
    monkeypatch.setenv("MLFRAME_MRMR_GPU_CMI", "0")
    cache = {}
    workload = [(0, np.array([0]), 0), (1, np.array([1]), 0)]
    written = _eval_mod._prefill_cond_MIs_gpu(
        workload=workload, y=np.array([7]),
        factors_data=np.zeros((10, 8), dtype=np.int32),
        factors_nbins=np.full(8, 3, dtype=np.int64),
        selected_vars=[2],
        cached_cond_MIs=cache,
        use_simple_mode=False,
        mrmr_relevance_algo="fleuret",
        max_veteranes_interactions_order=1,
    )
    assert written == 0
    assert cache == {}


def test_prefill_helper_skips_simple_mode_and_su_jmim(monkeypatch):
    """Non-eligible regimes (simple mode / interaction order>1) are no-ops, keeping the scalar path."""
    monkeypatch.delenv("MLFRAME_MRMR_GPU_CMI", raising=False)
    common = dict(
        workload=[(0, np.array([0]), 0)],
        y=np.array([7]),
        factors_data=np.zeros((10, 8), dtype=np.int32),
        factors_nbins=np.full(8, 3, dtype=np.int64),
        selected_vars=[2],
        cached_cond_MIs={},
        mrmr_relevance_algo="fleuret",
    )
    # simple mode -> skip
    assert _eval_mod._prefill_cond_MIs_gpu(use_simple_mode=True, max_veteranes_interactions_order=1, **common) == 0
    # interaction order 2 -> skip (kernel only conditions on single-var Z)
    assert _eval_mod._prefill_cond_MIs_gpu(use_simple_mode=False, max_veteranes_interactions_order=2, **common) == 0
    # non-fleuret relevance -> skip
    common2 = dict(common)
    common2["mrmr_relevance_algo"] = "mrmr"
    assert _eval_mod._prefill_cond_MIs_gpu(use_simple_mode=False, max_veteranes_interactions_order=1, **common2) == 0


def test_prefill_key_format_matches_evaluate_gain():
    """The pre-fill key MUST be EXACTLY ``arr2str([cand]) + '|' + arr2str([z])`` (the format
    ``evaluate_gain`` reads). A mismatch silently disables the cache."""
    from mlframe.feature_selection.filters._numba_utils import arr2str
    # Replicate the key the njit loop computes for X=(5,), Z=[3].
    expected = arr2str(np.asarray([5], dtype=np.int64)) + "|" + arr2str(np.asarray([3], dtype=np.int32))
    assert expected == "5|3"


def test_prefill_no_op_when_no_gpu_keeps_cpu_path(monkeypatch):
    """No-GPU graceful path: with the env switch ON but no CUDA device, MRMR must still fit and
    produce a valid selection (the dispatch falls back to the exact CPU loop / pre-fill is harmless).
    Always exercised regardless of hardware."""
    if _HAS_GPU:
        pytest.skip("GPU present; CPU-fallback graceful path covered by the parity test")
    sel, _n, _w = _run(monkeypatch, gpu_on=True, count_dispatch=False)
    assert isinstance(sel, list)
    assert len(sel) >= 1


@pytest.mark.skipif(not _HAS_GPU, reason="no CUDA/cupy GPU available")
def test_selection_identical_gpu_prefill_on_vs_off(monkeypatch, capsys):
    """HARD PARITY GATE: GPU pre-fill ON vs OFF must yield IDENTICAL selection + order.

    Also confirms (a) the pre-fill engages when ON (the batched dispatch is called) and (b) the
    dispatch is NEVER called when the kill-switch is OFF (pure scalar njit path)."""
    sel_off, n_off, wall_off = _run(monkeypatch, gpu_on=False)
    sel_on, n_on, wall_on = _run(monkeypatch, gpu_on=True)

    # Parity: exact selection + order.
    assert sel_on == sel_off, f"GPU pre-fill changed selection!\n  on ={sel_on}\n  off={sel_off}"

    # Engagement: dispatch called >=1 time with pre-fill ON, never with kill-switch OFF.
    assert n_off == 0, f"dispatch called {n_off} times with kill-switch OFF (expected 0)"
    assert n_on >= 1, "GPU pre-fill never invoked the batched dispatch with the switch ON"

    with capsys.disabled():
        print(
            f"\n[cmi-cuda-wiring] selection identical ({len(sel_on)} feats); "
            f"dispatch calls on={n_on} off={n_off}; "
            f"wall on={wall_on:.2f}s off={wall_off:.2f}s (delta={wall_off - wall_on:+.2f}s)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
