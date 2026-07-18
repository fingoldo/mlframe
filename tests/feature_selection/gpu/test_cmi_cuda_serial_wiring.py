"""Wiring + parity gate for the batched-GPU conditional-MI cache pre-fill on the SERIAL MRMR path.

The batched CUDA conditional-MI kernel (``info_theory._cmi_cuda``) is bit-parity (<~1e-9) with the
serial CPU ``conditional_mi``. The parallel redundancy path (``_evaluate_candidates_inner``, used
when ``n_workers > 1``) already pre-populates ``cached_cond_MIs`` with batched I(X; Y | Z) so the
@njit ``evaluate_gain`` loop hits the cache and skips the scalar path. This test covers the SERIAL
path (``_confirm_predictor.score_candidates``, ``n_workers=1`` -- the DEFAULT), which now pre-fills
the shared python ``cached_cond_MIs`` dict in place BEFORE the per-candidate loop via the SAME
``evaluation._prefill_cond_MIs_gpu`` helper. The serial loop has no concurrent iteration, so the
in-place python-dict write is race-free.

HARD GATE: MRMR's final selection + per-feature gain MUST be IDENTICAL with the GPU pre-fill ON vs
OFF at ``n_workers=1`` (the kernel is bit-parity, so selection cannot change). We also assert the
pre-fill engages (the batched dispatch is called and seeds cache slots) when ON and is never called
when the kill-switch is OFF. Each pre-seeded cache slot is one scalar ``conditional_mi`` evaluation
the njit loop skips -> the measured ``prefilled slots`` is the scalar-call reduction.

RAM-light by mandate: n ~ 8000, p ~ 300, single process, n_workers=1.
"""

from __future__ import annotations

import warnings
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR
from mlframe.feature_selection.filters.info_theory._cmi_cuda import cupy_available
from mlframe.feature_selection.filters import evaluation as _eval_mod
import mlframe.feature_selection.filters._confirm_predictor as _confirm_mod
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
    """Fit mrmr."""
    mrmr = MRMR(
        full_npermutations=3,
        baseline_npermutations=3,
        fe_max_steps=0,  # keep FE off: this test is about the redundancy CMI path only
        use_simple_mode=False,  # MUST be off so the conditional-MI redundancy branch runs
        max_runtime_mins=None,
        verbose=0,
        # SERIAL path: n_workers=1 routes through _confirm_predictor.score_candidates' else-branch
        # (the per-candidate loop), which is where the serial GPU CMI pre-fill is wired.
        n_workers=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr.fit(X, y)
    return mrmr


def _selection(mrmr):
    """Helper that selection."""
    return [str(n) for n in mrmr.get_feature_names_out()]


def _run(monkeypatch, gpu_on: bool, count: bool = True):
    """Fit MRMR (n_workers=1) with the GPU CMI pre-fill ON or OFF.

    Returns (selection, n_dispatch_calls, n_prefilled_slots, wall_s). ``n_prefilled_slots`` is the
    total number of cache entries the serial pre-fill wrote = scalar ``conditional_mi`` evaluations
    the njit loop avoided."""
    if gpu_on:
        monkeypatch.delenv("MLFRAME_MRMR_GPU_CMI", raising=False)
    else:
        monkeypatch.setenv("MLFRAME_MRMR_GPU_CMI", "0")

    n_dispatch = {"n": 0}
    n_slots = {"n": 0}
    if count:
        _orig_dispatch = _cmi_cuda.conditional_mi_batched_dispatch

        def _counting_dispatch(*a, **k):
            """Counting dispatch."""
            n_dispatch["n"] += 1
            return _orig_dispatch(*a, **k)

        monkeypatch.setattr(_cmi_cuda, "conditional_mi_batched_dispatch", _counting_dispatch)

        # Wrap the helper as imported into the SERIAL module to count slots written by the serial path.
        _orig_prefill = _confirm_mod._prefill_cond_MIs_gpu

        def _counting_prefill(*a, **k):
            """Counting prefill."""
            written = _orig_prefill(*a, **k)
            n_slots["n"] += int(written or 0)
            return written

        monkeypatch.setattr(_confirm_mod, "_prefill_cond_MIs_gpu", _counting_prefill)

    X, y = _make_frame()
    MRMR._FIT_CACHE.clear()
    t0 = timer()
    mrmr = _fit_mrmr(X, y)
    wall = timer() - t0
    return _selection(mrmr), n_dispatch["n"], n_slots["n"], wall


def test_serial_prefill_helper_imported():
    """The serial module imports the EXACT same helper (no reimplementation)."""
    assert _confirm_mod._prefill_cond_MIs_gpu is _eval_mod._prefill_cond_MIs_gpu


def test_serial_prefill_no_op_when_no_gpu_keeps_cpu_path(monkeypatch):
    """No-GPU graceful path: env switch ON but no CUDA -> MRMR still fits and produces a valid
    selection (dispatch falls back to the exact CPU loop / pre-fill is a harmless no-op). Always run."""
    if _HAS_GPU:
        pytest.skip("GPU present; CPU-fallback graceful path covered by the parity test")
    sel, _nd, _ns, _w = _run(monkeypatch, gpu_on=True, count=False)
    assert isinstance(sel, list)
    assert len(sel) >= 1


@pytest.mark.skipif(not _HAS_GPU, reason="no CUDA/cupy GPU available")
def test_serial_selection_identical_gpu_prefill_on_vs_off(monkeypatch, capsys):
    """HARD PARITY GATE (serial path): GPU pre-fill ON vs OFF at n_workers=1 -> IDENTICAL selection.

    Also confirms (a) the serial pre-fill engages when ON (batched dispatch called + cache slots
    seeded) and (b) the dispatch is NEVER called when the kill-switch is OFF (pure scalar path)."""
    sel_off, nd_off, ns_off, wall_off = _run(monkeypatch, gpu_on=False)
    sel_on, nd_on, ns_on, wall_on = _run(monkeypatch, gpu_on=True)

    # Parity: exact selection + order.
    assert sel_on == sel_off, f"serial GPU pre-fill changed selection!\n  on ={sel_on}\n  off={sel_off}"

    # Engagement / scalar-call reduction.
    assert nd_off == 0, f"dispatch called {nd_off} times with kill-switch OFF (expected 0)"
    assert ns_off == 0, f"pre-fill seeded {ns_off} slots with kill-switch OFF (expected 0)"
    assert nd_on >= 1, "serial GPU pre-fill never invoked the batched dispatch with the switch ON"
    assert ns_on >= 1, "serial GPU pre-fill seeded 0 cache slots with the switch ON"

    with capsys.disabled():
        print(
            f"\n[cmi-cuda-serial-wiring] selection identical ({len(sel_on)} feats); "
            f"dispatch calls on={nd_on} off={nd_off}; "
            f"scalar conditional_mi calls avoided (prefilled slots)={ns_on}; "
            f"wall on={wall_on:.2f}s off={wall_off:.2f}s (delta={wall_off - wall_on:+.2f}s)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
