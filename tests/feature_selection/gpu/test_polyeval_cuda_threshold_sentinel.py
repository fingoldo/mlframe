"""Regression: _run_sweep_polyeval must NOT persist the 500k source fallback as cuda_threshold when cuda
never wins on the host -- that value is itself a mis-route (dispatcher routes >=500k to the slower cuda path).
The sweep must emit a sentinel ABOVE the largest swept size so the dispatcher never picks cuda on this HW.

iter142: on the RTX 500 Ada the cheap degree-5 Horner kernel is transfer-bound; cuda loses to njit_par at
every swept n (incl. 1e6). Pre-fix the sweep left cuda_threshold=500_000, so the dispatcher kept routing
large-n calls to cuda (~6.6-8.5x slower than njit_par). Post-fix it emits swept_max*100.
"""

from __future__ import annotations

import pytest


def test_polyeval_sweep_disables_cuda_when_it_never_wins(monkeypatch):
    """Polyeval sweep disables cuda when it never wins."""
    pytest.importorskip("numba")
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import _auto_tune_sweeps_a as mod

    # Force the "cuda never wins" world: cuda path reports a huge time, njit_par is cheap.
    monkeypatch.setattr(mod, "_CUDA_AVAILABLE", True, raising=False)

    # Patch the three backend dicts so njit_par is always fast and cuda is always slow -> cuda never wins.
    import numpy as np

    # O(1) fake: allocating np.zeros_like(x) at n=1e6 is an 8MB alloc that under Windows paging can exceed the
    # 2ms cuda sleep below, making njit_par spuriously the LOSER at large n (flaky). Return a cached tiny array
    # so njit_par is reliably microseconds at every swept n -> the "cuda never wins" world the test asserts.
    _tiny = np.zeros(1)
    fast = lambda x, c: _tiny
    monkeypatch.setattr(mod, "_NJIT_FUNCS", {b: fast for b in ("hermite", "legendre", "chebyshev", "laguerre")}, raising=False)
    monkeypatch.setattr(mod, "_NJIT_PAR_FUNCS", {b: fast for b in ("hermite", "legendre", "chebyshev", "laguerre")}, raising=False)

    # Make _polyeval_cuda deliberately slow so it never beats njit_par.
    import time

    def slow_cuda(basis, x, c):
        """Slow cuda."""
        time.sleep(0.002)
        return np.zeros_like(x)

    monkeypatch.setattr(mod, "_polyeval_cuda", slow_cuda, raising=False)

    regions = mod._run_sweep_polyeval(n_iters=1)
    assert regions, "sweep produced no regions"
    for r in regions:
        # cuda never wins -> threshold must be ABOVE any realistic prod n, never the 500k mis-route default.
        assert r["cuda_threshold"] > 1_000_000, (
            f"basis={r['basis']} cuda_threshold={r['cuda_threshold']} would mis-route large-n to the slower "
            f"cuda path; expected an above-swept-range sentinel when cuda never wins"
        )
        assert r["cuda_threshold"] != 500_000, "must not persist the bare 500k source fallback when cuda loses"
