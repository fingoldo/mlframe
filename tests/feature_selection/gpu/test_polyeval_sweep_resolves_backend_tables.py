"""The polyeval tuning sweep must find the real backend tables instead of its empty placeholders.

The sweep module declares ``_NJIT_FUNCS`` / ``_NJIT_PAR_FUNCS`` as empty dicts (type annotation only) and filled them with ``setdefault``, which keeps an existing empty dict. Every basis
warmup then raised ``KeyError: 'hermite'``, the sweep timed nothing and persisted zero regions, so the polyeval backend choice never reached the tuning cache.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection._benchmarks.kernel_tuning_cache import _auto_tune_sweeps_a as sweeps


def test_the_sweep_times_every_basis_and_returns_regions(monkeypatch):
    """With the placeholders in place the sweep still resolves the real tables and returns a region per basis (small axis to stay fast)."""
    monkeypatch.setattr(sweeps, "_NJIT_FUNCS", {})
    monkeypatch.setattr(sweeps, "_NJIT_PAR_FUNCS", {})
    monkeypatch.delitem(vars(sweeps), "_CUDA_AVAILABLE", raising=False)
    monkeypatch.delitem(vars(sweeps), "_polyeval_cuda", raising=False)
    regions = sweeps._run_sweep_polyeval(n_iters=1)
    assert sweeps._NJIT_FUNCS and sweeps._NJIT_PAR_FUNCS
    assert {r.get("basis") for r in regions} >= {"hermite", "legendre", "chebyshev", "laguerre"}


def test_a_monkeypatched_table_still_wins(monkeypatch):
    """A non-empty table set by the caller (tests, host override) is not replaced by the real one."""
    marker = {name: (lambda x, c: np.zeros_like(x)) for name in ("hermite", "legendre", "chebyshev", "laguerre")}
    monkeypatch.setattr(sweeps, "_NJIT_FUNCS", marker)
    monkeypatch.setattr(sweeps, "_NJIT_PAR_FUNCS", marker)
    monkeypatch.setattr(sweeps, "_CUDA_AVAILABLE", False, raising=False)
    sweeps._run_sweep_polyeval(n_iters=1)
    assert sweeps._NJIT_FUNCS is marker
