"""Regression test for the site-2 joblib audit fix (2026-07-19): the FE order-2 pair-MI loky PROCESS pool
(``compute_pair_mis_and_floor``'s ``_run_loky_pair_mi_pool``) is now skipped when ``fe_npermutations`` is
below ``_LOKY_POOL_MIN_FE_NPERMUTATIONS`` (20), regardless of ``n_jobs``/``n_pairs``.

Measured (isolated/warmed, realistic ``fe_npermutations=3`` production regime): the loky pool LOSES to
serial at every pair count tested -- n_pairs=190 -> 0.03x, n_pairs=4950 -> 0.22-0.38x, n_pairs=20000 -> 0.21x
-- with no crossover found. 20 is a conservative floor extrapolated above the tested regime (not itself
measured to win).
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_selection.filters._mrmr_fe_step._step_pairmi as step_pairmi_mod


def _make_inputs(k: int = 12, n: int = 300):
    """Build a small synthetic pool for ``compute_pair_mis_and_floor``."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, size=(n, k + 1)).astype(np.int32)
    nbins = np.array([4] * (k + 1), dtype=np.int32)
    target_indices = (k,)
    classes_y = data[:, k].astype(np.int32)
    freqs_y = np.bincount(classes_y).astype(np.float64)
    numeric_vars_to_consider = set(range(k))
    cols = [f"f{i}" for i in range(k)] + ["y"]
    return data, nbins, target_indices, classes_y, freqs_y, numeric_vars_to_consider, cols


class _Fake:
    """Minimal ``self`` stand-in for ``compute_pair_mis_and_floor``'s attribute reads."""

    fe_max_engineered_operands = -1
    fe_escalation_feedforward_enable = True
    _fe_synergy_exhaustive_active_ = False
    feature_names_in_: list = []


def _run(monkeypatch, fe_npermutations, k=12, n_jobs=8, boom=True):
    """Invoke ``compute_pair_mis_and_floor`` with the batch precompute disabled (so the loky-pool gate is
    the only thing deciding whether pairs get computed) and either a raising or a recording ``LokyBackend``
    stub, returning the recorded calls list."""
    monkeypatch.setenv("MLFRAME_MRMR_BATCH_PAIR_MI", "0")
    data, nbins, target_indices, classes_y, freqs_y, numeric_vars_to_consider, cols = _make_inputs(k=k)
    _Fake.feature_names_in_ = cols[:-1]

    calls = []

    class _Backend:
        """Records/optionally blocks ``LokyBackend`` construction."""

        def __init__(self, *a, **kw):
            calls.append((a, kw))
            if boom:
                raise AssertionError("LokyBackend must not be constructed below _LOKY_POOL_MIN_FE_NPERMUTATIONS")

    monkeypatch.setattr(step_pairmi_mod, "LokyBackend", _Backend)

    step_pairmi_mod.compute_pair_mis_and_floor(
        _Fake(),
        data=data,
        cols=cols,
        nbins=nbins,
        X=None,
        classes_y=classes_y,
        classes_y_safe=classes_y,
        freqs_y=freqs_y,
        target_indices=target_indices,
        cached_MIs={},
        cached_confident_MIs={},
        numeric_vars_to_consider=numeric_vars_to_consider,
        _prevalence_debias_auto=False,
        n_jobs=n_jobs,
        prefetch_factor=2,
        parallel_kwargs={"backend": "threading"},
        fe_min_nonzero_confidence=0.99,
        fe_npermutations=fe_npermutations,
        fe_min_pair_mi=0.001,
        fe_min_pair_mi_prevalence=1.05,
        verbose=0,
    )
    return calls


def test_loky_pool_never_constructed_below_permutation_floor(monkeypatch):
    """``fe_npermutations=3`` (the realistic production default) must route to the serial sweep, never the
    loky pool, even with ``n_jobs`` high enough and enough pairs to normally reach the pool branch."""
    calls = _run(monkeypatch, fe_npermutations=3, boom=True)
    assert calls == [], "LokyBackend was constructed despite fe_npermutations below the measured floor"


def test_loky_pool_constructed_above_permutation_floor(monkeypatch):
    """Above the floor (``fe_npermutations=20``), the pool branch IS still reachable -- the gate only
    changes behaviour BELOW the floor, it does not remove the pool code path entirely. ``boom=False``: the
    stub records the construction but does not raise, since the subsequent ``Parallel(backend=<stub>)`` call
    would otherwise fail in a way this test does not care about (retried internally, logged, and swallowed
    by the function's own failure-recovery path) -- only reachability of ``LokyBackend`` itself is asserted.
    """
    calls = _run(monkeypatch, fe_npermutations=20, boom=False)
    assert len(calls) == 1, "LokyBackend must be constructed once at/above the measured floor"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
