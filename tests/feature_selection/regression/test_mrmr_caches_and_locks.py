"""Behavioural tests for the MRMR cache and lock discipline: keyed caches, memo tables and concurrent access."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


def test_regression_cmi_resident_cache_has_lock():
    """Pins that _CMI_RESIDENT_CACHE now has a dedicated lock (was previously the only one of 3
    sibling caches in this file with none) and that clear_cmi_resident_cache() acquires it too.
    """
    from mlframe.feature_selection.filters.info_theory import _cmi_cuda as cmi_cuda_mod

    assert hasattr(cmi_cuda_mod, "_CMI_RESIDENT_CACHE_LOCK")
    import threading

    assert isinstance(cmi_cuda_mod._CMI_RESIDENT_CACHE_LOCK, type(threading.Lock()))
    # Sanity: clear works and leaves the cache empty (also exercises the lock without deadlocking).
    cmi_cuda_mod._CMI_RESIDENT_CACHE["dummy"] = ("g", "sig")
    cmi_cuda_mod.clear_cmi_resident_cache()
    assert cmi_cuda_mod._CMI_RESIDENT_CACHE == {}


# ---------------------------------------------------------------------------
# INFO_THEORY_B-1 (P1): genie_mi_panel's default bias-rate fallback was the SAME
# constant (1/sqrt(N)) for every estimator name, making genie_weights' constraint
# matrix exactly singular by construction -- estimator='genie' silently collapsed
# to a plain unweighted mean on every production call.
# ---------------------------------------------------------------------------


def test_regression_dy_device_cache_has_lock():
    """Pre-fix: no lock guarded _DY_DEVICE_CACHE's LRU bookkeeping. Post-fix: both the numba.cuda
    and cupy-native caches are guarded by their own threading.Lock."""
    import threading

    import mlframe.feature_selection.filters.batch_mi_noise_gate_gpu as bng_gpu

    assert isinstance(bng_gpu._DY_DEVICE_CACHE_LOCK, type(threading.Lock()))
    assert isinstance(bng_gpu._DY_DEVICE_CACHE_CUPY_LOCK, type(threading.Lock()))

    # A lock that EXISTS is not a lock that is USED, and the defect this guards was unguarded LRU bookkeeping,
    # not an absent lock object -- `isinstance(..., Lock)` passes for a lock nothing ever acquires. Assert it is
    # actually ACQUIRED during a real cache call, by counting acquisitions on a spy.
    _acquired = {"n": 0}
    _real_lock = bng_gpu._DY_DEVICE_CACHE_LOCK

    class _CountingLock:
        """Delegates to the real lock, counting context-manager entries."""

        def __enter__(self):
            """Count and acquire."""
            _acquired["n"] += 1
            return _real_lock.__enter__()

        def __exit__(self, *exc):
            """Release."""
            return _real_lock.__exit__(*exc)

    import collections
    import contextlib
    import types
    import unittest.mock as _mock

    uploads: list = []

    def _fake_to_device(arr):
        """Host stand-in for the device upload so the cache bookkeeping runs without a GPU."""
        uploads.append(arr.shape)
        return arr

    _classes_y = np.arange(16, dtype=np.int64) % 2
    with contextlib.ExitStack() as stack:
        stack.enter_context(_mock.patch.object(bng_gpu, "_DY_DEVICE_CACHE_LOCK", _CountingLock()))
        stack.enter_context(_mock.patch.object(bng_gpu, "_DY_DEVICE_CACHE", collections.OrderedDict()))
        stack.enter_context(_mock.patch.object(bng_gpu, "_nb_cuda", types.SimpleNamespace(to_device=_fake_to_device)))
        first = bng_gpu._resident_y_all_device(_classes_y, _classes_y, base_seed=0, nperm=0, n=16, P=1)
        second = bng_gpu._resident_y_all_device(_classes_y, _classes_y, base_seed=0, nperm=0, n=16, P=1)
    assert first is second
    assert uploads == [(1, 16)]
    # miss: lookup + store; hit: lookup only
    assert _acquired["n"] == 3, "the device-cache lock must guard the lookup and the store of every cache call"


# ---------------------------------------------------------------------------
# GPU_INFRA_B-1 (P1): gpu_materialise_discretize_codes_host / gpu_discretize_codes_host both called
# clear_resident_codes_handoff() with NO argument at entry -- the blanket, whole-dict-clear form --
# instead of the module's own targeted per-host-array clear form, silently dropping another concurrent
# thread's still-pending deferred-fill entry under joblib threading.
# ---------------------------------------------------------------------------


def test_regression_install_cuda_teardown_guard_has_lock():
    """Pre-fix: no lock guarded the idempotency check. Post-fix: a module-level threading.Lock exists and
    guards the whole install sequence."""
    import threading

    from mlframe.feature_selection.filters import _gpu_teardown_guard as guard

    assert isinstance(guard._install_lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# GPU_INFRA_C-4 (P2): resident_bincount's docstring claimed OOB indices are "undefined behaviour (same as
# feeding cupy.bincount a bad minlength)" -- but cupy.bincount raises loudly on a negative index, while
# resident_bincount's cupyx.scatter_add silently wraps/corrupts. Added an opt-in debug_check_bounds guard.
# ---------------------------------------------------------------------------


def test_regression_gpu_pairs_shared_mem_set_has_lock():
    """Pre-fix: no lock guarded the property-set + launch pair. Post-fix: both files expose a
    threading.Lock guarding the sequence."""
    import threading

    import mlframe.feature_selection.filters._gpu_pairs as gp
    import mlframe.feature_selection.filters._batch_pair_mi_cuda_shared_fused as bpf

    assert isinstance(gp._SHARED_MEM_SET_LOCK, type(threading.Lock()))
    assert isinstance(bpf._SHARED_MEM_SET_LOCK, type(threading.Lock()))


# ---------------------------------------------------------------------------
# GPU_INFRA_D-3 (P1): run_numba_kernel_search / optimize_all_pairs_numba_kernel silently substituted 1 for
# any seed<=0 (including the valid seed=0), diverging from the cupy twin's plain np.random.default_rng(seed).
# ---------------------------------------------------------------------------


def test_regression_bayesian_blocks_default_subsample_threshold_is_safe(monkeypatch):
    """Pre-fix: per_feature_edges passed bb_subsample_threshold=0 (disabled, unbounded O(N^2)) by default.
    Post-fix: the default is a safe, already-validated cap. Observed at the real call boundary: the
    edges_bayesian_blocks call that per_feature_edges makes with no bb_subsample_threshold kwarg must receive
    the shared positive default, not 0."""
    import mlframe.feature_selection.filters._adaptive_nbins as an_mod
    from mlframe.feature_selection.filters._adaptive_nbins import _BB_DEFAULT_SUBSAMPLE_THRESHOLD, per_feature_edges

    seen_thresholds: list = []
    _orig = an_mod.edges_bayesian_blocks

    def _spy(*args, **kwargs):
        """Record the subsample_threshold per_feature_edges forwards, then delegate to the real builder."""
        seen_thresholds.append(kwargs.get("subsample_threshold"))
        return _orig(*args, **kwargs)

    monkeypatch.setattr(an_mod, "edges_bayesian_blocks", _spy)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6000, 1))
    per_feature_edges(X, None, method="bayesian_blocks", n_jobs=1)

    assert seen_thresholds, "per_feature_edges never reached edges_bayesian_blocks"
    # A 0 default restores the unbounded full-N bayesian-blocks DP, which is the regression.
    assert seen_thresholds == [_BB_DEFAULT_SUBSAMPLE_THRESHOLD], f"bb subsample threshold forwarded as {seen_thresholds!r}"
    assert _BB_DEFAULT_SUBSAMPLE_THRESHOLD > 0


# ---------------------------------------------------------------------------
# DISCRETIZATION-3 (P1): mdlp_bin_edges_oos_validated over-split on exact-duplicate (x, y) rows -- unlike
# its two siblings that dedupe before recursing. The root cause was deeper than a missing _dedupe_xy call:
# a duplicate row could straddle the train/holdout split (leaking the identical point into "held-out"
# validation), so deduping only the train fold was insufficient -- the fix dedupes the FULL array before
# the train/holdout split happens at all.
# ---------------------------------------------------------------------------


def test_regression_fourier_detect_gpu_resident_caches_have_lock():
    """Pre-fix: no lock guarded the get-or-insert sequence on either cache. Post-fix: both are guarded by
    a shared threading.Lock."""
    import threading

    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_gpu_resident as fd

    assert isinstance(fd._CACHE_LOCK, type(threading.Lock()))


# ---------------------------------------------------------------------------
# ORTH_BASIS_B-4 (P2): _orth_extra_basis_fe.py <-> _orth_extra_basis_fe_generate.py had a circular
# TOP-LEVEL import (each imports names from the other at module scope), which only resolved because every
# current caller happens to trigger _orth_extra_basis_fe.py's import first. A future caller importing
# _orth_extra_basis_fe_generate directly would have hit ImportError on a partially-initialized module.
# ---------------------------------------------------------------------------


def test_regression_oracle_rows_cache_concurrent_access_no_crash():
    """Pre-fix: 32 threads x 500 calls against distinct fake stores reproduced ~225 crashes under tightened
    thread-switch interval. Post-fix: the same stress pattern (reduced scale for test speed) completes with
    zero exceptions, since the whole get-or-insert sequence is now lock-guarded."""
    import sys
    import threading

    from mlframe.feature_selection.filters._oracle_scorer_select import _cached_read_rows, _ROWS_CACHE

    class _FakeStore:
        """In-memory stand-in for the persistence layer, so the test never touches real storage."""

        def __init__(self, path, rows):
            """Minimal stand-in constructor for the double used below."""
            self._path = path
            self._rows = rows

        def read_rows(self):
            """Test helper: read rows."""
            return list(self._rows)

    _ROWS_CACHE.clear()
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force aggressive interleaving to actually exercise the race window
    try:
        stores = [_FakeStore(f"/fake/store_{i}.parquet", [{"i": i}]) for i in range(20)]
        errors: list[BaseException] = []

        def _worker():
            """Thread body: exercises the call under test and records its result or error."""
            rng = np.random.default_rng()
            for _ in range(300):
                store = stores[rng.integers(0, len(stores))]
                try:
                    _cached_read_rows(store)
                except BaseException as exc:
                    errors.append(exc)

        # os.path.getmtime on a nonexistent fake path raises OSError -> _cached_read_rows falls back to the
        # uncached path for these fixtures, which still exercises _ROWS_CACHE.get() at the top under
        # concurrent access from the mtime=-1.0 branch... to actually hit the cache dict, patch getmtime.
        import mlframe.feature_selection.filters._oracle_scorer_select as mod

        orig_isfile = mod.os.path.isfile
        mod.os.path.isfile = lambda p: True  # nosec B110 - test monkeypatch, restored in finally
        mod.os.path.getmtime = lambda p: 1.0  # nosec B110 - fixed mtime so all calls for a store share one cache key

        threads = [threading.Thread(target=_worker) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        mod.os.path.isfile = orig_isfile
    finally:
        sys.setswitchinterval(old_interval)
    assert not errors, f"concurrent _cached_read_rows access raised: {errors[:3]}"


# ---------------------------------------------------------------------------
# ORTH_SCORING_B-4 (P2): _orthogonal_scorer_auto_fe.py's nested _call() helper's "plug_in"/"copula"
# branches are unreachable dead code (both names are always in _BATCHABLE, routed through _batch_scores
# instead) -- documented, not deleted, as the correct fallback contract.
# ---------------------------------------------------------------------------


def test_regression_gpu_resident_materialise_operand_caches_have_lock():
    """Pre-fix: no lock guarded either cache's get-or-insert sequence. Post-fix: both are guarded by a
    dedicated threading.Lock."""
    import threading

    import mlframe.feature_selection.filters._gpu_resident_materialise as mod

    assert isinstance(mod._OPERAND_TABLE_CACHE_LOCK, type(threading.Lock()))
    assert isinstance(mod._PREBUILT_OPERAND_TABLE_LOCK, type(threading.Lock()))


# ---------------------------------------------------------------------------
# FE_PAIRS_CORE-2 (P2): _fe_gpu_discretize_enabled/_fe_gpu_binning_enabled were called up to 3x per pair,
# once per chunk, and once per ext-val tied-leader-set, each re-reading env vars + calling
# is_cuda_available() + a kernel_tuning_cache lookup -- unlike their hoisted sibling _fe_env_gate.
# ---------------------------------------------------------------------------


def test_regression_fe_gpu_gates_are_memoized():
    """Pre-fix: every call re-did the full env/CUDA/KTC lookup. Post-fix: repeated calls with the same
    (n_rows, n_cands) hit an in-process cache instead of recomputing."""
    # The gates and their memo live in _pairs_gpu_gates (carved out of _pairs_core), so the uncached function is patched where it is looked up.
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_gpu_gates as mod

    mod._GPU_GATE_CACHE.clear()
    call_count = {"n": 0}
    orig = mod._fe_gpu_discretize_enabled_uncached

    def _counting(*a, **k):
        """Test helper: counting."""
        call_count["n"] += 1
        return orig(*a, **k)

    mod._fe_gpu_discretize_enabled_uncached = _counting
    try:
        r1 = mod._fe_gpu_discretize_enabled(12345, 67)
        r2 = mod._fe_gpu_discretize_enabled(12345, 67)
        assert r1 == r2
        assert call_count["n"] == 1, f"expected exactly 1 uncached call for a repeated (n_rows, n_cands); got {call_count['n']}"
    finally:
        mod._fe_gpu_discretize_enabled_uncached = orig
        mod._GPU_GATE_CACHE.clear()


# ---------------------------------------------------------------------------
# FE_PAIRS_CORE-3 (P2): the "Reject no-variance operands (a constant gate is dead)" comment in
# _pairs_setup.py did not match the code -- the only guard was an isfinite check, which does not reject a
# literal constant (finite-median) operand.
# ---------------------------------------------------------------------------


def test_regression_fe_gate_memo_caches_have_lock():
    """Pre-fix: no lock guarded any of the 4 caches' eviction. Post-fix: both modules expose a
    _FE_GATE_MEMO_LOCK guarding their respective caches."""
    import threading

    from mlframe.feature_selection.filters import _unified_fe_gate as ufg
    from mlframe.feature_selection.filters import _fe_accuracy_gate as fag

    assert isinstance(ufg._FE_GATE_MEMO_LOCK, type(threading.Lock()))
    assert isinstance(fag._FE_GATE_MEMO_LOCK, type(threading.Lock()))


def test_regression_fe_gate_memo_caches_concurrent_access_no_crash():
    """Stress test mirroring the audit's proposed regression: call raw_mi_noise_floor / infer_classification
    from many threads with distinct-content keys (forcing repeated FIFO eviction) and assert no KeyError."""
    import threading

    from mlframe.feature_selection.filters._unified_fe_gate import raw_mi_noise_floor
    from mlframe.feature_selection.filters._fe_accuracy_gate import infer_classification

    errors: list[BaseException] = []

    def _worker(seed):
        """Thread body: exercises the call under test and records its result or error."""
        rng = np.random.default_rng(seed)
        try:
            for _ in range(20):
                n = 200
                X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
                y = rng.standard_normal(n)
                raw_mi_noise_floor(X, y)
                infer_classification(y)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent FE gate memo access raised: {errors[:3]}"


# ---------------------------------------------------------------------------
# FE_ORCH_BUDGET-3 (P2): FeatureMatrix.numeric_column / from_feature_matrix both resolved a column by
# columns.index(name), which returns only the first index for a duplicate column name -- silently
# corrupting/dropping data on round-trip instead of raising.
# ---------------------------------------------------------------------------


def test_regression_y_dense_memo_has_lock():
    """Pre-fix: no lock guarded _Y_DENSE_MEMO. Post-fix: a dedicated threading.Lock exists."""
    import threading

    from mlframe.feature_selection.filters import _fe_cmi_redundancy_gate as mod

    assert isinstance(mod._Y_DENSE_MEMO_LOCK, type(threading.Lock()))


def test_regression_y_dense_memo_concurrent_access_no_crash():
    """Stress test: call the y-dense-memoizing gate path from many threads with distinct-content y arrays
    (forcing repeated FIFO eviction) and assert no RuntimeError/KeyError."""
    import threading

    from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import apply_cmi_redundancy_gate

    errors: list[BaseException] = []

    def _worker(seed):
        """Thread body: exercises the call under test and records its result or error."""
        rng = np.random.default_rng(seed)
        try:
            for _ in range(15):
                n = 300
                names = ["a", "b", "c"]
                candidates = {nm: (rng.standard_normal(n), float(rng.uniform(0.01, 0.5))) for nm in names}
                y = rng.integers(0, 3, n).astype(np.int64)
                apply_cmi_redundancy_gate(candidates, y)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"concurrent _Y_DENSE_MEMO access raised: {errors[:3]}"


# ---------------------------------------------------------------------------
# FE_REDUNDANCY_SYNERGY-8/9/10 (P2): stale "NOT YET WIRED" docstring, stale self-referential log-line-
# number strings across 6 files, and an unlogged except-Exception swallow in gbm_split_propensity.
# ---------------------------------------------------------------------------


def test_regression_residency_audit_has_lock():
    """Direct pin: residency_audit is now serialized via a module-level lock."""
    from mlframe.feature_selection.filters._gpu_strict_fe import _audit as mod

    assert hasattr(mod, "_AUDIT_LOCK")
    import threading

    assert isinstance(mod._AUDIT_LOCK, type(threading.RLock()))


def test_regression_pack_blocks_to_devices_rejects_empty_speeds():
    """Regression: pack_blocks_to_devices treated a genuinely empty speeds list (zero visible devices) identically to the single-device case, returning a
    bogus device index 0.
    """
    from mlframe.feature_selection.filters._fe_gpu_batch._packer import pack_blocks_to_devices

    with pytest.raises(ValueError, match="speeds is empty"):
        pack_blocks_to_devices([10, 20], [])


def test_regression_pack_blocks_to_devices_single_device_still_works():
    """Sanity: the legitimate single-device fast path is unaffected."""
    from mlframe.feature_selection.filters._fe_gpu_batch._packer import pack_blocks_to_devices

    assert pack_blocks_to_devices([10, 20], [1.0]) == [0, 0]


# ---------------------------------------------------------------------------
# X_EDGE_CASES_BEST_PRACTICES-1 (P1): the resident-operand cache was keyed purely on content (no device
# component), so a multi-GPU worker on device N could receive a device-M-resident array back from the
# cache on a content hit. Also: no lock guarding the get/insert/evict sequence, and a bare
# `except Exception: pass` in the multi-GPU fallback path with zero diagnostic trace.
# ---------------------------------------------------------------------------


def test_regression_resident_operand_cache_has_lock():
    """Regression guard for resident operand cache has lock."""
    import threading

    from mlframe.feature_selection.filters._fe_resident_operands import _FE_RESIDENT_OPERANDS_LOCK

    # Behaviour: while the lock is held, another thread cannot take it (the get/insert/evict sequence is serialised).
    assert _FE_RESIDENT_OPERANDS_LOCK.acquire(timeout=1)
    try:
        contender_got_it = []
        t = threading.Thread(target=lambda: contender_got_it.append(_FE_RESIDENT_OPERANDS_LOCK.acquire(blocking=False)))
        t.start()
        t.join(timeout=5)
        assert contender_got_it == [False]
    finally:
        _FE_RESIDENT_OPERANDS_LOCK.release()
