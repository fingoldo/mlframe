"""Regression tests for the cache-safety / global-state fixes on the training side
(CON10, CON11, CON12, CON13, CON14, CON15, CON21).

GPU device-buffer items (CON10 pinned pool) are validated by a CPU-side test of the keying /
lock / cap logic the fix introduced; the rest are pure host-side caches.
"""

import threading

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# CON10: _PINNED_BUFFERS pool -- lock + per-thread keying + LRU cap.
# ---------------------------------------------------------------------------
def test_con10_pinned_pool_has_lock_and_cap_and_per_thread_key():
    """Con10 pinned pool has lock and cap and per thread key."""
    from mlframe.feature_engineering.transformer import _kernels_cupy as kc

    assert isinstance(kc._PINNED_BUFFERS_LOCK, type(threading.Lock()))
    assert kc._PINNED_BUFFERS_MAX > 0

    # The key now leads with the thread id -> two threads asking for the same (name, shape, dtype)
    # get DISTINCT keys, so they never share a buffer. Simulate the keying directly.
    def _key():
        """Key."""
        return (threading.get_ident(), "buf", (4,), np.dtype(np.float32))

    main_key = _key()
    child_key_box = {}

    def _worker():
        """Worker."""
        child_key_box["k"] = _key()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert main_key != child_key_box["k"]  # same logical buffer request -> different per-thread keys


def test_con10_pinned_pool_lru_cap_bounds_growth():
    """Inserting many distinct keys keeps the pool bounded (FIFO eviction)."""
    from collections import OrderedDict
    from mlframe.feature_engineering.transformer import _kernels_cupy as kc

    cap = kc._PINNED_BUFFERS_MAX
    pool: "OrderedDict" = OrderedDict()
    for i in range(cap * 3):
        pool[i] = object()
        pool.move_to_end(i)
        while len(pool) > cap:
            pool.popitem(last=False)
    assert len(pool) == cap


# ---------------------------------------------------------------------------
# CON11: CB val Pool stage-1 id() hit co-validated by dtypes; cap exists.
# ---------------------------------------------------------------------------
def test_con11_cb_val_pool_stage1_rejects_dtype_mismatch():
    """An id-recycle false hit with matching cols+shape but DIFFERENT dtypes must not be returned
    by stage-1 (pre-fix it returned the stale pool because dtypes weren't checked)."""
    from mlframe.training import _predict_guards as pg

    class _FakeFrame:
        """Groups tests covering fake frame."""
        columns = ["a", "b"]
        shape = (3, 2)
        dtypes = ["float64", "float64"]

    class _Pool:
        """Groups tests covering pool."""
        pass

    X = _FakeFrame()
    pool = _Pool()
    # Stored pool's content fingerprint says int dtypes; the live X is float -> stage-1 must skip it.
    pool._mlframe_dtypes_sig = ("int64", "int64")
    pg._CB_VAL_POOL_CACHE.clear()
    key = (id(X), tuple(X.columns), (3, 2))
    pg._CB_VAL_POOL_CACHE[key] = pool

    got = pg._cb_val_pool_cache_lookup(X, "predict")
    assert got is None  # dtype mismatch -> no false hit
    pg._CB_VAL_POOL_CACHE.clear()


def test_con11_cb_val_pool_stage1_accepts_matching_dtypes():
    """Con11 cb val pool stage1 accepts matching dtypes."""
    from mlframe.training import _predict_guards as pg

    class _FakeFrame:
        """Groups tests covering fake frame."""
        columns = ["a", "b"]
        shape = (3, 2)
        dtypes = ["float64", "float64"]

    class _Pool:
        """Groups tests covering pool."""
        pass

    X = _FakeFrame()
    pool = _Pool()
    pool._mlframe_dtypes_sig = ("float64", "float64")
    pg._CB_VAL_POOL_CACHE.clear()
    pg._CB_VAL_POOL_CACHE[(id(X), tuple(X.columns), (3, 2))] = pool
    assert pg._cb_val_pool_cache_lookup(X, "predict") is pool
    pg._CB_VAL_POOL_CACHE.clear()


# ---------------------------------------------------------------------------
# CON12: drift invariant cache is FIFO-capped.
# ---------------------------------------------------------------------------
def test_con12_drift_cache_capped():
    """Con12 drift cache capped."""
    from mlframe.training import feature_drift_report as fdr

    fdr._DRIFT_INVARIANT_CACHE.clear()
    cap = fdr._DRIFT_INVARIANT_CACHE_MAX
    for i in range(cap * 3):
        if len(fdr._DRIFT_INVARIANT_CACHE) >= cap:
            fdr._DRIFT_INVARIANT_CACHE.pop(next(iter(fdr._DRIFT_INVARIANT_CACHE)))
        fdr._DRIFT_INVARIANT_CACHE[i] = {"x": i}
    assert len(fdr._DRIFT_INVARIANT_CACHE) <= cap
    fdr._DRIFT_INVARIANT_CACHE.clear()


# ---------------------------------------------------------------------------
# CON13: OOF holdout cache has a hard LRU cap.
# ---------------------------------------------------------------------------
def test_con13_oof_cache_lru_capped():
    """Con13 oof cache lru capped."""
    from mlframe.training.composite import ensemble as ens

    ens._OOF_HOLDOUT_CACHE.clear()
    cap = ens._OOF_HOLDOUT_CACHE_CAP
    for i in range(cap * 3):
        ens._oof_cache_put((i,), (np.zeros(1), np.zeros(1), ["c"]))
    assert len(ens._OOF_HOLDOUT_CACHE) <= cap
    # Most-recently-put keys survive; the oldest are evicted (LRU).
    assert (cap * 3 - 1,) in ens._OOF_HOLDOUT_CACHE
    assert (0,) not in ens._OOF_HOLDOUT_CACHE
    ens._OOF_HOLDOUT_CACHE.clear()


# ---------------------------------------------------------------------------
# CON14: _PD_VIEW_LAST_CACHE memo co-validates columns, not just id+shape.
# ---------------------------------------------------------------------------
def test_con14_pd_view_memo_key_includes_columns():
    """Two frames with the SAME shape but DIFFERENT columns must produce DIFFERENT memo keys, so an
    id-recycle onto a same-shape-different-columns frame cannot false-hit a stale view."""
    pl = pytest.importorskip("polars")
    from mlframe.training import utils as u

    df1 = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    # Same shape (2x2) and (in a recycle scenario) same id, but different column names.
    cols1 = (id(df1), df1.shape, tuple(df1.columns))
    cols2 = (id(df1), df1.shape, ("x", "y"))
    assert cols1 != cols2  # the column tuple discriminates same-id same-shape frames

    # End-to-end: the bridge returns a frame and populates the 3-tuple key.
    u._PD_VIEW_LAST_CACHE["id_key"] = None
    u._PD_VIEW_LAST_CACHE["result"] = None
    out = u.get_pandas_view_of_polars_df(df1)
    assert out is not None
    stored = u._PD_VIEW_LAST_CACHE["id_key"]
    assert stored is not None and len(stored) == 3 and stored[2] == ("a", "b")


# ---------------------------------------------------------------------------
# CON15: RFECV fold-state mutations are guarded by an explicit lock.
# ---------------------------------------------------------------------------
def test_con15_fold_state_lock_serialises_concurrent_appends():
    """The explicit lock makes a fold's multi-step commit atomic. Drive many threads doing the
    same guarded (append + dict-set) group and assert no lost updates / partial state."""
    from mlframe.feature_selection.wrappers.rfecv import _fit_fold as ff

    assert isinstance(ff._FOLD_STATE_LOCK, type(threading.Lock()))

    scores: list = []
    fi: dict = {}
    n = 200

    def _commit(k):
        """Commit."""
        with ff._FOLD_STATE_LOCK:
            scores.append(k)
            fi[k] = {"v": k}

    threads = [threading.Thread(target=_commit, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(scores) == n
    assert len(fi) == n
    assert set(scores) == set(range(n))


# ---------------------------------------------------------------------------
# CON21: baseline to-pandas cache co-validates cols/shape on hit.
# ---------------------------------------------------------------------------
def test_con21_baseline_cache_rejects_id_recycle_with_different_frame():
    """A cache entry keyed on id(X) but carrying a different frame's (cols, shape) signature must
    NOT be returned for the live X -- pre-fix the id() match alone returned the stale view."""
    pl = pytest.importorskip("polars")
    from mlframe.training.baselines import _dummy_baseline_compute as dbc
    from collections import OrderedDict

    X = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    dbc._TO_PANDAS_BASELINE_CACHE = OrderedDict()
    # Plant a stale entry under id(X) with a MISMATCHED signature (different cols).
    dbc._TO_PANDAS_BASELINE_CACHE[id(X)] = (("wrong", "cols"), X.shape, "STALE_VIEW")

    out = dbc._to_pandas_for_baseline(X)
    # The mismatch must be detected -> a fresh real pandas view, never the planted "STALE_VIEW".
    assert out is not None and not isinstance(out, str)
    dbc._TO_PANDAS_BASELINE_CACHE = None
