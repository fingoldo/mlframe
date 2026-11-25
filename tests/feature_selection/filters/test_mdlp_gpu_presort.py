"""The device half of the MDLP edge search gives the host passes' arrays bit for bit, and the edges of a call do not depend on where the presort ran."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:  # audit-exempt: except_skip_masks_call_under_test -- optional CUDA device guard: the probe decides whether the module can run, it is not the code under test
    cp.cuda.runtime.getDeviceCount()
except Exception:  # no device: nothing to compare on
    pytest.skip("no CUDA device", allow_module_level=True)

from mlframe.feature_selection.filters import _adaptive_nbins_columns as columns_module
from mlframe.feature_selection.filters import _mdlp_gpu_presort as G
from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges
from mlframe.feature_selection.filters._mdlp_validated_split import _dedupe_xy
from mlframe.feature_selection.filters.discretization._quantile_radix import sort_xy, sorted_unique_with_counts


def _labels(n, classes, seed=0):
    """Target codes as the MDLP encoder returns them: small non-negative integers."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, classes, n).astype(np.int8) if classes > 2 else (rng.random(n) < 0.4).astype(np.int8)


def _columns(n, seed=1):
    """One column of every kind the search meets, alternating float32 and float32-valued double precision."""
    rng = np.random.default_rng(seed)
    base = [
        rng.standard_normal(n),
        np.round(rng.standard_normal(n), 1),  # heavy ties and duplicate pairs
        np.where(rng.random(n) < 0.5, 0.0, -0.0) + (rng.random(n) < 0.1) * rng.standard_normal(n),  # both zeros
        np.round(rng.standard_normal(n) * 3),  # a few distinct values
        np.zeros(n),  # constant
        rng.standard_cauchy(n),
        np.arange(n, dtype=float),  # strictly increasing: no ties, no duplicates
        np.where(rng.random(n) < 0.99, 1.0, rng.standard_normal(n)),  # one value for 99 percent of the rows
    ]
    return [b.astype(np.float32) if i % 2 == 0 else b.astype(np.float32).astype(np.float64) for i, b in enumerate(base)]


def _host(col, labels):
    """What the host passes return for a finite column: the sorted pair, the runs and the deduplicated pairs."""
    xs, ys = sort_xy(col, labels)
    uniq, counts = sorted_unique_with_counts(xs)
    xd, yd = _dedupe_xy(xs, ys)
    return xs, ys, uniq.astype(col.dtype, copy=False), counts, xd, yd, (xd is xs and yd is ys)


@pytest.mark.parametrize("classes", [2, 5])
def test_the_block_equals_the_host_passes_bit_for_bit(classes):
    """Sorted values (zeros of both signs included), labels, distinct values, counts and the deduplicated pairs equal the host's for every kind of column."""
    n = 9000
    labels = _labels(n, classes)
    cols = _columns(n)
    got = G.presort_block(cols, labels)
    assert all(g is not None for g in got)
    for col, g in zip(cols, got):
        xs, ys, uniq, counts, xd, yd, same = _host(col, labels)
        assert np.array_equal(g.xs.view(np.int64), xs.view(np.int64)) and g.ys.dtype == ys.dtype and np.array_equal(g.ys, ys)
        assert g.uniq.dtype == uniq.dtype and np.array_equal(g.uniq, uniq) and np.array_equal(g.counts, counts)
        if same:
            assert g.dedup is None
        else:
            assert g.dedup is not None and np.array_equal(g.dedup[0], xd) and np.array_equal(g.dedup[1], yd) and g.dedup[1].dtype == yd.dtype


def test_a_column_the_device_does_not_take_is_none():
    """A column with NaN or infinity, a double that is not a float32 value, an integer column and a column of another length give ``None``; the rest of the block is still sorted."""
    n = 6000
    labels = _labels(n, 2)
    ok = np.random.default_rng(2).standard_normal(n).astype(np.float32)
    with_nan = ok.copy()
    with_nan[3] = np.nan
    with_inf = ok.copy()
    with_inf[5] = np.inf
    inexact = ok.astype(np.float64) + 1e-9
    got = G.presort_block([with_nan, ok, with_inf, inexact, np.arange(n), ok[:-1]], labels)
    assert [g is None for g in got] == [True, False, True, True, True, True]


def test_applicable_needs_keyable_labels_enough_rows_and_a_device(monkeypatch):
    """The device presort is off by default; ``auto`` takes it from the row floor, ``1`` drops the floor, and negative, float and huge labels are declined in every mode."""
    labels = _labels(200_000, 2)
    monkeypatch.delenv("MLFRAME_MDLP_GPU_PRESORT", raising=False)
    assert G.mode() == "0" and not G.applicable(200_000, labels)
    monkeypatch.setenv("MLFRAME_MDLP_GPU_PRESORT", "auto")
    assert G.applicable(200_000, labels)
    assert not G.applicable(50_000, labels[:50_000])
    monkeypatch.setenv("MLFRAME_MDLP_GPU_PRESORT", "1")
    assert G.applicable(50_000, labels[:50_000])
    assert not G.applicable(1000, np.array([0, -1, 2]) * np.ones(1000, dtype=np.int64)[:3]) and not G.applicable(1000, labels[:1000].astype(float))
    assert not G.applicable(1000, np.full(1000, 1 << 21, dtype=np.int64))
    monkeypatch.setenv("MLFRAME_MDLP_GPU_PRESORT", "0")
    assert not G.applicable(200_000, labels)


def test_the_supply_is_seen_by_its_own_column_and_thread_only():
    """Inside ``supplied`` the column and the supplied pair are recognised by identity; another column, another pair and another thread see nothing."""
    import threading

    n = 5000
    labels = _labels(n, 2)
    col = np.random.default_rng(4).standard_normal(n).astype(np.float32)
    presorted = G.presort_block([col], labels)[0]
    seen = []
    with G.supplied(col, presorted):
        got = G.supplied_presort(col)
        assert got is not None and got[0] is presorted.xs and G.supplied_presort(col.copy()) is None
        assert G.supplied_dedupe(presorted.xs, presorted.ys) is not None and G.supplied_dedupe(presorted.xs.copy(), presorted.ys) is None
        worker = threading.Thread(target=lambda: seen.append(G.supplied_presort(col)))
        worker.start()
        worker.join()
    assert seen == [None] and G.supplied_presort(col) is None
    with G.supplied(col, None):
        assert G.supplied_presort(col) is None


@pytest.mark.parametrize("classes", [2, 4])
def test_edges_of_a_call_do_not_depend_on_where_the_presort_ran(monkeypatch, classes):
    """``per_feature_edges`` with the MDLP search on a frame with ties, zeros, a NaN column and a constant gives the same edge arrays with the device presort and without it; the device took the columns it can."""
    n = 12000
    labels = _labels(n, classes, seed=7)
    cols = _columns(n, seed=8)
    nan_col = cols[0].copy()
    nan_col[10] = np.nan
    frame = np.column_stack([*[c.astype(np.float64) for c in cols], nan_col.astype(np.float64)])
    y = labels
    monkeypatch.setenv("MLFRAME_MDLP_GPU_PRESORT", "0")
    columns_module._EDGE_MEMO.clear()
    host = per_feature_edges(frame, y, method="mdlp", n_jobs=4)
    taken = []
    real = G.presort_block
    monkeypatch.setattr(G, "presort_block", lambda c, l: (lambda r: (taken.append(sum(x is not None for x in r)), r)[1])(real(c, l)))
    monkeypatch.setenv("MLFRAME_MDLP_GPU_PRESORT", "1")
    columns_module._EDGE_MEMO.clear()
    device = per_feature_edges(frame, y, method="mdlp", n_jobs=4)
    assert sum(taken) >= 6, taken
    assert len(host) == len(device) and all(np.array_equal(a, b) and np.asarray(a).dtype == np.asarray(b).dtype for a, b in zip(host, device))


def test_a_column_whose_edges_the_memo_holds_is_not_sorted_again(monkeypatch):
    """The second call over the same frame finds every column in the content memo, so no block goes to the device."""
    n = 8000
    cols = _columns(n, seed=9)
    frame = np.column_stack([c.astype(np.float64) for c in cols])
    y = _labels(n, 2, seed=3)
    monkeypatch.setenv("MLFRAME_MDLP_GPU_PRESORT", "1")
    columns_module._EDGE_MEMO.clear()
    first = per_feature_edges(frame, y, method="mdlp", n_jobs=4)
    sent = []
    real = G.presort_block
    monkeypatch.setattr(G, "presort_block", lambda c, l: (sent.append(len(c)), real(c, l))[1])
    second = per_feature_edges(frame, y, method="mdlp", n_jobs=4)
    assert sent == [] and all(np.array_equal(a, b) for a, b in zip(first, second))
